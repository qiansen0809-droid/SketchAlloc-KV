"""Run the complete 48 x 12 half-scale Gate 0B utility matrix.

For every prompt, score the frozen LU baseline and every frozen action using
Answer-NLL. Gold answers are offline labels only; they never affect action
selection. The runner audits all 48 x 12 action cells for legality before model
loading and re-checks the LU curve-derived baseline before each prompt.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.sketchalloc.build_candidates import (
    SwapCandidate,
    Unit,
    apply_swap,
    budget_to_override,
)
from evaluation.sketchalloc.gate0a_runtime import (
    best_gold_answer_nll,
    build_full_prompt_state,
    prefill_context,
)
from evaluation.sketchalloc.run_gate0a_replication import curve_budget_from_array
from kvpress import LUPress, SnapKVPress


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--actions", type=Path, required=True)
    p.add_argument("--budget-curve-path", required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sketchalloc/gate0b/half48_raw"),
    )
    p.add_argument("--compression-ratio", type=float, default=0.80)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--limit", type=int, default=None, help="Engineering smoke only; legality audits all rows.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_actions(path: Path) -> tuple[list[SwapCandidate], dict, dict[str, dict]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("actions")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("Gate 0B half-scale action file must contain exactly 12 actions")

    actions: list[SwapCandidate] = []
    metadata: dict[str, dict] = {}
    seen: set[str] = set()
    for row in rows:
        candidate = SwapCandidate(
            donor=Unit(int(row["donor_layer"]), int(row["donor_head"])),
            receiver=Unit(int(row["receiver_layer"]), int(row["receiver_head"])),
            amount=int(row["amount"]),
            lu_remove_cost=float(row["lu_remove_cost"]),
            lu_next_gain=float(row["lu_next_gain"]),
            lu_marginal_delta=float(row["lu_marginal_delta"]),
            category=str(row["category"]),
        )
        action_id = (
            f"L{candidate.donor.layer}H{candidate.donor.head}"
            f"->L{candidate.receiver.layer}H{candidate.receiver.head}"
            f"@{candidate.amount}"
        )
        if action_id != str(row["action_id"]):
            raise ValueError(f"action ID mismatch: {row['action_id']} != {action_id}")
        if action_id in seen:
            raise ValueError(f"duplicate action ID: {action_id}")
        seen.add(action_id)
        actions.append(candidate)
        metadata[action_id] = dict(row)

    return actions, manifest, metadata


def action_id(candidate: SwapCandidate) -> str:
    return (
        f"L{candidate.donor.layer}H{candidate.donor.head}"
        f"->L{candidate.receiver.layer}H{candidate.receiver.head}"
        f"@{candidate.amount}"
    )


def validate_all_legality(
    rows: list[dict],
    actions: list[SwapCandidate],
    curve_path: str,
    compression_ratio: float,
) -> None:
    curve = np.load(curve_path)
    failures: list[str] = []
    for row in rows:
        seq_len = int(row["context_tokens"])
        baseline = curve_budget_from_array(curve, compression_ratio, seq_len)
        min_keep = min(seq_len, 4 + 32)
        for action in actions:
            try:
                apply_swap(
                    baseline,
                    action,
                    min_keep=min_keep,
                    max_keep=seq_len,
                )
            except Exception as exc:
                failures.append(f"{row['id']} {action_id(action)}: {exc}")

    if failures:
        preview = "\n".join(failures[:20])
        raise ValueError(
            f"Gate 0B frozen actions are illegal for {len(failures)} cells. "
            f"No Gate 0B labels were computed. First failures:\n{preview}"
        )

    print(
        f"[Gate0B] legality audit passed: {len(rows)} prompts x "
        f"{len(actions)} actions = {len(rows) * len(actions)} cells"
    )


def load_model(name: str, dtype_name: str):
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def make_press(curve_path: str, compression_ratio: float) -> LUPress:
    press = LUPress(
        press=SnapKVPress(compression_ratio=compression_ratio),
        budget_curve_path=curve_path,
        sink=4,
        window=32,
    )
    press.compression_ratio = compression_ratio
    press._post_setup_init()
    return press


def curve_budget(press: LUPress, model, seq_len: int) -> list[list[int]]:
    budget = []
    for layer_idx in range(model.config.num_hidden_layers):
        counts = press.get_keep_counts(
            layer_idx=layer_idx,
            num_heads=model.config.num_key_value_heads,
            seq_len=seq_len,
            device=torch.device("cpu"),
        )
        if counts is None:
            raise RuntimeError(f"no LU budget for layer {layer_idx}")
        budget.append([int(v) for v in counts.tolist()])
    return budget


def score_condition(model, tokenizer, press, context_ids, query_ids, answers, override):
    press.set_keep_counts_override(override)
    context_cache = prefill_context(model, context_ids, press=press)
    state = build_full_prompt_state(model, context_cache, query_ids)
    best, all_scores = best_gold_answer_nll(model, tokenizer, state, answers)
    del state, context_cache
    return best, all_scores


def main() -> None:
    args = parse_args()
    all_rows = read_jsonl(args.data)
    actions, action_manifest, action_meta = load_actions(args.actions)

    if len(all_rows) != 48:
        raise ValueError(f"Gate 0B half-scale expects 48 prompts; got {len(all_rows)}")
    split_counts = {}
    for row in all_rows:
        split = str(row.get("split", ""))
        split_counts[split] = split_counts.get(split, 0) + 1
    if split_counts != {"discovery": 24, "calibration": 12, "heldout": 12}:
        raise ValueError(f"unexpected Gate 0B split counts: {split_counts}")

    frozen_ratio = float(action_manifest.get("compression_ratio", args.compression_ratio))
    if abs(frozen_ratio - args.compression_ratio) > 1e-9:
        raise ValueError("frozen Gate 0B action compression ratio does not match runtime")

    validate_all_legality(
        all_rows,
        actions,
        args.budget_curve_path,
        args.compression_ratio,
    )

    rows = all_rows if args.limit is None else all_rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, args.dtype)
    press = make_press(args.budget_curve_path, args.compression_ratio)
    audit_curve = np.load(args.budget_curve_path)

    for prompt_idx, row in enumerate(rows):
        out_path = args.output_dir / f"{row['id']}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[Gate0B] skip existing {row['id']}")
            continue

        print(
            f"\n[Gate0B] {prompt_idx + 1}/{len(rows)} "
            f"{row['id']} split={row['split']} family={row['family']} task={row['task']}"
        )

        context_ids = tokenizer(
            row["context"],
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids
        query_ids = tokenizer(
            row["question"] + row["answer_prefix"],
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        context_len = int(context_ids.shape[1])
        if context_len != int(row["context_tokens"]):
            raise ValueError(
                f"{row['id']}: tokenizer length drift {context_len} != {row['context_tokens']}"
            )

        press.clear_keep_counts_override()
        baseline = curve_budget(press, model, context_len)
        expected = curve_budget_from_array(audit_curve, args.compression_ratio, context_len)
        if baseline != expected:
            raise RuntimeError(
                f"{row['id']}: runtime LU baseline differs from frozen curve audit"
            )
        min_keep = min(context_len, press.sink + press.window)

        lu_nll, lu_all = score_condition(
            model,
            tokenizer,
            press,
            context_ids,
            query_ids,
            row["answers"],
            budget_to_override(baseline, None),
        )
        conditions = [
            {
                "name": "lu_baseline",
                "answer_nll": lu_nll,
                "all_gold_answer_nll": lu_all,
            }
        ]

        for idx, action in enumerate(actions):
            aid = action_id(action)
            print(f"  action {idx + 1}/12 {aid} [{action.category}]")
            override = budget_to_override(
                baseline,
                action,
                min_keep=min_keep,
                max_keep=context_len,
            )
            cand_nll, cand_all = score_condition(
                model,
                tokenizer,
                press,
                context_ids,
                query_ids,
                row["answers"],
                override,
            )
            meta = action_meta[aid]
            conditions.append(
                {
                    "name": aid,
                    "category": str(meta["category"]),
                    "donor": {
                        "layer": action.donor.layer,
                        "head": action.donor.head,
                    },
                    "receiver": {
                        "layer": action.receiver.layer,
                        "head": action.receiver.head,
                    },
                    "amount": action.amount,
                    "lu_remove_cost": float(meta["lu_remove_cost"]),
                    "lu_next_gain": float(meta["lu_next_gain"]),
                    "lu_marginal_delta": float(meta["lu_marginal_delta"]),
                    "answer_nll": cand_nll,
                    "all_gold_answer_nll": cand_all,
                    "answer_nll_gain_vs_lu": lu_nll - cand_nll,
                }
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        payload = {
            "protocol": "SketchAlloc-KV Gate 0B half-scale structural matrix",
            "model": args.model,
            "id": row["id"],
            "split": row["split"],
            "family": row["family"],
            "task": row["task"],
            "source_dataset": row["source_dataset"],
            "source_config": row["source_config"],
            "source_revision": row["source_revision"],
            "source_index": row["source_index"],
            "context_sha256": row.get("context_sha256"),
            "context_tokens": context_len,
            "query_tokens": int(query_ids.shape[1]),
            "compression_ratio": args.compression_ratio,
            "swap_size": int(action_manifest["swap_size"]),
            "num_swaps": len(actions),
            "candidate_generation": "frozen_gate0b_half12_before_labels",
            "action_manifest": str(args.actions),
            "answers": row["answers"],
            "conditions": conditions,
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[Gate0B] saved {out_path}")

        press.clear_keep_counts_override()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[Gate0B] completed requested prompts.")


if __name__ == "__main__":
    main()
