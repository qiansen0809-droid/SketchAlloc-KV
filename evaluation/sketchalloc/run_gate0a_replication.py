"""Run the preregistered fixed-action Gate 0A replication.

The action dictionary is loaded from a frozen JSON file. Candidate selection
never uses the current prompt's answer or model outputs. Gold answers are used
only after LU/candidate budgets are fixed, as offline Answer-NLL labels.
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
        default=Path("results/sketchalloc/gate0a/confirm24_raw"),
    )
    p.add_argument("--compression-ratio", type=float, default=0.80)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--limit", type=int, default=None, help="Execution smoke test only; legality still checks all prompts.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_actions(path: Path) -> tuple[list[SwapCandidate], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("actions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("frozen action file has no actions")

    actions: list[SwapCandidate] = []
    seen: set[str] = set()
    for row in rows:
        candidate = SwapCandidate(
            donor=Unit(int(row["donor_layer"]), int(row["donor_head"])),
            receiver=Unit(int(row["receiver_layer"]), int(row["receiver_head"])),
            amount=int(row["amount"]),
            category="fixed_confirmatory",
        )
        expected = str(row["action_id"])
        stable = f"L{candidate.donor.layer}H{candidate.donor.head}->L{candidate.receiver.layer}H{candidate.receiver.head}@{candidate.amount}"
        if stable != expected:
            raise ValueError(f"action ID mismatch: {expected} != {stable}")
        if stable in seen:
            raise ValueError(f"duplicate frozen action: {stable}")
        seen.add(stable)
        actions.append(candidate)
    return actions, payload


def curve_budget_from_array(curve: np.ndarray, compression_ratio: float, seq_len: int) -> list[list[int]]:
    target_idx = int(round(compression_ratio * 100)) - 1
    target_idx = max(0, min(98, target_idx))
    slice_ = np.asarray(curve[target_idx], dtype=np.float64)
    budgets: list[list[int]] = []
    for local_prune_ratios in slice_:
        ideal = (1.0 - local_prune_ratios) * seq_len
        total_target = int(np.rint(ideal.sum()))
        keep = np.floor(ideal).astype(np.int64)
        remainder = total_target - int(keep.sum())
        if remainder > 0:
            fractional = ideal - keep
            order = np.argsort(-fractional, kind="stable")
            keep[order[: min(remainder, keep.size)]] += 1
        keep = np.clip(keep, 1, seq_len)
        budgets.append([int(x) for x in keep.tolist()])
    return budgets


def validate_all_legality(rows: list[dict], actions: list[SwapCandidate], curve_path: str, compression_ratio: float) -> None:
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
                failures.append(f"{row['id']} {action.name}: {exc}")
    if failures:
        preview = "\n".join(failures[:20])
        raise ValueError(
            f"frozen action dictionary is illegal for {len(failures)} prompt/action cells. "
            f"No replication labels were computed. First failures:\n{preview}"
        )
    print(f"[Gate0A] legality audit passed: {len(rows)} prompts x {len(actions)} actions")


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
    budget: list[list[int]] = []
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


def candidate_payload(candidate: SwapCandidate, answer_nll: float, all_scores: list[float], lu_nll: float) -> dict:
    action_id = (
        f"L{candidate.donor.layer}H{candidate.donor.head}"
        f"->L{candidate.receiver.layer}H{candidate.receiver.head}"
        f"@{candidate.amount}"
    )
    return {
        "name": action_id,
        "category": "fixed_confirmatory",
        "donor": {"layer": candidate.donor.layer, "head": candidate.donor.head},
        "receiver": {"layer": candidate.receiver.layer, "head": candidate.receiver.head},
        "amount": candidate.amount,
        "answer_nll": answer_nll,
        "all_gold_answer_nll": all_scores,
        "answer_nll_gain_vs_lu": lu_nll - answer_nll,
    }


def main() -> None:
    args = parse_args()
    all_rows = read_jsonl(args.data)
    actions, action_manifest = load_actions(args.actions)

    if abs(float(action_manifest.get("compression_ratio", args.compression_ratio)) - args.compression_ratio) > 1e-9:
        raise ValueError("frozen action compression ratio does not match runtime")
    if any(action.amount != int(action_manifest.get("swap_size", action.amount)) for action in actions):
        raise ValueError("frozen action swap size does not match manifest")

    # This happens before model loading and before any answer-NLL label is computed.
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

    for prompt_idx, row in enumerate(rows):
        out_path = args.output_dir / f"{row['id']}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[Gate0A] skip existing {row['id']}")
            continue

        print(
            f"\n[Gate0A] {prompt_idx + 1}/{len(rows)} "
            f"{row['id']} family={row['family']} task={row['task']}"
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
                f"{row['id']}: tokenizer length drift {context_len} != stored {row['context_tokens']}"
            )

        baseline = curve_budget(press, model, context_len)
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
        conditions: list[dict] = [
            {
                "name": "lu_baseline",
                "answer_nll": lu_nll,
                "all_gold_answer_nll": lu_all,
            }
        ]

        for action_idx, action in enumerate(actions):
            print(f"  fixed action {action_idx + 1}/{len(actions)} {action.name}")
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
            conditions.append(candidate_payload(action, cand_nll, cand_all, lu_nll))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        payload = {
            "protocol": "SketchAlloc-KV Gate 0A fixed-action confirmatory replication",
            "model": args.model,
            "id": row["id"],
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
            "swap_size": int(action_manifest.get("swap_size", actions[0].amount)),
            "num_swaps": len(actions),
            "candidate_generation": "frozen_common5_before_replication_labels",
            "action_manifest": str(args.actions),
            "answers": row["answers"],
            "conditions": conditions,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[Gate0A] saved {out_path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[Gate0A] completed requested prompts.")


if __name__ == "__main__":
    main()
