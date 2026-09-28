from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.gate0.build_candidates import (
    budget_to_override,
    generate_lu_boundary_swaps,
    load_lu_marginal_slice,
)
from evaluation.gate0.metrics import (
    continuation_nll,
    gate0_task_score,
    jensen_shannon,
    probability_overlap,
    teacher_to_candidate_kl,
    topk_agreement,
)
from evaluation.gate0.minigate_runtime import (
    best_gold_answer_nll,
    build_full_prompt_state,
    greedy_generate,
    prefill_context,
    replay_query_probe,
)
from kvpress import LUPress, SnapKVPress


def parse_args():
    p = argparse.ArgumentParser(description="Run the 24-prompt CausalDuel-KV Gate-0 MiniGate.")
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--budget-curve-path", required=True)
    p.add_argument("--marginal-profile-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/gate0/minigate/raw"))
    p.add_argument("--compression-ratio", type=float, default=0.80)
    p.add_argument("--swap-size", type=int, default=16)
    p.add_argument("--num-swaps", type=int, default=6)
    p.add_argument("--probe-lens", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--limit", type=int, default=None, help="Engineering pilot only.")
    p.add_argument("--skip-generation", action="store_true", help="Engineering pilot only.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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
    scorer = SnapKVPress(compression_ratio=compression_ratio)
    press = LUPress(
        press=scorer,
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
            raise RuntimeError(f"no LU-KV budget for layer {layer_idx}")
        budget.append([int(v) for v in counts.tolist()])
    return budget


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def tokenize_pair(tokenizer, row):
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
    if query_ids.shape[1] < 2:
        raise ValueError(f"{row['id']} query is too short for replay")
    return context_ids, query_ids


def score_probe(ref_logits, cand_logits, targets):
    return {
        "overlap": float(probability_overlap(ref_logits, cand_logits).item()),
        "kl": float(teacher_to_candidate_kl(ref_logits, cand_logits).item()),
        "js": float(jensen_shannon(ref_logits, cand_logits).item()),
        "top10_agreement": float(topk_agreement(ref_logits, cand_logits, k=10).item()),
        "probe_nll": float(continuation_nll(cand_logits, targets).item()),
    }


def evaluate_context_cache(
    *,
    model,
    tokenizer,
    context_cache,
    query_ids,
    answers,
    family,
    max_new_tokens,
    probe_lens,
    reference_probes,
    skip_generation,
):
    prompt_state = build_full_prompt_state(model, context_cache, query_ids)
    answer_nll_value, answer_nll_all = best_gold_answer_nll(
        model,
        tokenizer,
        prompt_state,
        answers,
    )

    prediction = None
    task_score = None
    if not skip_generation:
        prediction = greedy_generate(
            model,
            tokenizer,
            prompt_state,
            max_new_tokens=max_new_tokens,
        )
        task_score = gate0_task_score(family, prediction, answers)

    del prompt_state

    probes = {}
    for probe_len in probe_lens:
        probe = replay_query_probe(
            model,
            context_cache,
            query_ids,
            probe_len=probe_len,
        )
        ref = reference_probes[probe_len]
        scores = score_probe(ref.logits, probe.logits, probe.targets)
        scores["actual_probe_tokens"] = probe.probe_tokens
        probes[str(probe_len)] = scores
        del probe

    return {
        "answer_nll": answer_nll_value,
        "all_gold_answer_nll": answer_nll_all,
        "prediction": prediction,
        "task_score": task_score,
        "probes": probes,
    }


def candidate_meta(candidate):
    return {
        "name": candidate.name,
        "category": candidate.category,
        "donor": candidate.donor.__dict__,
        "receiver": candidate.receiver.__dict__,
        "amount": candidate.amount,
        "lu_remove_cost": candidate.lu_remove_cost,
        "lu_next_gain": candidate.lu_next_gain,
        "lu_marginal_delta": candidate.lu_marginal_delta,
    }


def add_lu_deltas(row, baseline, probe_lens):
    row["answer_nll_gain_vs_lu"] = baseline["answer_nll"] - row["answer_nll"]
    if row["task_score"] is not None and baseline["task_score"] is not None:
        row["task_score_gain_vs_lu"] = row["task_score"] - baseline["task_score"]
    else:
        row["task_score_gain_vs_lu"] = None

    for probe_len in probe_lens:
        key = str(probe_len)
        cur = row["probes"][key]
        base = baseline["probes"][key]
        cur["delta_overlap_vs_lu"] = cur["overlap"] - base["overlap"]
        cur["kl_gain_vs_lu"] = base["kl"] - cur["kl"]
        cur["js_gain_vs_lu"] = base["js"] - cur["js"]
        cur["top10_gain_vs_lu"] = cur["top10_agreement"] - base["top10_agreement"]
        cur["probe_nll_gain_vs_lu"] = base["probe_nll"] - cur["probe_nll"]


def main():
    args = parse_args()
    if sorted(set(args.probe_lens)) != sorted(args.probe_lens):
        raise ValueError("--probe-lens must be unique")
    if any(x < 2 for x in args.probe_lens):
        raise ValueError("all probe lengths must be >= 2")

    rows = read_jsonl(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.model, args.dtype)
    press = make_press(args.budget_curve_path, args.compression_ratio)
    marginal = load_lu_marginal_slice(
        args.marginal_profile_path,
        compression_ratio=args.compression_ratio,
        expected_step_tokens=args.swap_size,
    )

    if marginal.sink_size != press.sink or marginal.window_size != press.window:
        raise ValueError(
            "marginal/runtime sink-window mismatch: "
            f"{marginal.sink_size}/{marginal.window_size} vs "
            f"{press.sink}/{press.window}"
        )

    for prompt_idx, row in enumerate(rows):
        out_path = args.output_dir / f"{row['id']}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[MiniGate] skip existing {row['id']}")
            continue

        print(
            f"\n[MiniGate] {prompt_idx + 1}/{len(rows)} "
            f"{row['id']} family={row['family']} task={row['task']}"
        )

        context_ids, query_ids = tokenize_pair(tokenizer, row)
        context_len = int(context_ids.shape[1])

        # FullKV teacher and offline full-cache reference.
        full_context_cache = prefill_context(model, context_ids, press=None)
        reference_probes = {
            probe_len: replay_query_probe(
                model,
                full_context_cache,
                query_ids,
                probe_len=probe_len,
            )
            for probe_len in args.probe_lens
        }

        max_new_tokens = min(
            int(row.get("max_new_tokens", args.max_new_tokens)),
            int(args.max_new_tokens),
        )

        fullkv = evaluate_context_cache(
            model=model,
            tokenizer=tokenizer,
            context_cache=full_context_cache,
            query_ids=query_ids,
            answers=row["answers"],
            family=row["family"],
            max_new_tokens=max_new_tokens,
            probe_lens=args.probe_lens,
            reference_probes=reference_probes,
            skip_generation=args.skip_generation,
        )
        fullkv["name"] = "fullkv"

        baseline_budget = curve_budget(press, model, context_len)
        min_keep = min(context_len, press.sink + press.window)

        candidates, donor_pool, receiver_pool = generate_lu_boundary_swaps(
            baseline_budget,
            remove_cost=marginal.remove_cost,
            next_gain=marginal.next_gain,
            amount=args.swap_size,
            num_candidates=args.num_swaps,
            min_keep=min_keep,
            max_keep=context_len,
        )
        if len(candidates) != args.num_swaps:
            raise RuntimeError(
                f"{row['id']}: requested {args.num_swaps} candidates, got {len(candidates)}"
            )

        # LU baseline.
        press.set_keep_counts_override(budget_to_override(baseline_budget, None))
        lu_context_cache = prefill_context(model, context_ids, press=press)
        lu = evaluate_context_cache(
            model=model,
            tokenizer=tokenizer,
            context_cache=lu_context_cache,
            query_ids=query_ids,
            answers=row["answers"],
            family=row["family"],
            max_new_tokens=max_new_tokens,
            probe_lens=args.probe_lens,
            reference_probes=reference_probes,
            skip_generation=args.skip_generation,
        )
        lu["name"] = "lu_baseline"

        conditions = [fullkv, lu]

        for candidate_idx, candidate in enumerate(candidates):
            print(
                f"  candidate {candidate_idx + 1}/{len(candidates)} "
                f"{candidate.name} {candidate.category} "
                f"LUdelta={candidate.lu_marginal_delta:.3e}"
            )

            override = budget_to_override(
                baseline_budget,
                candidate,
                min_keep=min_keep,
                max_keep=context_len,
            )
            press.set_keep_counts_override(override)
            candidate_cache = prefill_context(model, context_ids, press=press)

            result = evaluate_context_cache(
                model=model,
                tokenizer=tokenizer,
                context_cache=candidate_cache,
                query_ids=query_ids,
                answers=row["answers"],
                family=row["family"],
                max_new_tokens=max_new_tokens,
                probe_lens=args.probe_lens,
                reference_probes=reference_probes,
                skip_generation=args.skip_generation,
            )
            result.update(candidate_meta(candidate))
            add_lu_deltas(result, lu, args.probe_lens)
            conditions.append(result)

            del candidate_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        payload = {
            "protocol": "CausalDuel-KV Gate-0 MiniGate",
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
            "swap_size": args.swap_size,
            "num_swaps": len(candidates),
            "probe_lens": args.probe_lens,
            "candidate_generation": "global_lu_boundary_marginal",
            "marginal_profile": {
                "path": str(args.marginal_profile_path),
                "compression_ratio": marginal.global_compression_ratio,
                "step_tokens": marginal.marginal_step_tokens,
                "calibration_pairs": marginal.calibration_pairs,
            },
            "donor_pool": [u.__dict__ for u in donor_pool],
            "receiver_pool": [u.__dict__ for u in receiver_pool],
            "answers": row["answers"],
            "conditions": conditions,
        }

        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[MiniGate] saved {out_path}")

        del full_context_cache, lu_context_cache, reference_probes
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[MiniGate] completed requested prompts.")


if __name__ == "__main__":
    main()
