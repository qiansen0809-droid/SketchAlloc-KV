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
    jensen_shannon,
    probability_overlap,
    teacher_to_candidate_kl,
    topk_agreement,
)
from evaluation.gate0.minigate_runtime import (
    best_gold_answer_nll,
    build_full_prompt_state,
    greedy_reference_trace,
    prefill_context,
    teacher_force_trace,
)
from kvpress import LUPress, SnapKVPress


def parse_args():
    p = argparse.ArgumentParser(
        description="Gate-0-v2 discovery: future-facing answer-onset behavioral duel."
    )
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--budget-curve-path", required=True)
    p.add_argument("--marginal-profile-path", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gate0/answer_onset/discovery24"),
    )
    p.add_argument("--compression-ratio", type=float, default=0.80)
    p.add_argument("--swap-size", type=int, default=16)
    p.add_argument("--num-swaps", type=int, default=6)
    p.add_argument("--trace-lens", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--limit", type=int, default=None)
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
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
    return context_ids, query_ids


def score_trace(ref_logits, cand_logits, pseudo_ids):
    return {
        "overlap": float(probability_overlap(ref_logits, cand_logits).item()),
        "kl": float(teacher_to_candidate_kl(ref_logits, cand_logits).item()),
        "js": float(jensen_shannon(ref_logits, cand_logits).item()),
        "top10_agreement": float(
            topk_agreement(ref_logits, cand_logits, k=10).item()
        ),
        "pseudo_nll": float(
            continuation_nll(cand_logits, pseudo_ids).item()
        ),
    }


def evaluate_state(
    *,
    model,
    tokenizer,
    context_cache,
    query_ids,
    answers,
    pseudo_ids,
    full_trace,
    trace_lens,
):
    state = build_full_prompt_state(model, context_cache, query_ids)
    answer_nll_value, answer_nll_all = best_gold_answer_nll(
        model,
        tokenizer,
        state,
        answers,
    )
    trace = teacher_force_trace(
        model,
        state,
        pseudo_ids,
    )

    metrics = {}
    for length in trace_lens:
        ref = full_trace[:, :length]
        cand = trace[:, :length]
        ids = pseudo_ids[:, :length]
        metrics[str(length)] = score_trace(ref, cand, ids)

    del state, trace
    return {
        "answer_nll": answer_nll_value,
        "all_gold_answer_nll": answer_nll_all,
        "answer_trace": metrics,
    }


def add_deltas(row, baseline, trace_lens):
    row["answer_nll_gain_vs_lu"] = baseline["answer_nll"] - row["answer_nll"]
    for length in trace_lens:
        key = str(length)
        cur = row["answer_trace"][key]
        base = baseline["answer_trace"][key]
        cur["delta_overlap_vs_lu"] = cur["overlap"] - base["overlap"]
        cur["kl_gain_vs_lu"] = base["kl"] - cur["kl"]
        cur["js_gain_vs_lu"] = base["js"] - cur["js"]
        cur["top10_gain_vs_lu"] = (
            cur["top10_agreement"] - base["top10_agreement"]
        )
        cur["pseudo_nll_gain_vs_lu"] = (
            base["pseudo_nll"] - cur["pseudo_nll"]
        )


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


def main():
    args = parse_args()
    trace_lens = sorted(set(int(x) for x in args.trace_lens))
    if not trace_lens or trace_lens[0] <= 0:
        raise ValueError("--trace-lens must contain positive integers")

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

    max_trace = max(trace_lens)

    for prompt_idx, row in enumerate(rows):
        out_path = args.output_dir / f"{row['id']}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[AnswerOnset] skip existing {row['id']}")
            continue

        print(
            f"\n[AnswerOnset] {prompt_idx + 1}/{len(rows)} "
            f"{row['id']} family={row['family']} task={row['task']}"
        )

        context_ids, query_ids = tokenize_pair(tokenizer, row)
        context_len = int(context_ids.shape[1])

        # FullKV answer-onset teacher.
        full_cache = prefill_context(model, context_ids, press=None)
        full_state = build_full_prompt_state(model, full_cache, query_ids)
        pseudo_ids, full_trace = greedy_reference_trace(
            model,
            full_state,
            steps=max_trace,
        )
        pseudo_text = tokenizer.decode(
            pseudo_ids[0].tolist(),
            skip_special_tokens=True,
        )

        full_answer_nll, full_answer_nll_all = best_gold_answer_nll(
            model,
            tokenizer,
            full_state,
            row["answers"],
        )
        fullkv = {
            "name": "fullkv",
            "answer_nll": full_answer_nll,
            "all_gold_answer_nll": full_answer_nll_all,
            "answer_trace": {},
        }
        for length in trace_lens:
            ref = full_trace[:, :length]
            ids = pseudo_ids[:, :length]
            fullkv["answer_trace"][str(length)] = score_trace(
                ref,
                ref,
                ids,
            )
        del full_state

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
                f"{row['id']}: requested {args.num_swaps} candidates, "
                f"got {len(candidates)}"
            )

        press.set_keep_counts_override(budget_to_override(baseline_budget, None))
        lu_cache = prefill_context(model, context_ids, press=press)
        lu = evaluate_state(
            model=model,
            tokenizer=tokenizer,
            context_cache=lu_cache,
            query_ids=query_ids,
            answers=row["answers"],
            pseudo_ids=pseudo_ids,
            full_trace=full_trace,
            trace_lens=trace_lens,
        )
        lu["name"] = "lu_baseline"

        conditions = [fullkv, lu]

        for idx, candidate in enumerate(candidates):
            print(
                f"  candidate {idx + 1}/{len(candidates)} "
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
            result = evaluate_state(
                model=model,
                tokenizer=tokenizer,
                context_cache=candidate_cache,
                query_ids=query_ids,
                answers=row["answers"],
                pseudo_ids=pseudo_ids,
                full_trace=full_trace,
                trace_lens=trace_lens,
            )
            result.update(candidate_meta(candidate))
            add_deltas(result, lu, trace_lens)
            conditions.append(result)

            del candidate_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        payload = {
            "protocol": "CausalDuel-KV Gate-0-v2 answer-onset discovery",
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
            "trace_lens": trace_lens,
            "teacher": "FullKV greedy pseudo-answer; no gold used online",
            "pseudo_answer_token_ids": pseudo_ids[0].tolist(),
            "pseudo_answer_text": pseudo_text,
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
        print(f"[AnswerOnset] saved {out_path}")

        del full_cache, lu_cache, pseudo_ids, full_trace
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[AnswerOnset] completed requested prompts.")


if __name__ == "__main__":
    main()
