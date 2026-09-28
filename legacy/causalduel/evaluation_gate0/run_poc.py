from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.gate0.build_candidates import (
    budget_to_override,
    generate_lu_boundary_swaps,
    generate_poc_swaps,
    load_lu_marginal_slice,
)
from evaluation.gate0.metrics import (
    continuation_nll,
    jensen_shannon,
    probability_overlap,
    teacher_to_candidate_kl,
    topk_agreement,
)
from evaluation.gate0.suffix_replay import replay_suffix
from kvpress import LUPress, SnapKVPress


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--budget-curve-path", required=True)
    p.add_argument(
        "--marginal-profile-path",
        type=Path,
        default=None,
        help=(
            "Gate-0 LU boundary marginal profile (.npz). When provided, "
            "formal donor/receiver pools are selected by LU remove-cost and "
            "next-gain. Without it, the script uses the engineering-only fallback."
        ),
    )
    p.add_argument("--text-file", type=Path, required=True)
    p.add_argument("--compression-ratio", type=float, default=0.80)
    p.add_argument("--suffix-len", type=int, default=64)
    p.add_argument("--swap-size", type=int, default=16)
    p.add_argument(
        "--num-swaps",
        type=int,
        default=6,
        help="Formal Gate 0 uses 6-8 swaps; engineering fallback may use fewer.",
    )
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--output", type=Path, default=Path("gate0_poc.json"))
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


def tokenize_prompt(tokenizer, text: str, max_tokens: int) -> torch.Tensor:
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=max_tokens,
    ).input_ids
    if ids.shape[1] < 128:
        raise ValueError("POC input is too short; use a long-context sample")
    return ids


def curve_budget(press: LUPress, model, seq_len: int) -> list[list[int]]:
    num_heads = model.config.num_key_value_heads
    num_layers = model.config.num_hidden_layers
    budget = []
    for layer_idx in range(num_layers):
        counts = press.get_keep_counts(
            layer_idx=layer_idx,
            num_heads=num_heads,
            seq_len=seq_len,
            device=torch.device("cpu"),
        )
        if counts is None:
            raise RuntimeError(f"no LU-KV budget available for layer {layer_idx}")
        budget.append([int(v) for v in counts.tolist()])
    return budget


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


def score_candidate(ref_logits, cand_logits, target_ids):
    return {
        "overlap": float(probability_overlap(ref_logits, cand_logits).item()),
        "kl": float(teacher_to_candidate_kl(ref_logits, cand_logits).item()),
        "js": float(jensen_shannon(ref_logits, cand_logits).item()),
        "top10_agreement": float(
            topk_agreement(ref_logits, cand_logits, k=10).item()
        ),
        "suffix_nll": float(continuation_nll(cand_logits, target_ids).item()),
    }


def candidate_metadata(candidate):
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
    model, tokenizer = load_model(args.model, args.dtype)

    text = args.text_file.read_text(encoding="utf-8")
    ids = tokenize_prompt(tokenizer, text, args.max_tokens)

    if ids.shape[1] <= args.suffix_len:
        raise ValueError("suffix length must be shorter than the tokenized prompt")

    prefix_ids = ids[:, :-args.suffix_len]
    suffix_ids = ids[:, -args.suffix_len:]
    targets = suffix_ids[:, 1:].to(model.device)

    reference = replay_suffix(model, prefix_ids, suffix_ids, press=None)
    ref_logits = reference.logits

    press = make_press(args.budget_curve_path, args.compression_ratio)
    baseline_budget = curve_budget(press, model, prefix_ids.shape[1])

    min_keep = min(prefix_ids.shape[1], press.sink + press.window)

    generation_mode = "engineering_fallback"
    donor_pool = []
    receiver_pool = []
    marginal_meta = None

    if args.marginal_profile_path is not None:
        marginal = load_lu_marginal_slice(
            args.marginal_profile_path,
            compression_ratio=args.compression_ratio,
            expected_step_tokens=args.swap_size,
        )

        if marginal.sink_size != press.sink or marginal.window_size != press.window:
            raise ValueError(
                "marginal profile protection settings do not match runtime LUPress: "
                f"profile sink/window={marginal.sink_size}/{marginal.window_size}, "
                f"runtime={press.sink}/{press.window}"
            )

        candidates, donor_pool, receiver_pool = generate_lu_boundary_swaps(
            baseline_budget,
            remove_cost=marginal.remove_cost,
            next_gain=marginal.next_gain,
            amount=args.swap_size,
            num_candidates=args.num_swaps,
            min_keep=min_keep,
            max_keep=prefix_ids.shape[1],
        )
        generation_mode = "lu_boundary_marginal"
        marginal_meta = {
            "profile_path": str(args.marginal_profile_path),
            "profile_compression_ratio": marginal.global_compression_ratio,
            "marginal_step_tokens": marginal.marginal_step_tokens,
            "calibration_pairs": marginal.calibration_pairs,
        }
    else:
        fallback_n = min(args.num_swaps, 3)
        candidates = generate_poc_swaps(
            baseline_budget,
            amount=args.swap_size,
            num_candidates=fallback_n,
            min_keep=min_keep,
            max_keep=prefix_ids.shape[1],
        )

    if not candidates:
        raise RuntimeError("no legal Gate-0 swap candidates were generated")

    runs = []

    press.set_keep_counts_override(budget_to_override(baseline_budget, None))
    baseline = replay_suffix(model, prefix_ids, suffix_ids, press=press)
    baseline_scores = score_candidate(ref_logits, baseline.logits, targets)
    runs.append({"name": "lu_baseline", "scores": baseline_scores})

    for candidate in candidates:
        override = budget_to_override(
            baseline_budget,
            candidate,
            min_keep=min_keep,
            max_keep=prefix_ids.shape[1],
        )
        press.set_keep_counts_override(override)
        result = replay_suffix(model, prefix_ids, suffix_ids, press=press)
        scores = score_candidate(ref_logits, result.logits, targets)
        scores["delta_overlap_vs_lu"] = scores["overlap"] - baseline_scores["overlap"]
        scores["delta_nll_vs_lu"] = (
            scores["suffix_nll"] - baseline_scores["suffix_nll"]
        )

        row = candidate_metadata(candidate)
        row["scores"] = scores
        runs.append(row)

    payload = {
        "model": args.model,
        "prompt_tokens": int(ids.shape[1]),
        "prefix_tokens": int(prefix_ids.shape[1]),
        "suffix_tokens": int(suffix_ids.shape[1]),
        "compression_ratio": args.compression_ratio,
        "swap_size": args.swap_size,
        "num_swaps": len(candidates),
        "candidate_generation": generation_mode,
        "donor_pool": [u.__dict__ for u in donor_pool],
        "receiver_pool": [u.__dict__ for u in receiver_pool],
        "marginal_profile": marginal_meta,
        "runs": runs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
