"""Audit whether a legacy LU marginal profile yields one shared action set.

The legacy answer-onset candidates were generated from a static LU profile but
the legal action check also depended on context length.  This tool reconstructs
the LU budget at each manifest length and verifies whether candidate generation
is invariant across those lengths.  It is a provenance check, not a substitute
for the missing per-prompt intervention outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.sketchalloc.build_candidates import generate_lu_boundary_swaps


def reconstruct_lu_budget(prune_ratios: np.ndarray, seq_len: int) -> list[list[int]]:
    """Reproduce LUPress integer keep-count rounding without importing torch."""

    ratios = np.asarray(prune_ratios, dtype=np.float64)
    if ratios.ndim != 2:
        raise ValueError(f"prune_ratios must be [layers, kv_heads], got {ratios.shape}")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    budget: list[list[int]] = []
    for layer_ratios in ratios:
        ideal = (1.0 - layer_ratios) * seq_len
        target = int(np.rint(ideal.sum()))
        keep = np.floor(ideal).astype(np.int64)
        remainder = target - int(keep.sum())
        if remainder > 0:
            fractional = ideal - keep
            order = np.argsort(-fractional, kind="stable")
            keep[order[: min(remainder, len(order))]] += 1
        keep = np.clip(keep, 1, seq_len)
        budget.append([int(value) for value in keep])
    return budget


def _action_id(candidate) -> str:
    return (
        f"L{candidate.donor.layer}H{candidate.donor.head}->"
        f"L{candidate.receiver.layer}H{candidate.receiver.head}@{candidate.amount}"
    )


def audit_profile(
    profile_path: Path,
    manifest_path: Path,
    compression_ratio: float,
    num_candidates: int,
    extra_lengths: list[int] | None = None,
) -> dict:
    with np.load(profile_path, allow_pickle=False) as profile:
        ratios = np.asarray(profile["global_compression_ratio"], dtype=np.float64)
        ratio_idx = int(np.argmin(np.abs(ratios - compression_ratio)))
        resolved_ratio = float(ratios[ratio_idx])
        if abs(resolved_ratio - compression_ratio) > 0.0051:
            raise ValueError(
                f"profile has no row near compression_ratio={compression_ratio}; " f"nearest is {resolved_ratio}"
            )
        prune = np.asarray(profile["budget_prune_ratio"][ratio_idx], dtype=np.float64)
        remove_cost = np.asarray(profile["remove_cost"][ratio_idx], dtype=np.float64)
        next_gain = np.asarray(profile["next_gain"][ratio_idx], dtype=np.float64)
        amount = int(np.asarray(profile["marginal_step_tokens"]).item())
        profile_meta = {
            "resolved_compression_ratio": resolved_ratio,
            "marginal_step_tokens": amount,
            "sink_size": int(np.asarray(profile["sink_size"]).item()),
            "window_size": int(np.asarray(profile["window_size"]).item()),
            "calibration_pairs": int(np.asarray(profile["calibration_pairs"]).item()),
            "num_layers": int(prune.shape[0]),
            "num_kv_heads": int(prune.shape[1]),
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_lengths = {int(story["token_count"]) for story in manifest["stories"]}
    requested_lengths = {int(value) for value in (extra_lengths or [])}
    if any(value <= 0 for value in requested_lengths):
        raise ValueError("all extra context lengths must be positive")
    token_lengths = sorted(manifest_lengths | requested_lengths)
    per_length = []
    action_sequences: list[tuple[str, ...]] = []
    for seq_len in token_lengths:
        budget = reconstruct_lu_budget(prune, seq_len)
        candidates, _, _ = generate_lu_boundary_swaps(
            budget,
            remove_cost,
            next_gain,
            amount=amount,
            num_candidates=num_candidates,
            min_keep=1,
            max_keep=seq_len,
        )
        action_ids = tuple(_action_id(candidate) for candidate in candidates)
        action_sequences.append(action_ids)
        per_length.append(
            {
                "context_tokens": seq_len,
                "total_keep_tokens": int(sum(sum(row) for row in budget)),
                "actions": [
                    {
                        "action_id": _action_id(candidate),
                        "category": candidate.category,
                        "lu_marginal_delta": candidate.lu_marginal_delta,
                    }
                    for candidate in candidates
                ],
            }
        )

    unique_sequences = sorted(set(action_sequences))
    shared_actions = list(unique_sequences[0]) if len(unique_sequences) == 1 else []
    return {
        "profile_path": str(profile_path),
        "manifest_path": str(manifest_path),
        **profile_meta,
        "num_context_lengths": len(token_lengths),
        "manifest_context_lengths": sorted(manifest_lengths),
        "extra_context_lengths": sorted(requested_lengths),
        "candidate_count_requested": num_candidates,
        "shared_ordered_action_set": len(unique_sequences) == 1,
        "shared_action_ids": shared_actions,
        "num_unique_action_sequences": len(unique_sequences),
        "per_length": per_length,
        "interpretation": (
            "The static profile produces one shared action dictionary at all audited lengths. "
            "Actual intervention JSON is still required for Gate 0A utility statistics."
            if len(unique_sequences) == 1
            else "Candidate actions vary with length; old outcomes cannot be treated as one rectangular utility matrix."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compression-ratio", type=float, default=0.80)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--extra-lengths", type=int, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_profile(
        args.profile,
        args.manifest,
        compression_ratio=args.compression_ratio,
        num_candidates=args.num_candidates,
        extra_lengths=args.extra_lengths,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
