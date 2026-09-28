"""Build the frozen 12-action dictionary for Gate 0B half-scale testing.

Actions are selected using only the LU marginal profile and prompt-length
legality.  No Answer-NLL, task score, or model output is consulted.

The default 12-action mix is:
  - 4 LU-promising actions (largest marginal gain - remove cost)
  - 6 near-boundary actions (3 closest positive + 3 closest negative when possible)
  - 2 LU-unfavored controls (most negative marginal deltas)

A soft unit-use cap encourages donor/receiver diversity.  If a category cannot
be filled under the cap, the selector deterministically relaxes it rather than
changing the requested category counts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.sketchalloc.build_candidates import (
    SwapCandidate,
    Unit,
    apply_swap,
    load_lu_marginal_slice,
)
from evaluation.sketchalloc.run_gate0a_replication import curve_budget_from_array


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--budget-curve-path", required=True)
    p.add_argument("--marginal-profile-path", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/sketchalloc/gate0b_half12_actions.json"),
    )
    p.add_argument("--compression-ratio", type=float, default=0.80)
    p.add_argument("--swap-size", type=int, default=16)
    p.add_argument("--num-promising", type=int, default=4)
    p.add_argument("--num-boundary", type=int, default=6)
    p.add_argument("--num-control", type=int, default=2)
    p.add_argument("--unit-use-cap", type=int, default=3)
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_id(candidate: SwapCandidate) -> str:
    return (
        f"L{candidate.donor.layer}H{candidate.donor.head}"
        f"->L{candidate.receiver.layer}H{candidate.receiver.head}"
        f"@{candidate.amount}"
    )


def enumerate_shared_legal_pairs(
    rows: list[dict],
    curve: np.ndarray,
    marginal,
    compression_ratio: float,
    swap_size: int,
) -> list[SwapCandidate]:
    if marginal.remove_cost.shape != marginal.next_gain.shape:
        raise ValueError("LU marginal remove_cost and next_gain shapes differ")

    baselines = []
    for row in rows:
        seq_len = int(row["context_tokens"])
        baselines.append(
            (
                row["id"],
                seq_len,
                curve_budget_from_array(curve, compression_ratio, seq_len),
            )
        )

    layers, heads = marginal.remove_cost.shape
    candidates: list[SwapCandidate] = []
    for dl in range(layers):
        for dh in range(heads):
            remove = float(marginal.remove_cost[dl, dh])
            if not np.isfinite(remove):
                continue
            donor = Unit(dl, dh)
            for rl in range(layers):
                for rh in range(heads):
                    receiver = Unit(rl, rh)
                    if donor == receiver:
                        continue
                    gain = float(marginal.next_gain[rl, rh])
                    if not np.isfinite(gain):
                        continue

                    candidate = SwapCandidate(
                        donor=donor,
                        receiver=receiver,
                        amount=swap_size,
                        lu_remove_cost=remove,
                        lu_next_gain=gain,
                        lu_marginal_delta=gain - remove,
                    )

                    legal = True
                    for _, seq_len, baseline in baselines:
                        min_keep = min(seq_len, 4 + 32)
                        try:
                            apply_swap(
                                baseline,
                                candidate,
                                min_keep=min_keep,
                                max_keep=seq_len,
                            )
                        except Exception:
                            legal = False
                            break
                    if legal:
                        candidates.append(candidate)

    if not candidates:
        raise RuntimeError("no donor->receiver pair is legal across every Gate 0B prompt")
    return candidates


def select_with_cap(
    pool: list[SwapCandidate],
    count: int,
    selected_ids: set[str],
    donor_use: Counter,
    receiver_use: Counter,
    unit_use_cap: int,
    category: str,
) -> list[SwapCandidate]:
    chosen: list[SwapCandidate] = []

    def try_add(candidate: SwapCandidate, enforce_cap: bool) -> bool:
        sid = stable_id(candidate)
        if sid in selected_ids:
            return False
        if enforce_cap and (
            donor_use[candidate.donor] >= unit_use_cap
            or receiver_use[candidate.receiver] >= unit_use_cap
        ):
            return False
        out = SwapCandidate(
            donor=candidate.donor,
            receiver=candidate.receiver,
            amount=candidate.amount,
            lu_remove_cost=candidate.lu_remove_cost,
            lu_next_gain=candidate.lu_next_gain,
            lu_marginal_delta=candidate.lu_marginal_delta,
            category=category,
        )
        chosen.append(out)
        selected_ids.add(sid)
        donor_use[candidate.donor] += 1
        receiver_use[candidate.receiver] += 1
        return True

    for enforce_cap in (True, False):
        for candidate in pool:
            if len(chosen) >= count:
                break
            try_add(candidate, enforce_cap)
        if len(chosen) >= count:
            break

    if len(chosen) != count:
        raise RuntimeError(
            f"could only select {len(chosen)}/{count} actions for category {category}"
        )
    return chosen


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("Gate 0B data file is empty")

    expected_total = args.num_promising + args.num_boundary + args.num_control
    if expected_total != 12:
        raise ValueError(
            f"Gate 0B half-scale action dictionary must contain 12 actions; requested {expected_total}"
        )

    curve = np.load(args.budget_curve_path)
    marginal = load_lu_marginal_slice(
        args.marginal_profile_path,
        compression_ratio=args.compression_ratio,
        expected_step_tokens=args.swap_size,
    )
    shared = enumerate_shared_legal_pairs(
        rows,
        curve,
        marginal,
        args.compression_ratio,
        args.swap_size,
    )

    promising_pool = sorted(
        shared,
        key=lambda c: (
            -float(c.lu_marginal_delta),
            c.donor.layer,
            c.donor.head,
            c.receiver.layer,
            c.receiver.head,
        ),
    )

    positive_boundary = sorted(
        [c for c in shared if float(c.lu_marginal_delta) >= 0],
        key=lambda c: (
            abs(float(c.lu_marginal_delta)),
            c.donor.layer,
            c.donor.head,
            c.receiver.layer,
            c.receiver.head,
        ),
    )
    negative_boundary = sorted(
        [c for c in shared if float(c.lu_marginal_delta) < 0],
        key=lambda c: (
            abs(float(c.lu_marginal_delta)),
            c.donor.layer,
            c.donor.head,
            c.receiver.layer,
            c.receiver.head,
        ),
    )

    boundary_pool: list[SwapCandidate] = []
    pos_target = args.num_boundary // 2
    neg_target = args.num_boundary - pos_target
    boundary_pool.extend(positive_boundary[: max(pos_target * 8, pos_target)])
    boundary_pool.extend(negative_boundary[: max(neg_target * 8, neg_target)])
    boundary_pool.sort(
        key=lambda c: (
            abs(float(c.lu_marginal_delta)),
            0 if float(c.lu_marginal_delta) >= 0 else 1,
            c.donor.layer,
            c.donor.head,
            c.receiver.layer,
            c.receiver.head,
        )
    )

    control_pool = sorted(
        shared,
        key=lambda c: (
            float(c.lu_marginal_delta),
            c.donor.layer,
            c.donor.head,
            c.receiver.layer,
            c.receiver.head,
        ),
    )

    selected_ids: set[str] = set()
    donor_use: Counter = Counter()
    receiver_use: Counter = Counter()
    selected: list[SwapCandidate] = []

    selected.extend(
        select_with_cap(
            promising_pool,
            args.num_promising,
            selected_ids,
            donor_use,
            receiver_use,
            args.unit_use_cap,
            "lu_promising",
        )
    )

    # Preserve both boundary signs when possible.
    pos_chosen = select_with_cap(
        positive_boundary,
        pos_target,
        selected_ids,
        donor_use,
        receiver_use,
        args.unit_use_cap,
        "near_boundary_positive",
    )
    neg_chosen = select_with_cap(
        negative_boundary,
        neg_target,
        selected_ids,
        donor_use,
        receiver_use,
        args.unit_use_cap,
        "near_boundary_negative",
    )
    selected.extend(pos_chosen)
    selected.extend(neg_chosen)

    selected.extend(
        select_with_cap(
            control_pool,
            args.num_control,
            selected_ids,
            donor_use,
            receiver_use,
            args.unit_use_cap,
            "lu_unfavored_control",
        )
    )

    if len(selected) != 12 or len({stable_id(c) for c in selected}) != 12:
        raise AssertionError("failed to build 12 unique Gate 0B actions")

    payload = {
        "name": "gate0b_half12_v1",
        "status": "frozen_before_gate0b_half_labels",
        "selection_uses_model_outputs": False,
        "selection_uses_answer_labels": False,
        "compression_ratio": args.compression_ratio,
        "retention_ratio": 1.0 - args.compression_ratio,
        "swap_size": args.swap_size,
        "num_prompts_legality_checked": len(rows),
        "num_shared_legal_pairs": len(shared),
        "unit_use_cap": args.unit_use_cap,
        "category_counts": dict(Counter(c.category for c in selected)),
        "unique_donors": len({c.donor for c in selected}),
        "unique_receivers": len({c.receiver for c in selected}),
        "marginal_profile": {
            "path": str(args.marginal_profile_path),
            "resolved_compression_ratio": marginal.global_compression_ratio,
            "step_tokens": marginal.marginal_step_tokens,
            "sink_size": marginal.sink_size,
            "window_size": marginal.window_size,
            "calibration_pairs": marginal.calibration_pairs,
        },
        "actions": [
            {
                "action_id": stable_id(c),
                "donor_layer": c.donor.layer,
                "donor_head": c.donor.head,
                "receiver_layer": c.receiver.layer,
                "receiver_head": c.receiver.head,
                "amount": c.amount,
                "category": c.category,
                "lu_remove_cost": c.lu_remove_cost,
                "lu_next_gain": c.lu_next_gain,
                "lu_marginal_delta": c.lu_marginal_delta,
            }
            for c in selected
        ],
        "selection_rule": (
            "shared legal intersection across all half48 prompt lengths; "
            "LU marginal profile only; 4 strongest LU-promising + "
            "3 closest nonnegative boundary + 3 closest negative boundary + "
            "2 most LU-unfavored controls; soft donor/receiver use cap for diversity; "
            "no Gate 0B model output or label consulted"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote Gate 0B action dictionary: {args.output}")
    print(
        json.dumps(
            {
                "num_shared_legal_pairs": payload["num_shared_legal_pairs"],
                "category_counts": payload["category_counts"],
                "unique_donors": payload["unique_donors"],
                "unique_receivers": payload["unique_receivers"],
                "actions": payload["actions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
