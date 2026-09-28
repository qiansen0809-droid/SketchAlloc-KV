from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Unit:
    layer: int
    head: int


@dataclass(frozen=True)
class SwapCandidate:
    donor: Unit
    receiver: Unit
    amount: int
    lu_remove_cost: float | None = None
    lu_next_gain: float | None = None
    lu_marginal_delta: float | None = None
    category: str = "unspecified"

    @property
    def name(self) -> str:
        return (
            f"L{self.donor.layer}H{self.donor.head}"
            f"_to_L{self.receiver.layer}H{self.receiver.head}"
            f"_q{self.amount}"
        )


@dataclass(frozen=True)
class LUMarginalSlice:
    remove_cost: np.ndarray
    next_gain: np.ndarray
    global_compression_ratio: float
    marginal_step_tokens: int
    sink_size: int
    window_size: int
    calibration_pairs: int


def clone_budget(budget: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(map(int, row)) for row in budget]


def total_budget(budget: Sequence[Sequence[int]]) -> int:
    return sum(sum(int(v) for v in row) for row in budget)


def apply_swap(
    baseline: Sequence[Sequence[int]],
    candidate: SwapCandidate,
    min_keep: int = 1,
    max_keep: int | None = None,
) -> list[list[int]]:
    out = clone_budget(baseline)
    before = total_budget(out)

    d = candidate.donor
    r = candidate.receiver
    q = int(candidate.amount)

    if q <= 0:
        raise ValueError("swap amount must be positive")
    if d == r:
        raise ValueError("donor and receiver must be different units")

    try:
        donor_value = out[d.layer][d.head]
        receiver_value = out[r.layer][r.head]
    except IndexError as e:
        raise ValueError("swap references an invalid layer/head") from e

    if donor_value - q < min_keep:
        raise ValueError(
            f"donor {d} would keep {donor_value - q}, below min_keep={min_keep}"
        )
    if max_keep is not None and receiver_value + q > max_keep:
        raise ValueError(
            f"receiver {r} would keep {receiver_value + q}, above max_keep={max_keep}"
        )

    out[d.layer][d.head] -= q
    out[r.layer][r.head] += q

    after = total_budget(out)
    if after != before:
        raise AssertionError(f"budget changed from {before} to {after}")

    return out


def budget_to_override(
    baseline: Sequence[Sequence[int]],
    candidate: SwapCandidate | None,
    min_keep: int = 1,
    max_keep: int | None = None,
) -> dict[int, list[int]]:
    budget = (
        clone_budget(baseline)
        if candidate is None
        else apply_swap(baseline, candidate, min_keep=min_keep, max_keep=max_keep)
    )
    return {layer: row for layer, row in enumerate(budget)}


def _flatten_units(budget: Sequence[Sequence[int]]) -> list[Unit]:
    return [
        Unit(layer=layer_idx, head=head_idx)
        for layer_idx, row in enumerate(budget)
        for head_idx, _ in enumerate(row)
    ]


def _validate_marginal_shape(
    baseline: Sequence[Sequence[int]],
    values: np.ndarray,
    name: str,
):
    expected_layers = len(baseline)
    if values.ndim != 2 or values.shape[0] != expected_layers:
        raise ValueError(
            f"{name} must have shape [layers, kv_heads]; got {values.shape}"
        )

    for layer_idx, row in enumerate(baseline):
        if values.shape[1] != len(row):
            raise ValueError(
                f"{name} KV-head count mismatch at layer {layer_idx}: "
                f"profile has {values.shape[1]}, budget has {len(row)}"
            )


def load_lu_marginal_slice(
    path: str | Path,
    compression_ratio: float,
    expected_step_tokens: int | None = None,
) -> LUMarginalSlice:
    """
    Load the Gate-0 marginal export created by step2_compute_curve.py.

    compression_ratio follows LU-KV's convention: 0.80 means 80% pruning.
    """
    data = np.load(path)
    ratios = np.asarray(data["global_compression_ratio"], dtype=np.float64)

    target_idx = int(np.argmin(np.abs(ratios - compression_ratio)))
    resolved_ratio = float(ratios[target_idx])
    if abs(resolved_ratio - compression_ratio) > 0.0051:
        raise ValueError(
            f"marginal profile has no row near compression_ratio={compression_ratio:.4f}; "
            f"nearest is {resolved_ratio:.4f}"
        )

    step_tokens = int(np.asarray(data["marginal_step_tokens"]).item())
    if expected_step_tokens is not None and step_tokens != expected_step_tokens:
        raise ValueError(
            f"marginal profile was exported for step={step_tokens} tokens, "
            f"but Gate 0 requested swap-size={expected_step_tokens}. "
            "Re-export the profile with matching --marginal_step_tokens."
        )

    return LUMarginalSlice(
        remove_cost=np.asarray(data["remove_cost"][target_idx], dtype=np.float64),
        next_gain=np.asarray(data["next_gain"][target_idx], dtype=np.float64),
        global_compression_ratio=resolved_ratio,
        marginal_step_tokens=step_tokens,
        sink_size=int(np.asarray(data["sink_size"]).item()),
        window_size=int(np.asarray(data["window_size"]).item()),
        calibration_pairs=int(np.asarray(data["calibration_pairs"]).item()),
    )


def generate_lu_boundary_swaps(
    baseline: Sequence[Sequence[int]],
    remove_cost: np.ndarray,
    next_gain: np.ndarray,
    amount: int,
    num_candidates: int = 6,
    min_keep: int = 1,
    max_keep: int | None = None,
) -> tuple[list[SwapCandidate], list[Unit], list[Unit]]:
    """
    Construct the formal Gate-0 candidate neighborhood from LU-KV boundary
    marginals.

    Candidate search is GLOBAL over every legal donor->receiver pair. This is
    important: restricting the search to only the lowest-remove-cost donors and
    highest-next-gain receivers makes every candidate LU-favored by
    construction, which prevents a genuine near-zero boundary set and an
    LU-unfavored control.

    For six candidates the registered mix is:
      - 2 LU-favored swaps: largest positive gain - cost
      - 3 near-boundary swaps: closest to zero, with both signs represented
        when possible
      - 1 low-priority control: most LU-unfavored legal swap

    The returned donor/receiver pools are the unique units actually used by the
    selected candidates, not a pre-filter that constrains candidate search.
    """
    if num_candidates < 3:
        raise ValueError("formal Gate-0 generation requires at least 3 candidates")
    if amount <= 0:
        raise ValueError("amount must be positive")

    remove_cost = np.asarray(remove_cost, dtype=np.float64)
    next_gain = np.asarray(next_gain, dtype=np.float64)
    _validate_marginal_shape(baseline, remove_cost, "remove_cost")
    _validate_marginal_shape(baseline, next_gain, "next_gain")

    units = _flatten_units(baseline)

    donor_eligible = [
        u
        for u in units
        if baseline[u.layer][u.head] - amount >= min_keep
        and np.isfinite(remove_cost[u.layer, u.head])
    ]
    receiver_eligible = [
        u
        for u in units
        if (max_keep is None or baseline[u.layer][u.head] + amount <= max_keep)
        and np.isfinite(next_gain[u.layer, u.head])
    ]

    all_pairs: list[SwapCandidate] = []
    for donor in donor_eligible:
        for receiver in receiver_eligible:
            if donor == receiver:
                continue

            remove = float(remove_cost[donor.layer, donor.head])
            gain = float(next_gain[receiver.layer, receiver.head])
            delta = gain - remove

            candidate = SwapCandidate(
                donor=donor,
                receiver=receiver,
                amount=amount,
                lu_remove_cost=remove,
                lu_next_gain=gain,
                lu_marginal_delta=delta,
            )

            apply_swap(
                baseline,
                candidate,
                min_keep=min_keep,
                max_keep=max_keep,
            )
            all_pairs.append(candidate)

    if not all_pairs:
        return [], [], []

    promising_n = 3 if num_candidates >= 8 else 2
    boundary_n = num_candidates - promising_n - 1

    selected: list[SwapCandidate] = []
    selected_keys: set[tuple[Unit, Unit]] = set()

    def add(candidate: SwapCandidate, category: str) -> bool:
        key = (candidate.donor, candidate.receiver)
        if key in selected_keys:
            return False
        selected.append(
            SwapCandidate(
                donor=candidate.donor,
                receiver=candidate.receiver,
                amount=candidate.amount,
                lu_remove_cost=candidate.lu_remove_cost,
                lu_next_gain=candidate.lu_next_gain,
                lu_marginal_delta=candidate.lu_marginal_delta,
                category=category,
            )
        )
        selected_keys.add(key)
        return True

    # 1) Strong LU-favored moves.
    promising = sorted(
        all_pairs,
        key=lambda c: (
            -float(c.lu_marginal_delta),
            c.donor.layer,
            c.donor.head,
            c.receiver.layer,
            c.receiver.head,
        ),
    )
    for cand in promising:
        if len([c for c in selected if c.category == "lu_promising"]) >= promising_n:
            break
        add(cand, "lu_promising")

    # 2) Genuine local boundary probes. For three boundary candidates, prefer
    # one closest positive, one closest negative, then the closest remaining
    # pair (which may be exactly zero). This gives the duel meaningful
    # directional ambiguity instead of only LU-favored moves.
    positive_boundary = sorted(
        [c for c in all_pairs if float(c.lu_marginal_delta) > 0],
        key=lambda c: (
            float(c.lu_marginal_delta),
            c.donor.layer,
            c.donor.head,
            c.receiver.layer,
            c.receiver.head,
        ),
    )
    negative_boundary = sorted(
        [c for c in all_pairs if float(c.lu_marginal_delta) < 0],
        key=lambda c: (
            -float(c.lu_marginal_delta),
            c.donor.layer,
            c.donor.head,
            c.receiver.layer,
            c.receiver.head,
        ),
    )

    if boundary_n > 0:
        for pool in (positive_boundary, negative_boundary):
            for cand in pool:
                if add(cand, "near_boundary"):
                    break

    boundary = sorted(
        all_pairs,
        key=lambda c: (
            abs(float(c.lu_marginal_delta)),
            -float(c.lu_marginal_delta),
            c.donor.layer,
            c.donor.head,
            c.receiver.layer,
            c.receiver.head,
        ),
    )
    while len([c for c in selected if c.category == "near_boundary"]) < boundary_n:
        added = False
        for cand in boundary:
            if add(cand, "near_boundary"):
                added = True
                break
        if not added:
            break

    # 3) A real LU-unfavored control whenever one exists.
    control = sorted(
        all_pairs,
        key=lambda c: (
            float(c.lu_marginal_delta),
            c.donor.layer,
            c.donor.head,
            c.receiver.layer,
            c.receiver.head,
        ),
    )
    for cand in control:
        if add(cand, "low_priority_control"):
            break

    # Very small legal spaces can still yield fewer unique pairs than the
    # requested mix. Fill deterministically from the strongest remaining pairs.
    if len(selected) < num_candidates:
        for cand in promising:
            if len(selected) >= num_candidates:
                break
            add(cand, "fill")

    selected = selected[:num_candidates]

    donor_pool = sorted(
        {c.donor for c in selected},
        key=lambda u: (
            float(remove_cost[u.layer, u.head]),
            u.layer,
            u.head,
        ),
    )
    receiver_pool = sorted(
        {c.receiver for c in selected},
        key=lambda u: (
            -float(next_gain[u.layer, u.head]),
            u.layer,
            u.head,
        ),
    )

    return selected, donor_pool, receiver_pool


def generate_poc_swaps(
    baseline: Sequence[Sequence[int]],
    amount: int,
    num_candidates: int = 3,
    min_keep: int = 1,
    max_keep: int | None = None,
    donor_order: Iterable[Unit] | None = None,
    receiver_order: Iterable[Unit] | None = None,
) -> list[SwapCandidate]:
    """
    Deterministic engineering-only fallback.

    This does NOT represent the formal Gate-0 candidate policy. It is kept so a
    2-sample smoke test can run before a marginal profile has been exported.
    """
    units = _flatten_units(baseline)

    if donor_order is None:
        donors = sorted(
            units,
            key=lambda u: baseline[u.layer][u.head],
            reverse=True,
        )
    else:
        donors = list(donor_order)

    if receiver_order is None:
        receivers = sorted(
            units,
            key=lambda u: baseline[u.layer][u.head],
        )
    else:
        receivers = list(receiver_order)

    candidates: list[SwapCandidate] = []
    seen: set[tuple[Unit, Unit]] = set()

    for donor in donors:
        if baseline[donor.layer][donor.head] - amount < min_keep:
            continue

        for receiver in receivers:
            if donor == receiver:
                continue
            if (
                max_keep is not None
                and baseline[receiver.layer][receiver.head] + amount > max_keep
            ):
                continue

            key = (donor, receiver)
            if key in seen:
                continue

            cand = SwapCandidate(
                donor=donor,
                receiver=receiver,
                amount=amount,
                category="engineering_fallback",
            )
            apply_swap(
                baseline,
                cand,
                min_keep=min_keep,
                max_keep=max_keep,
            )
            candidates.append(cand)
            seen.add(key)

            if len(candidates) >= num_candidates:
                return candidates

    return candidates
