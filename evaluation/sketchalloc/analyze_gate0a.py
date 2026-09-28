"""Run the SketchAlloc-KV Gate 0A adaptive-necessity audit.

This is deliberately stricter than the legacy per-prompt oracle summary.  A
cross-prompt comparison is only produced when every prompt contains every
action.  Model selection uses leave-one-prompt-out estimates, and LU-KV with
zero gain is always an available fallback.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Gate0ASummary:
    status: str
    num_prompts: int
    num_actions: int
    num_families: int
    complete_action_matrix: bool
    missing_cells: int
    oracle_mean_gain: float | None = None
    random_action_mean_gain: float | None = None
    lopo_best_fixed_mean_gain: float | None = None
    lopo_family_static_mean_gain: float | None = None
    lofo_best_fixed_mean_gain: float | None = None
    oracle_minus_best_fixed_mean: float | None = None
    oracle_minus_best_fixed_ci95: list[float] | None = None
    oracle_minus_family_static_mean: float | None = None
    oracle_minus_family_static_ci95: list[float] | None = None
    best_fixed_headroom_recovery: float | None = None
    family_static_headroom_recovery: float | None = None
    pairwise_action_crossover_rate: float | None = None
    prompt_crossover_participation: float | None = None
    oracle_action_entropy_normalized: float | None = None
    oracle_action_counts: dict[str, int] | None = None
    screening_recommendation: str | None = None


def _validate_and_pivot(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, int]:
    required = {"prompt_id", "action_id", "gain", "family", "context_cluster"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    frame["action_id"] = frame["action_id"].astype(str)
    duplicated = frame.duplicated(["prompt_id", "action_id"], keep=False)
    if duplicated.any():
        raise ValueError("duplicate prompt/action rows are not allowed")

    metadata = frame.groupby("prompt_id").agg(
        family=("family", lambda values: set(map(str, values))),
        context_cluster=("context_cluster", lambda values: set(map(str, values))),
    )
    if metadata["family"].map(len).max() != 1 or metadata["context_cluster"].map(len).max() != 1:
        raise ValueError("family and context_cluster must be constant within a prompt")
    families = metadata["family"].map(lambda value: next(iter(value)))
    clusters = metadata["context_cluster"].map(lambda value: next(iter(value)))
    pivot = frame.pivot(index="prompt_id", columns="action_id", values="gain").sort_index().sort_index(axis=1)
    missing_cells = int(pivot.isna().sum().sum())
    return pivot, families.reindex(pivot.index), clusters.reindex(pivot.index), missing_cells


def _choose_with_lu(mean_gain: np.ndarray) -> int:
    """Return -1 for LU if no intervention has strictly positive mean gain."""

    idx = int(np.argmax(mean_gain))
    return idx if float(mean_gain[idx]) > 0.0 else -1


def _selected_gain(row: np.ndarray, action_idx: int) -> float:
    return 0.0 if action_idx < 0 else float(row[action_idx])


def _cluster_bootstrap_ci(
    values: np.ndarray,
    clusters: pd.Series,
    n_bootstrap: int,
    seed: int,
) -> list[float]:
    grouped = []
    cluster_values = clusters.astype(str).to_numpy()
    for cluster in sorted(set(cluster_values)):
        grouped.append(values[cluster_values == cluster])
    if not grouped:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        sample = np.concatenate([grouped[j] for j in selected])
        draws[i] = float(np.mean(sample))
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def _normalized_entropy(indices: np.ndarray, num_choices: int) -> tuple[float, dict[int, int]]:
    unique, counts = np.unique(indices, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    denominator = math.log(num_choices) if num_choices > 1 else 0.0
    normalized = entropy / denominator if denominator > 0 else 0.0
    return normalized, {int(k): int(v) for k, v in zip(unique, counts)}


def _crossover_metrics(matrix: np.ndarray) -> tuple[float, float]:
    num_prompts, num_actions = matrix.shape
    crossover_pairs = 0
    total_pairs = 0
    participating = np.zeros(num_prompts, dtype=bool)
    for left in range(num_actions):
        for right in range(left + 1, num_actions):
            delta = matrix[:, left] - matrix[:, right]
            positive = delta > 1e-12
            negative = delta < -1e-12
            total_pairs += 1
            if positive.any() and negative.any():
                crossover_pairs += 1
                participating |= positive | negative
    pair_rate = crossover_pairs / total_pairs if total_pairs else 0.0
    prompt_rate = float(np.mean(participating)) if num_prompts else 0.0
    return float(pair_rate), prompt_rate


def evaluate_gate0a(frame: pd.DataFrame, n_bootstrap: int = 5000, seed: int = 20260928) -> Gate0ASummary:
    pivot, families, clusters, missing_cells = _validate_and_pivot(frame)
    num_prompts, num_actions = pivot.shape
    base = Gate0ASummary(
        status="incomplete_action_matrix" if missing_cells else "screening_complete",
        num_prompts=int(num_prompts),
        num_actions=int(num_actions),
        num_families=int(families.nunique()),
        complete_action_matrix=missing_cells == 0,
        missing_cells=missing_cells,
    )
    if missing_cells:
        base.screening_recommendation = (
            "Do not compare actions across prompts. Re-run a fixed shared action dictionary; "
            "the legacy results remain discovery-only evidence."
        )
        return base
    if num_prompts < 4:
        raise ValueError("Gate 0A requires at least four prompts")

    matrix = pivot.to_numpy(dtype=np.float64)
    oracle_idx = np.argmax(np.column_stack([np.zeros(num_prompts), matrix]), axis=1) - 1
    oracle_gain = np.maximum(0.0, np.max(matrix, axis=1))

    lopo_fixed = np.empty(num_prompts, dtype=np.float64)
    lopo_family = np.empty(num_prompts, dtype=np.float64)
    family_values = families.to_numpy()
    for i in range(num_prompts):
        train_mask = np.ones(num_prompts, dtype=bool)
        train_mask[i] = False
        fixed_idx = _choose_with_lu(matrix[train_mask].mean(axis=0))
        lopo_fixed[i] = _selected_gain(matrix[i], fixed_idx)

        family_mask = (family_values == family_values[i]) & train_mask
        family_idx = _choose_with_lu(matrix[family_mask].mean(axis=0)) if family_mask.any() else -1
        lopo_family[i] = _selected_gain(matrix[i], family_idx)

    lofo = np.empty(num_prompts, dtype=np.float64)
    for family in sorted(set(family_values)):
        test_mask = family_values == family
        train_mask = ~test_mask
        action_idx = _choose_with_lu(matrix[train_mask].mean(axis=0)) if train_mask.any() else -1
        lofo[test_mask] = [_selected_gain(row, action_idx) for row in matrix[test_mask]]

    diff_fixed = oracle_gain - lopo_fixed
    diff_family = oracle_gain - lopo_family
    oracle_mean = float(np.mean(oracle_gain))
    fixed_mean = float(np.mean(lopo_fixed))
    family_mean = float(np.mean(lopo_family))
    pair_rate, prompt_rate = _crossover_metrics(matrix)
    entropy, action_counts_idx = _normalized_entropy(oracle_idx, num_actions + 1)
    action_labels = ["LU"] + list(pivot.columns)
    action_counts = {action_labels[idx + 1]: count for idx, count in action_counts_idx.items()}

    fixed_recovery = fixed_mean / oracle_mean if oracle_mean > 0 else None
    family_recovery = family_mean / oracle_mean if oracle_mean > 0 else None
    fixed_ci = _cluster_bootstrap_ci(diff_fixed, clusters, n_bootstrap, seed)
    family_ci = _cluster_bootstrap_ci(diff_family, clusters, n_bootstrap, seed + 1)

    recommendation = "proceed_to_fixed_action_replication"
    if fixed_ci[0] <= 0 or pair_rate < 0.30:
        recommendation = "do_not_claim_prompt_adaptation_yet"

    base.oracle_mean_gain = oracle_mean
    base.random_action_mean_gain = float(np.mean(matrix))
    base.lopo_best_fixed_mean_gain = fixed_mean
    base.lopo_family_static_mean_gain = family_mean
    base.lofo_best_fixed_mean_gain = float(np.mean(lofo))
    base.oracle_minus_best_fixed_mean = float(np.mean(diff_fixed))
    base.oracle_minus_best_fixed_ci95 = fixed_ci
    base.oracle_minus_family_static_mean = float(np.mean(diff_family))
    base.oracle_minus_family_static_ci95 = family_ci
    base.best_fixed_headroom_recovery = None if fixed_recovery is None else float(fixed_recovery)
    base.family_static_headroom_recovery = None if family_recovery is None else float(family_recovery)
    base.pairwise_action_crossover_rate = pair_rate
    base.prompt_crossover_participation = prompt_rate
    base.oracle_action_entropy_normalized = entropy
    base.oracle_action_counts = action_counts
    base.screening_recommendation = recommendation
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    summary = evaluate_gate0a(frame, n_bootstrap=args.bootstrap, seed=args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
