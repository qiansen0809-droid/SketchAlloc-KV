"""Analyze the half-scale Gate 0B prompt-by-action utility matrix.

This is a structural upper bound, not an online selector.  Held-out row
coordinates are recovered by projecting the *true* held-out utility residual
onto a right-singular-vector basis learned only from discovery prompts.  Gate
0C must later show that no-answer probes can identify those coordinates.

The primary Gate 0B object is the family-centered residual:
  R[p,a] = U[p,a] - mean_discovery(U[:,a] | family[p])

Rank is selected on calibration only.  Held-out labels are then used once for
the preregistered structural diagnostics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class EvalMetrics:
    rank: int
    variance_explained: float
    rmse: float
    median_within_prompt_spearman: float
    best_fixed_mean_gain: float
    selected_mean_gain: float
    oracle_mean_gain: float
    mean_top_action_regret: float
    oracle_over_best_fixed_headroom: float
    regret_fraction_of_headroom: float | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/sketchalloc/gate0b/half48_rank.json"),
    )
    p.add_argument("--max-rank", type=int, default=8)
    p.add_argument("--selection-max-rank", type=int, default=4)
    p.add_argument("--parallel-permutations", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260928)
    return p.parse_args()


def row_spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = pd.Series(actual).rank(method="average").to_numpy(dtype=np.float64)
    b = pd.Series(predicted).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def load_matrix(frame: pd.DataFrame):
    required = {"prompt_id", "action_id", "gain", "family", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    frame = frame.copy()
    for col in ("prompt_id", "action_id", "family", "split"):
        frame[col] = frame[col].astype(str)

    duplicated = frame.duplicated(["prompt_id", "action_id"], keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, ["prompt_id", "action_id"]].head().to_dict("records")
        raise ValueError(f"duplicate prompt/action rows: {examples}")

    pivot = (
        frame.pivot(index="prompt_id", columns="action_id", values="gain")
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.isna().any().any():
        raise ValueError(
            f"utility matrix incomplete: {int(pivot.isna().sum().sum())} missing cells"
        )

    meta = (
        frame[["prompt_id", "family", "split"]]
        .drop_duplicates()
        .set_index("prompt_id")
        .loc[pivot.index]
    )
    if meta.index.duplicated().any():
        raise ValueError("prompt metadata is inconsistent")

    split_counts = meta["split"].value_counts().to_dict()
    expected = {"discovery": 24, "calibration": 12, "heldout": 12}
    if split_counts != expected:
        raise ValueError(f"unexpected split counts: {split_counts} != {expected}")

    return (
        pivot.to_numpy(dtype=np.float64),
        list(pivot.index.astype(str)),
        list(pivot.columns.astype(str)),
        meta["family"].to_numpy(dtype=str),
        meta["split"].to_numpy(dtype=str),
    )


def discovery_baselines(
    matrix: np.ndarray,
    families: np.ndarray,
    discovery_mask: np.ndarray,
):
    discovery = matrix[discovery_mask]
    global_action_mean = discovery.mean(axis=0)

    family_action_mean: dict[str, np.ndarray] = {}
    discovery_families = families[discovery_mask]
    for family in sorted(set(families)):
        mask = discovery_families == family
        if not mask.any():
            raise ValueError(f"family {family} has no discovery prompts")
        family_action_mean[family] = discovery[mask].mean(axis=0)

    return global_action_mean, family_action_mean


def baseline_for_rows(
    mode: str,
    families: np.ndarray,
    global_action_mean: np.ndarray,
    family_action_mean: dict[str, np.ndarray],
) -> np.ndarray:
    if mode == "raw":
        return np.zeros((len(families), len(global_action_mean)), dtype=np.float64)
    if mode == "action_centered":
        return np.repeat(global_action_mean[None, :], len(families), axis=0)
    if mode == "family_centered":
        return np.stack([family_action_mean[str(f)] for f in families], axis=0)
    raise ValueError(f"unknown centering mode: {mode}")


def fit_basis(discovery_residual: np.ndarray, rank: int) -> np.ndarray:
    _, _, vt = np.linalg.svd(discovery_residual, full_matrices=False)
    return vt[:rank].T


def evaluate_projection(
    actual_utility: np.ndarray,
    baseline: np.ndarray,
    basis: np.ndarray,
    best_fixed_idx: int,
) -> EvalMetrics:
    residual = actual_utility - baseline
    latent = residual @ basis
    predicted = baseline + latent @ basis.T
    predicted_residual = predicted - baseline

    error = residual - predicted_residual
    centered = residual - residual.mean(axis=0, keepdims=True)
    denominator = float(np.sum(centered**2))
    r2 = 1.0 - float(np.sum(error**2)) / denominator if denominator > 0 else 0.0
    rmse = float(np.sqrt(np.mean(error**2)))
    spearman = float(
        np.median(
            [
                row_spearman(actual, pred)
                for actual, pred in zip(actual_utility, predicted)
            ]
        )
    )

    row_idx = np.arange(actual_utility.shape[0])
    selected_idx = np.argmax(predicted, axis=1)
    oracle_idx = np.argmax(actual_utility, axis=1)

    best_fixed_gain = actual_utility[:, best_fixed_idx]
    selected_gain = actual_utility[row_idx, selected_idx]
    oracle_gain = actual_utility[row_idx, oracle_idx]
    regret = oracle_gain - selected_gain
    headroom = float(np.mean(oracle_gain - best_fixed_gain))
    regret_fraction = float(np.mean(regret)) / headroom if headroom > 0 else None

    return EvalMetrics(
        rank=int(basis.shape[1]),
        variance_explained=float(r2),
        rmse=rmse,
        median_within_prompt_spearman=spearman,
        best_fixed_mean_gain=float(np.mean(best_fixed_gain)),
        selected_mean_gain=float(np.mean(selected_gain)),
        oracle_mean_gain=float(np.mean(oracle_gain)),
        mean_top_action_regret=float(np.mean(regret)),
        oracle_over_best_fixed_headroom=headroom,
        regret_fraction_of_headroom=regret_fraction,
    )


def choose_rank(calibration_metrics: list[EvalMetrics], selection_max_rank: int) -> int:
    eligible = [m for m in calibration_metrics if m.rank <= selection_max_rank]
    if not eligible:
        raise ValueError("no rank is eligible for calibration selection")

    best_r2 = max(m.variance_explained for m in eligible)
    if best_r2 <= 0:
        return max(
            eligible,
            key=lambda m: (
                m.variance_explained,
                m.median_within_prompt_spearman,
                -m.mean_top_action_regret,
                -m.rank,
            ),
        ).rank

    threshold = 0.95 * best_r2
    near_best = [m for m in eligible if m.variance_explained >= threshold]
    return min(m.rank for m in near_best)


def spectrum_summary(residual: np.ndarray, max_rank: int) -> dict:
    singular = np.linalg.svd(residual, full_matrices=False, compute_uv=False)
    energy = singular**2
    total = float(energy.sum())
    cumulative = np.cumsum(energy) / total if total > 0 else np.zeros_like(energy)
    k = min(max_rank, len(singular))
    return {
        "singular_values": [float(x) for x in singular[:k]],
        "cumulative_variance_explained": [float(x) for x in cumulative[:k]],
    }


def parallel_analysis(
    residual: np.ndarray,
    max_rank: int,
    permutations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    observed = np.linalg.svd(residual, full_matrices=False, compute_uv=False)
    k = min(max_rank, len(observed))
    null = np.empty((permutations, k), dtype=np.float64)

    for b in range(permutations):
        permuted = np.empty_like(residual)
        for col in range(residual.shape[1]):
            permuted[:, col] = residual[rng.permutation(residual.shape[0]), col]
        null[b] = np.linalg.svd(permuted, full_matrices=False, compute_uv=False)[:k]

    q95 = np.quantile(null, 0.95, axis=0)
    return {
        "observed_singular_values": [float(x) for x in observed[:k]],
        "permutation_q95": [float(x) for x in q95],
        "num_components_above_q95": int(np.sum(observed[:k] > q95)),
    }


def bootstrap_subspace_stability(
    discovery_residual: np.ndarray,
    rank: int,
    bootstrap: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    reference = fit_basis(discovery_residual, rank)
    affinities = []

    for _ in range(bootstrap):
        idx = rng.integers(0, discovery_residual.shape[0], size=discovery_residual.shape[0])
        sample = discovery_residual[idx]
        basis = fit_basis(sample, rank)
        affinity = float(np.linalg.norm(reference.T @ basis, ord="fro") ** 2 / rank)
        affinities.append(affinity)

    values = np.asarray(affinities, dtype=np.float64)
    return {
        "rank": rank,
        "mean_subspace_affinity": float(values.mean()),
        "median_subspace_affinity": float(np.median(values)),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    matrix, prompt_ids, action_ids, families, splits = load_matrix(frame)

    discovery_mask = splits == "discovery"
    calibration_mask = splits == "calibration"
    heldout_mask = splits == "heldout"

    global_action_mean, family_action_mean = discovery_baselines(
        matrix,
        families,
        discovery_mask,
    )
    best_fixed_idx = int(np.argmax(global_action_mean))

    max_rank = min(args.max_rank, matrix.shape[1], int(discovery_mask.sum()))
    modes = {}
    selected_family_rank = None
    heldout_family_metrics = None

    for mode_idx, mode in enumerate(("raw", "action_centered", "family_centered")):
        baseline_all = baseline_for_rows(
            mode,
            families,
            global_action_mean,
            family_action_mean,
        )
        residual_all = matrix - baseline_all
        discovery_residual = residual_all[discovery_mask]

        calibration_metrics: list[EvalMetrics] = []
        heldout_by_rank: list[EvalMetrics] = []
        for rank in range(1, max_rank + 1):
            basis = fit_basis(discovery_residual, rank)
            calibration_metrics.append(
                evaluate_projection(
                    matrix[calibration_mask],
                    baseline_all[calibration_mask],
                    basis,
                    best_fixed_idx,
                )
            )
            heldout_by_rank.append(
                evaluate_projection(
                    matrix[heldout_mask],
                    baseline_all[heldout_mask],
                    basis,
                    best_fixed_idx,
                )
            )

        selected_rank = choose_rank(calibration_metrics, args.selection_max_rank)
        selected_heldout = heldout_by_rank[selected_rank - 1]

        mode_payload = {
            "discovery_spectrum": spectrum_summary(discovery_residual, max_rank),
            "calibration_by_rank": [asdict(x) for x in calibration_metrics],
            "selected_rank_from_calibration": selected_rank,
            "heldout_at_selected_rank": asdict(selected_heldout),
            "heldout_by_rank": [asdict(x) for x in heldout_by_rank],
        }

        if mode == "family_centered":
            mode_payload["parallel_analysis"] = parallel_analysis(
                discovery_residual,
                max_rank=max_rank,
                permutations=args.parallel_permutations,
                seed=args.seed + 100,
            )
            mode_payload["bootstrap_subspace_stability"] = bootstrap_subspace_stability(
                discovery_residual,
                rank=selected_rank,
                bootstrap=args.bootstrap,
                seed=args.seed + 200,
            )
            selected_family_rank = selected_rank
            heldout_family_metrics = selected_heldout

        modes[mode] = mode_payload

    assert selected_family_rank is not None
    assert heldout_family_metrics is not None

    primary_pass = {
        "selected_rank_le_4": selected_family_rank <= 4,
        "heldout_variance_explained_ge_0_70": (
            heldout_family_metrics.variance_explained >= 0.70
        ),
        "heldout_median_spearman_ge_0_60": (
            heldout_family_metrics.median_within_prompt_spearman >= 0.60
        ),
        "heldout_regret_fraction_le_0_30": (
            heldout_family_metrics.regret_fraction_of_headroom is not None
            and heldout_family_metrics.regret_fraction_of_headroom <= 0.30
        ),
    }
    halfscale_go = all(primary_pass.values())

    summary = {
        "status": "gate0b_halfscale_complete",
        "scope": "halfscale_structural_diagnostic_not_formal_gate0b_pass",
        "num_prompts": len(prompt_ids),
        "num_actions": len(action_ids),
        "split_counts": {
            "discovery": int(discovery_mask.sum()),
            "calibration": int(calibration_mask.sum()),
            "heldout": int(heldout_mask.sum()),
        },
        "action_ids": action_ids,
        "best_fixed_action_from_discovery": action_ids[best_fixed_idx],
        "best_fixed_discovery_mean_gain": float(global_action_mean[best_fixed_idx]),
        "family_centered_selected_rank": selected_family_rank,
        "primary_halfscale_checks": primary_pass,
        "recommendation": (
            "expand_to_formal_96x24_gate0b"
            if halfscale_go
            else "do_not_expand_yet_reinspect_low_rank_hypothesis"
        ),
        "modes": modes,
        "notes": [
            "Held-out row coordinates use the true held-out utility residual projected onto the discovery basis.",
            "This is an optimistic structural upper bound and is not an online no-answer selector.",
            "Gate 0C, not Gate 0B, tests whether sparse mechanical probes can identify latent coordinates.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
