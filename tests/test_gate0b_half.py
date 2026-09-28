import numpy as np
import pandas as pd

from evaluation.sketchalloc.analyze_gate0b_half import (
    baseline_for_rows,
    choose_rank,
    discovery_baselines,
    evaluate_projection,
    fit_basis,
    load_matrix,
)


def _make_frame(seed=7):
    rng = np.random.default_rng(seed)
    families = ["ruler_retrieval", "longbench_single", "longbench_multi"]
    actions = [f"a{i}" for i in range(12)]
    basis, _ = np.linalg.qr(rng.normal(size=(12, 3)))

    rows = []
    prompt_index = 0
    for family in families:
        family_prior = rng.normal(scale=0.05, size=12)
        for split, count in [("discovery", 8), ("calibration", 4), ("heldout", 4)]:
            for _ in range(count):
                latent = rng.normal(size=3)
                utility = family_prior + latent @ basis.T
                prompt_id = f"p{prompt_index:03d}"
                prompt_index += 1
                for action_id, gain in zip(actions, utility):
                    rows.append(
                        {
                            "prompt_id": prompt_id,
                            "action_id": action_id,
                            "gain": float(gain),
                            "family": family,
                            "split": split,
                        }
                    )
    return pd.DataFrame(rows)


def test_halfscale_matrix_loads_expected_shape():
    frame = _make_frame()
    matrix, prompt_ids, action_ids, families, splits = load_matrix(frame)

    assert matrix.shape == (48, 12)
    assert len(prompt_ids) == 48
    assert len(action_ids) == 12
    assert (splits == "discovery").sum() == 24
    assert (splits == "calibration").sum() == 12
    assert (splits == "heldout").sum() == 12


def test_family_centered_low_rank_projection_recovers_synthetic_structure():
    frame = _make_frame()
    matrix, _, _, families, splits = load_matrix(frame)
    discovery = splits == "discovery"
    heldout = splits == "heldout"

    global_mean, family_mean = discovery_baselines(matrix, families, discovery)
    baseline = baseline_for_rows(
        "family_centered",
        families,
        global_mean,
        family_mean,
    )
    residual_discovery = matrix[discovery] - baseline[discovery]
    basis = fit_basis(residual_discovery, rank=3)

    best_fixed_idx = int(np.argmax(global_mean))
    metrics = evaluate_projection(
        matrix[heldout],
        baseline[heldout],
        basis,
        best_fixed_idx,
    )

    assert metrics.variance_explained > 0.95
    assert metrics.median_within_prompt_spearman > 0.95
    assert metrics.mean_top_action_regret < 1e-6


def test_choose_rank_prefers_smallest_near_best_rank():
    from evaluation.sketchalloc.analyze_gate0b_half import EvalMetrics

    def metric(rank, r2):
        return EvalMetrics(
            rank=rank,
            variance_explained=r2,
            rmse=0.0,
            median_within_prompt_spearman=1.0,
            best_fixed_mean_gain=0.0,
            selected_mean_gain=1.0,
            oracle_mean_gain=1.0,
            mean_top_action_regret=0.0,
            oracle_over_best_fixed_headroom=1.0,
            regret_fraction_of_headroom=0.0,
        )

    metrics = [
        metric(1, 0.50),
        metric(2, 0.78),
        metric(3, 0.81),
        metric(4, 0.82),
    ]
    assert choose_rank(metrics, selection_max_rank=4) == 2
