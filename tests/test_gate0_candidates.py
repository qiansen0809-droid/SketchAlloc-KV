import numpy as np
import pytest
import torch

from kvpress import LUPress, SnapKVPress

from evaluation.sketchalloc.build_candidates import (
    SwapCandidate,
    Unit,
    apply_swap,
    generate_lu_boundary_swaps,
    generate_poc_swaps,
    load_lu_marginal_slice,
    total_budget,
)


def test_apply_swap_preserves_budget():
    baseline = [[10, 20], [30, 40]]
    candidate = SwapCandidate(Unit(1, 1), Unit(0, 0), 5)

    swapped = apply_swap(baseline, candidate, min_keep=1, max_keep=100)

    assert total_budget(swapped) == total_budget(baseline)
    assert swapped[1][1] == 35
    assert swapped[0][0] == 15


def test_apply_swap_rejects_min_keep_violation():
    baseline = [[8, 20]]
    candidate = SwapCandidate(Unit(0, 0), Unit(0, 1), 4)

    try:
        apply_swap(baseline, candidate, min_keep=5, max_keep=100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected min_keep violation")


def test_poc_candidates_are_budget_preserving():
    baseline = [[20, 40], [30, 10]]
    candidates = generate_poc_swaps(
        baseline,
        amount=5,
        num_candidates=3,
        min_keep=5,
        max_keep=50,
    )

    assert len(candidates) == 3
    for candidate in candidates:
        swapped = apply_swap(
            baseline,
            candidate,
            min_keep=5,
            max_keep=50,
        )
        assert total_budget(swapped) == total_budget(baseline)


def test_lu_boundary_candidates_cover_registered_categories():
    baseline = [[40, 40], [40, 40]]
    remove_cost = np.array(
        [
            [1.0, 4.0],
            [2.0, 3.0],
        ]
    )
    next_gain = np.array(
        [
            [5.0, 1.0],
            [4.0, 2.0],
        ]
    )

    candidates, donors, receivers = generate_lu_boundary_swaps(
        baseline,
        remove_cost=remove_cost,
        next_gain=next_gain,
        amount=5,
        num_candidates=6,
        min_keep=10,
        max_keep=100,
    )

    assert len(candidates) == 6
    assert donors
    assert receivers

    categories = {candidate.category for candidate in candidates}
    assert "lu_promising" in categories
    assert "near_boundary" in categories
    assert "low_priority_control" in categories

    promising = [c for c in candidates if c.category == "lu_promising"]
    boundary = [c for c in candidates if c.category == "near_boundary"]
    control = [c for c in candidates if c.category == "low_priority_control"]

    assert len(promising) == 2
    assert len(boundary) == 3
    assert len(control) == 1
    assert all(c.lu_marginal_delta > 0 for c in promising)
    assert any(c.lu_marginal_delta > 0 for c in boundary)
    assert any(c.lu_marginal_delta < 0 for c in boundary)
    assert control[0].lu_marginal_delta < 0

    for candidate in candidates:
        assert candidate.donor != candidate.receiver
        assert candidate.lu_marginal_delta == (
            candidate.lu_next_gain - candidate.lu_remove_cost
        )

        swapped = apply_swap(
            baseline,
            candidate,
            min_keep=10,
            max_keep=100,
        )
        assert total_budget(swapped) == total_budget(baseline)


def test_load_lu_marginal_slice(tmp_path):
    path = tmp_path / "marginals.npz"
    ratios = np.arange(1, 100, dtype=np.float64) / 100.0

    remove = np.zeros((99, 2, 2), dtype=np.float64)
    gain = np.ones((99, 2, 2), dtype=np.float64)
    remove[79] = 3.0
    gain[79] = 7.0

    np.savez_compressed(
        path,
        remove_cost=remove,
        next_gain=gain,
        global_compression_ratio=ratios,
        marginal_step_tokens=np.array(16),
        sink_size=np.array(4),
        window_size=np.array(32),
        calibration_pairs=np.array(30),
    )

    profile = load_lu_marginal_slice(
        path,
        compression_ratio=0.80,
        expected_step_tokens=16,
    )

    assert profile.global_compression_ratio == 0.80
    assert profile.marginal_step_tokens == 16
    assert profile.sink_size == 4
    assert profile.window_size == 32
    assert profile.calibration_pairs == 30
    assert np.all(profile.remove_cost == 3.0)
    assert np.all(profile.next_gain == 7.0)



def test_lu_override_allows_unchanged_official_budget_below_protected_span(tmp_path):
    curve = np.zeros((99, 1, 2), dtype=np.float32)
    curve[79, 0] = np.array([0.99, 0.985], dtype=np.float32)
    path = tmp_path / "curve.npy"
    np.save(path, curve)

    press = LUPress(
        press=SnapKVPress(compression_ratio=0.80),
        budget_curve_path=str(path),
        sink=4,
        window=32,
    )
    press._post_setup_init()

    baseline = press.get_keep_counts(
        layer_idx=0,
        num_heads=2,
        seq_len=2048,
        device=torch.device("cpu"),
    )
    assert baseline.tolist() == [20, 31]

    press.set_keep_counts_override({0: baseline.tolist()})
    unchanged = press.get_keep_counts(
        layer_idx=0,
        num_heads=2,
        seq_len=2048,
        device=torch.device("cpu"),
    )
    assert unchanged.tolist() == [20, 31]

    press.set_keep_counts_override({0: [21, 31]})
    upward = press.get_keep_counts(
        layer_idx=0,
        num_heads=2,
        seq_len=2048,
        device=torch.device("cpu"),
    )
    assert upward.tolist() == [21, 31]

    press.set_keep_counts_override({0: [19, 31]})
    with pytest.raises(ValueError, match="donor override"):
        press.get_keep_counts(
            layer_idx=0,
            num_heads=2,
            seq_len=2048,
            device=torch.device("cpu"),
        )
