import numpy as np

from evaluation.curve_data.step2_compute_curve import exact_runtime_keep_counts


def test_exact_runtime_keep_counts_matches_lu_fractional_rounding():
    ratios = np.array([0.50, 0.5004, 0.2501], dtype=np.float64)
    counts = exact_runtime_keep_counts(ratios, length=10)

    # ideal keeps = [5.0, 4.996, 7.499], rounded layer total = 17.
    # floor gives [5, 4, 7], so the single remainder goes to head 1,
    # which has the largest fractional part.
    assert counts.tolist() == [5, 5, 7]
    assert int(counts.sum()) == 17


def test_exact_runtime_keep_counts_clamps_to_valid_range():
    ratios = np.array([1.0, 0.0], dtype=np.float64)
    counts = exact_runtime_keep_counts(ratios, length=8)

    assert counts.tolist() == [1, 8]
