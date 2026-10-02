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


def test_exact_runtime_keep_counts_matches_torch_float32_boundary():
    ratios = np.array([
        0.7226096, 0.7133434, 0.8101599, 0.7055573,
        0.77900493, 0.8791592, 0.7937769, 0.8993674,
    ], dtype=np.float32)

    counts = exact_runtime_keep_counts(ratios, length=2048)

    # Float64 NumPy arithmetic gives one fewer entry to head 5 here. Runtime
    # LUPress uses torch.float32 and keeps 248, so the offline helper must too.
    assert counts.tolist() == [568, 587, 389, 603, 453, 248, 422, 206]
