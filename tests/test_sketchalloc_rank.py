import numpy as np

from evaluation.sketchalloc.analyze_utility_rank import evaluate_rank_gate


def test_low_rank_utility_is_recovered_from_sparse_sentinels():
    rng = np.random.default_rng(7)
    num_train, num_test, num_actions, rank = 80, 20, 18, 3
    basis, _ = np.linalg.qr(rng.normal(size=(num_actions, rank)))
    prior = rng.normal(scale=0.1, size=num_actions)
    train = prior + rng.normal(size=(num_train, rank)) @ basis.T
    test = prior + rng.normal(size=(num_test, rank)) @ basis.T

    summary = evaluate_rank_gate(
        train,
        test,
        [f"a{i}" for i in range(num_actions)],
        rank=rank,
        num_sentinels=6,
    )

    assert summary.heldout_variance_explained > 0.99
    assert summary.heldout_rmse < 1e-3
    assert summary.median_within_prompt_spearman > 0.99


def test_sentinel_count_must_cover_latent_rank():
    train = np.eye(4)
    test = np.eye(4)
    try:
        evaluate_rank_gate(train, test, ["a", "b", "c", "d"], rank=3, num_sentinels=2)
    except ValueError as exc:
        assert "at least the requested rank" in str(exc)
    else:
        raise AssertionError("expected a ValueError")
