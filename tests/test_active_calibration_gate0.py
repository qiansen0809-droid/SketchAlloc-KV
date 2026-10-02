import argparse
import json

import numpy as np
import pytest

from evaluation.active_calibration.gate0 import (
    PairData,
    discover_pairs,
    evaluate_profile,
    run,
)


def test_static_average_is_not_raw_question_data(tmp_path):
    np.save(tmp_path / "snapkv_avg_ratio.npy", np.zeros((99, 2, 2)))
    with pytest.raises(FileNotFoundError, match="NOT per-question raw data"):
        discover_pairs(tmp_path, "snapkv")


def test_proxy_loss_respects_head_budget():
    pair = PairData(
        pair_id="toy",
        prune_ratio=np.zeros((1, 2)),
        prefix_utility=np.array([[[0, 5, 9, 12, 14, 15, 16], [0, 1, 2, 3, 4, 5, 6]]]),
        context_length=6,
    )
    favor_valuable_head = np.array([[1 / 3, 2 / 3]])
    favor_other_head = np.array([[2 / 3, 1 / 3]])
    valuable = evaluate_profile(favor_valuable_head, [pair], sink=1, window=1)
    other = evaluate_profile(favor_other_head, [pair], sink=1, window=1)
    assert valuable["mean_proxy_loss"] < other["mean_proxy_loss"]
    assert valuable["mean_kept_entries"] == other["mean_kept_entries"] == 6


def test_gate0_has_disjoint_holdout_and_reproducible_outputs(tmp_path):
    raw = tmp_path / "raw"
    context = raw / "context_0"
    context.mkdir(parents=True)
    scorer = np.broadcast_to(
        np.linspace(0.1, 1.0, 12, dtype=np.float32), (2, 2, 12)
    ).copy()
    np.save(context / "snapkv.npy", scorer)
    for question in range(6):
        oracle = np.ones((2, 2, 12), dtype=np.float32)
        oracle[0, question % 2, 2:10] += question + 1
        oracle[1, (question + 1) % 2, 2:10] += 6 - question
        np.save(context / f"question_{question}.npy", oracle)

    out = tmp_path / "output"
    args = argparse.Namespace(
        raw_root=raw,
        output_dir=out,
        method="snapkv",
        sink=1,
        window=1,
        compression=0.5,
        threshold=99,
        calibration_size=4,
        subset_sizes="2,3",
        repeats=3,
        seed=11,
    )
    result = run(args)
    assert len(result["calibration_ids"]) == 4
    assert len(result["heldout_ids"]) == 2
    assert set(result["calibration_ids"]).isdisjoint(result["heldout_ids"])
    assert len(result["random_subset_records"]) == 6
    assert np.load(out / "full_calibration_profile.npy").shape == (2, 2)
    assert json.loads((out / "gate0_results.json").read_text(encoding="utf-8"))[
        "heldout_ids"
    ] == result["heldout_ids"]

    args.output_dir = tmp_path / "second_output"
    second = run(args)
    assert second["random_subset_summary"] == result["random_subset_summary"]

