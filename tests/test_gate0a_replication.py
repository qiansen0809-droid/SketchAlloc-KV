import json

import numpy as np

from evaluation.sketchalloc.build_candidates import SwapCandidate, Unit
from evaluation.sketchalloc.run_gate0a_replication import (
    curve_budget_from_array,
    load_actions,
    validate_all_legality,
)


def test_curve_budget_from_array_preserves_rounded_layer_total():
    curve = np.zeros((99, 2, 3), dtype=np.float32)
    curve[79, 0] = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    curve[79, 1] = np.array([0.7, 0.8, 0.9], dtype=np.float32)

    budget = curve_budget_from_array(curve, 0.80, 100)

    assert sum(budget[0]) == 60
    assert sum(budget[1]) == 60


def test_load_actions_round_trips_frozen_ids(tmp_path):
    path = tmp_path / "actions.json"
    path.write_text(
        json.dumps(
            {
                "compression_ratio": 0.8,
                "swap_size": 16,
                "actions": [
                    {
                        "action_id": "L0H0->L1H1@16",
                        "donor_layer": 0,
                        "donor_head": 0,
                        "receiver_layer": 1,
                        "receiver_head": 1,
                        "amount": 16,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    actions, manifest = load_actions(path)

    assert manifest["swap_size"] == 16
    assert actions == [
        SwapCandidate(
            donor=Unit(0, 0),
            receiver=Unit(1, 1),
            amount=16,
            category="fixed_confirmatory",
        )
    ]


def test_legality_audit_rejects_bad_fixed_action(tmp_path):
    curve = np.zeros((99, 1, 2), dtype=np.float32)
    curve[79, 0] = np.array([0.8, 0.8], dtype=np.float32)
    path = tmp_path / "curve.npy"
    np.save(path, curve)

    rows = [{"id": "p0", "context_tokens": 100}]
    actions = [
        SwapCandidate(
            donor=Unit(0, 0),
            receiver=Unit(0, 1),
            amount=16,
            category="fixed_confirmatory",
        )
    ]

    try:
        validate_all_legality(rows, actions, str(path), 0.80)
    except ValueError as exc:
        assert "No replication labels were computed" in str(exc)
    else:
        raise AssertionError("expected frozen action legality failure")


def test_balanced_sampler_allows_one_exhausted_task():
    from evaluation.sketchalloc.prepare_gate0a_replication import _select_balanced_from_pools

    def row(task, idx, tokens):
        return {
            "task": task,
            "source_index": idx,
            "context_tokens": tokens,
            "context_sha256": f"{task}-{idx}",
        }

    pools = {
        "narrativeqa": [],
        "qasper": [row("qasper", i, 8192 + i) for i in range(8)],
        "multifieldqa_en": [row("multifieldqa_en", i, 8192 - i) for i in range(8)],
    }

    selected, diag = _select_balanced_from_pools(
        pools,
        ["narrativeqa", "qasper", "multifieldqa_en"],
        target=8192,
        total_size=8,
    )

    assert len(selected) == 8
    assert diag["excluded_tasks_insufficient_fresh_support"] == {"narrativeqa": 0}
    assert diag["selected_task_counts"] == {
        "qasper": 4,
        "multifieldqa_en": 4,
    }
