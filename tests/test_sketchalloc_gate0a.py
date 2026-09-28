import pandas as pd
import pytest
import numpy as np

from evaluation.sketchalloc.analyze_gate0a import evaluate_gate0a
from evaluation.sketchalloc.audit_legacy_profile import reconstruct_lu_budget
from evaluation.sketchalloc.normalize_legacy_results import (
    audit_coverage,
    normalize_payload,
    stable_action_id,
)


def _payload(prompt_id="p0"):
    return {
        "id": prompt_id,
        "family": "retrieval",
        "task": "niah",
        "context_sha256": f"ctx-{prompt_id}",
        "context_tokens": 8000,
        "compression_ratio": 0.2,
        "conditions": [
            {"name": "fullkv", "answer_nll": 1.0},
            {"name": "lu_baseline", "answer_nll": 2.0},
            {
                "name": "legacy-arbitrary-name",
                "category": "boundary",
                "donor": {"layer": 2, "head": 1},
                "receiver": {"layer": 5, "head": 3},
                "amount": 16,
                "answer_nll": 1.75,
                "answer_nll_gain_vs_lu": 0.25,
                "lu_remove_cost": 0.1,
                "lu_next_gain": 0.2,
                "lu_marginal_delta": 0.1,
            },
        ],
    }


def test_normalizer_uses_structural_action_id():
    payload = _payload()
    rows = normalize_payload(payload, "p0.json")
    assert rows[0]["action_id"] == "L2H1->L5H3@16"
    assert rows[0]["gain"] == 0.25
    assert stable_action_id(payload["conditions"][2]) == rows[0]["action_id"]


def test_coverage_detects_non_rectangular_legacy_candidates():
    rows = normalize_payload(_payload("p0"))
    second = _payload("p1")
    second["conditions"][2]["receiver"]["head"] = 4
    rows += normalize_payload(second)
    audit = audit_coverage(pd.DataFrame(rows), ["p0.json", "p1.json"])
    assert not audit.complete_action_matrix
    assert audit.num_unique_actions == 2
    assert audit.num_common_actions == 0
    assert audit.missing_rectangular_cells == 2


def test_gate0a_separates_oracle_from_fixed_action():
    rows = []
    gains = {
        "p0": (1.0, -1.0),
        "p1": (0.8, -0.8),
        "p2": (-1.0, 1.0),
        "p3": (-0.8, 0.8),
    }
    for i, (prompt, values) in enumerate(gains.items()):
        for action, gain in zip(["a", "b"], values):
            rows.append(
                {
                    "prompt_id": prompt,
                    "action_id": action,
                    "gain": gain,
                    "family": "f0" if i % 2 == 0 else "f1",
                    "context_cluster": prompt,
                }
            )
    summary = evaluate_gate0a(pd.DataFrame(rows), n_bootstrap=200, seed=7)
    assert summary.complete_action_matrix
    assert summary.oracle_mean_gain == pytest.approx(0.9)
    assert summary.lopo_best_fixed_mean_gain < 0
    assert summary.pairwise_action_crossover_rate == 1.0
    assert summary.oracle_minus_best_fixed_ci95[0] > 0


def test_gate0a_refuses_incomplete_matrix():
    frame = pd.DataFrame(
        [
            {"prompt_id": "p0", "action_id": "a", "gain": 1, "family": "f", "context_cluster": "p0"},
            {"prompt_id": "p0", "action_id": "b", "gain": 0, "family": "f", "context_cluster": "p0"},
            {"prompt_id": "p1", "action_id": "a", "gain": 0, "family": "f", "context_cluster": "p1"},
            {"prompt_id": "p2", "action_id": "a", "gain": 0, "family": "f", "context_cluster": "p2"},
            {"prompt_id": "p3", "action_id": "a", "gain": 0, "family": "f", "context_cluster": "p3"},
        ]
    )
    summary = evaluate_gate0a(frame, n_bootstrap=20)
    assert summary.status == "incomplete_action_matrix"
    assert summary.missing_cells == 3
    assert summary.oracle_mean_gain is None


def test_reconstruct_lu_budget_conserves_layer_rounding_target():
    prune = np.array([[0.5, 0.45], [0.2, 0.8]], dtype=np.float64)
    budget = reconstruct_lu_budget(prune, seq_len=10)
    assert budget == [[5, 5], [8, 2]]
    assert [sum(row) for row in budget] == [10, 10]
