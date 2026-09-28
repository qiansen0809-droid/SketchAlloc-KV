from evaluation.sketchalloc.metrics import (
    gate0_task_score,
    max_qa_f1_score,
    qa_f1_score,
    ruler_string_match_score,
)


def test_longbench_qa_f1_matches_normalized_token_overlap():
    assert qa_f1_score("The Eiffel Tower", "Eiffel Tower") == 1.0
    assert qa_f1_score("Paris France", "Paris") == 2.0 / 3.0


def test_longbench_uses_best_reference():
    score = max_qa_f1_score(
        "New York City",
        ["NYC", "New York City"],
    )
    assert score == 1.0


def test_ruler_string_match_requires_all_references_fractionally():
    assert ruler_string_match_score("alpha beta", ["alpha", "beta"]) == 1.0
    assert ruler_string_match_score("alpha only", ["alpha", "beta"]) == 0.5


def test_gate0_family_metric_dispatch():
    assert gate0_task_score(
        "ruler_retrieval",
        "the code is 12345",
        ["12345"],
    ) == 1.0
    assert gate0_task_score(
        "longbench_single",
        "Eiffel Tower",
        ["The Eiffel Tower"],
    ) == 1.0
