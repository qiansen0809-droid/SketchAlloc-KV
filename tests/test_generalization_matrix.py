from pathlib import Path

from evaluation.sketchalloc.generalization_matrix import (
    FULL_LENGTHS,
    SCREENING_LENGTHS,
    SCREENING_TASKS,
    build_specs,
    family_for_task,
)


def test_screening_matrix_covers_all_three_shift_axes():
    specs = build_specs(
        [("source", "/tmp/source.npy")],
        Path("results/generalization"),
        dimension="all",
        mode="screening",
    )
    dims = {s.dimension for s in specs}
    assert dims == {"domain", "length", "compression"}

    domain = [s for s in specs if s.dimension == "domain"]
    compression = [s for s in specs if s.dimension == "compression"]
    length = [s for s in specs if s.dimension == "length"]

    assert len(domain) == len(SCREENING_TASKS)
    assert len(compression) == len(SCREENING_TASKS) * 3
    assert len(length) == len(SCREENING_LENGTHS) * 2


def test_full_mode_includes_longer_ruler_context():
    specs = build_specs(
        [("source", "/tmp/source.npy")],
        Path("results/generalization"),
        dimension="length",
        mode="full",
    )
    assert {s.context_length for s in specs} == set(FULL_LENGTHS)


def test_task_family_mapping_is_stable():
    assert family_for_task("qasper") == "single_doc_qa"
    assert family_for_task("hotpotqa") == "multi_doc_qa"
    assert family_for_task("gov_report") == "summarization"
    assert family_for_task("repobench-p") == "code"
