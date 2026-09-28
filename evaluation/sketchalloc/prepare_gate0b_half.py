"""Prepare the fresh 48-prompt half-scale Gate 0B dataset.

The sampler is label-free and excludes every prompt/source/context used by the
legacy discovery and Gate 0A confirmatory experiments.  The half-scale split is
24 discovery / 12 calibration / 12 held-out, with each broad family contributing
8 / 4 / 4 prompts respectively.

For RULER, task/template families are kept split-disjoint: four retrieval tasks
feed discovery, two feed calibration, and two feed held-out, with two fresh
prompts per task.  LongBench uses distinct natural contexts and balances task
counts as far as the available fresh 8K-window support permits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from evaluation.sketchalloc.prepare_gate0a_replication import (
    eligible_sorted,
    is_excluded,
    load_exclusions,
)
from evaluation.sketchalloc.prepare_minigate import (
    MULTI_DOC_TASKS,
    RULER_RETRIEVAL_TASKS,
    SINGLE_DOC_TASKS,
    make_record,
)


FAMILY_SIZE = 16
FAMILY_SPLIT_TARGETS = {"discovery": 8, "calibration": 4, "heldout": 4}
MIN_TASK_SUPPORT = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--exclude-raw-dir",
        type=Path,
        action="append",
        required=True,
        help="Repeat for every prior raw-result directory that must be excluded.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/sketchalloc/gate0b/half48.jsonl"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/sketchalloc/gate0b/half48_manifest.json"),
    )
    p.add_argument("--target-context-tokens", type=int, default=8192)
    p.add_argument("--min-context-tokens", type=int, default=6144)
    p.add_argument("--max-context-tokens", type=int, default=9216)
    p.add_argument("--ruler-revision", default="24adceac8a0e6532936e8d721cd9e9084d2e4686")
    p.add_argument("--longbench-revision", default="0ce23c4aa955accf17527097eb12a8f00e2743e6")
    p.add_argument("--ruler-local-parquet", type=Path, default=None)
    return p.parse_args()


def union_exclusions(raw_dirs: list[Path]):
    prompt_ids: set[str] = set()
    context_hashes: set[str] = set()
    source_keys: set[tuple[str, str, int]] = set()
    per_dir = {}

    for raw_dir in raw_dirs:
        ids, hashes, sources = load_exclusions(raw_dir)
        prompt_ids.update(ids)
        context_hashes.update(hashes)
        source_keys.update(sources)
        per_dir[str(raw_dir)] = {
            "prompt_ids": len(ids),
            "context_hashes": len(hashes),
            "source_keys": len(sources),
        }

    return (prompt_ids, context_hashes, source_keys), per_dir


def load_ruler(tokenizer, revision, target, exclusions, local_parquet=None):
    if local_parquet is not None:
        ds = load_dataset("parquet", data_files={"test": str(local_parquet)}, split="test")
    else:
        ds = load_dataset("simonjegou/ruler", "8192", split="test", revision=revision)

    by_task: dict[str, list[dict]] = {task: [] for task in RULER_RETRIEVAL_TASKS}
    for i, row in enumerate(ds):
        task = str(row["task"])
        if task not in by_task:
            continue
        record = make_record(
            family="ruler_retrieval",
            task=task,
            source_dataset="simonjegou/ruler",
            source_config="8192",
            source_revision=revision,
            source_index=i,
            row=row,
            tokenizer=tokenizer,
        )
        if not is_excluded(record, *exclusions):
            by_task[task].append(record)

    selected: list[dict] = []
    task_split = {}
    for task_idx, task in enumerate(RULER_RETRIEVAL_TASKS):
        if task_idx < 4:
            split = "discovery"
        elif task_idx < 6:
            split = "calibration"
        else:
            split = "heldout"
        task_split[task] = split

        candidates = sorted(
            by_task[task],
            key=lambda row: (
                abs(int(row["context_tokens"]) - target),
                int(row["source_index"]),
            ),
        )
        seen_contexts: set[str] = set()
        fresh = []
        for row in candidates:
            if row["context_sha256"] in seen_contexts:
                continue
            fresh.append(row)
            seen_contexts.add(row["context_sha256"])
            if len(fresh) == 2:
                break
        if len(fresh) < 2:
            raise RuntimeError(
                f"RULER task {task}: only {len(fresh)} fresh distinct prompts remain; need 2"
            )
        for row in fresh:
            row = dict(row)
            row["split"] = split
            selected.append(row)

    return selected, {
        "task_split": task_split,
        "fresh_pool_sizes": {task: len(rows) for task, rows in by_task.items()},
        "selection_rule": (
            "two fresh prompts per retrieval task; task/template families are "
            "split-disjoint (4 tasks discovery, 2 calibration, 2 heldout); "
            "within task choose closest context lengths to 8192; no model outputs used"
        ),
    }


def select_longbench_family(
    tokenizer,
    tasks,
    family,
    revision,
    target,
    low,
    high,
    exclusions,
):
    pools: dict[str, list[dict]] = {}
    for task in tasks:
        ds = load_dataset("Xnhyacinth/LongBench", task, split="test", revision=revision)
        records = [
            make_record(
                family=family,
                task=task,
                source_dataset="Xnhyacinth/LongBench",
                source_config=task,
                source_revision=revision,
                source_index=i,
                row=row,
                tokenizer=tokenizer,
            )
            for i, row in enumerate(ds)
        ]
        pools[task] = eligible_sorted(records, target, low, high, exclusions)

    eligible_tasks = [task for task in tasks if len(pools[task]) >= MIN_TASK_SUPPORT]
    if len(eligible_tasks) < 2:
        raise RuntimeError(
            f"{family}: fresh support too narrow after exclusions: "
            f"{ {task: len(pools[task]) for task in tasks} }"
        )

    selected: list[dict] = []
    used_contexts: set[str] = set()
    taken = Counter()
    cursors = {task: 0 for task in eligible_tasks}

    while len(selected) < FAMILY_SIZE:
        choices = []
        for task in eligible_tasks:
            cursor = cursors[task]
            while cursor < len(pools[task]) and pools[task][cursor]["context_sha256"] in used_contexts:
                cursor += 1
            cursors[task] = cursor
            if cursor >= len(pools[task]):
                continue
            row = pools[task][cursor]
            choices.append(
                (
                    taken[task],
                    abs(int(row["context_tokens"]) - target),
                    task,
                    int(row["source_index"]),
                    row,
                )
            )
        if not choices:
            break
        _, _, task, _, row = min(choices, key=lambda item: item[:4])
        selected.append(dict(row))
        used_contexts.add(row["context_sha256"])
        taken[task] += 1
        cursors[task] += 1

    if len(selected) != FAMILY_SIZE:
        raise RuntimeError(
            f"{family}: only {len(selected)} fresh globally distinct prompts available; "
            f"need {FAMILY_SIZE}"
        )

    # Assign 8/4/4 without consulting labels, while balancing task composition.
    remaining = dict(FAMILY_SPLIT_TARGETS)
    per_split_task = defaultdict(Counter)
    assigned: list[dict] = []
    rows_by_task = defaultdict(list)
    for row in selected:
        rows_by_task[row["task"]].append(row)

    # Interleave tasks so no split is filled by one task merely due to source order.
    interleaved = []
    while any(rows_by_task.values()):
        for task in sorted(rows_by_task):
            if rows_by_task[task]:
                interleaved.append(rows_by_task[task].pop(0))

    for row in interleaved:
        candidates = [
            split for split, capacity in remaining.items()
            if capacity > 0
        ]
        split = min(
            candidates,
            key=lambda s: (
                per_split_task[s][row["task"]],
                -remaining[s],
                {"discovery": 0, "calibration": 1, "heldout": 2}[s],
            ),
        )
        out = dict(row)
        out["split"] = split
        assigned.append(out)
        remaining[split] -= 1
        per_split_task[split][row["task"]] += 1

    if any(remaining.values()):
        raise AssertionError(f"{family}: split assignment incomplete: {remaining}")

    return assigned, {
        "pool_sizes_after_exclusion": {task: len(pools[task]) for task in tasks},
        "eligible_tasks": eligible_tasks,
        "excluded_tasks_insufficient_fresh_support": {
            task: len(pools[task]) for task in tasks if task not in eligible_tasks
        },
        "selected_task_counts": dict(Counter(row["task"] for row in assigned)),
        "split_task_counts": {
            split: dict(Counter(row["task"] for row in assigned if row["split"] == split))
            for split in FAMILY_SPLIT_TARGETS
        },
        "selection_rule": (
            "keep the frozen 6144-9216 token window; exclude all prior prompt/source/context "
            "overlap; choose 16 distinct contexts approximately evenly across tasks with >=2 "
            "fresh contexts; assign 8/4/4 discovery/calibration/heldout without labels while "
            "balancing task counts"
        ),
    }


def main() -> None:
    args = parse_args()
    exclusions, exclusion_diag = union_exclusions(args.exclude_raw_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ruler, ruler_diag = load_ruler(
        tokenizer,
        args.ruler_revision,
        args.target_context_tokens,
        exclusions,
        args.ruler_local_parquet,
    )
    single, single_diag = select_longbench_family(
        tokenizer,
        SINGLE_DOC_TASKS,
        "longbench_single",
        args.longbench_revision,
        args.target_context_tokens,
        args.min_context_tokens,
        args.max_context_tokens,
        exclusions,
    )
    multi, multi_diag = select_longbench_family(
        tokenizer,
        MULTI_DOC_TASKS,
        "longbench_multi",
        args.longbench_revision,
        args.target_context_tokens,
        args.min_context_tokens,
        args.max_context_tokens,
        exclusions,
    )

    records = ruler + single + multi
    if len(records) != 48:
        raise AssertionError(f"expected 48 prompts, got {len(records)}")

    family_counts = Counter(row["family"] for row in records)
    split_counts = Counter(row["split"] for row in records)
    family_split_counts = {
        family: dict(Counter(row["split"] for row in records if row["family"] == family))
        for family in sorted(family_counts)
    }

    expected_families = {
        "ruler_retrieval": 16,
        "longbench_single": 16,
        "longbench_multi": 16,
    }
    expected_splits = {"discovery": 24, "calibration": 12, "heldout": 12}
    if dict(family_counts) != expected_families:
        raise AssertionError(f"family counts mismatch: {dict(family_counts)}")
    if dict(split_counts) != expected_splits:
        raise AssertionError(f"split counts mismatch: {dict(split_counts)}")
    for family, counts in family_split_counts.items():
        if counts != FAMILY_SPLIT_TARGETS:
            raise AssertionError(f"{family} split counts mismatch: {counts}")

    old_ids, old_hashes, old_sources = exclusions
    for row in records:
        if is_excluded(row, old_ids, old_hashes, old_sources):
            raise AssertionError(f"leakage detected after selection: {row['id']}")

    # No context/document proxy may cross splits.
    context_to_splits = defaultdict(set)
    for row in records:
        context_to_splits[row["context_sha256"]].add(row["split"])
    leaking_contexts = {
        h: sorted(splits) for h, splits in context_to_splits.items() if len(splits) > 1
    }
    if leaking_contexts:
        raise AssertionError(f"context clusters cross splits: {leaking_contexts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "protocol": "SketchAlloc-KV Gate 0B half-scale structural matrix",
        "selection_uses_model_outputs": False,
        "num_prompts": len(records),
        "family_counts": dict(family_counts),
        "split_counts": dict(split_counts),
        "family_split_counts": family_split_counts,
        "target_context_tokens": args.target_context_tokens,
        "longbench_context_token_window": [
            args.min_context_tokens,
            args.max_context_tokens,
        ],
        "exclusions": exclusion_diag,
        "ruler_selection": ruler_diag,
        "longbench_single_selection": single_diag,
        "longbench_multi_selection": multi_diag,
        "samples": [
            {
                key: row[key]
                for key in (
                    "id",
                    "context_sha256",
                    "family",
                    "task",
                    "split",
                    "source_dataset",
                    "source_config",
                    "source_revision",
                    "source_index",
                    "context_tokens",
                    "query_tokens",
                    "prompt_tokens",
                )
            }
            for row in records
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote Gate 0B half-scale data: {args.output}")
    print(f"Wrote manifest:                {args.manifest}")
    print("Family counts:", dict(family_counts))
    print("Split counts:", dict(split_counts))
    print("Family/split counts:", family_split_counts)
    for row in records:
        print(
            row["split"],
            row["id"],
            f"context={row['context_tokens']}",
            f"query={row['query_tokens']}",
        )


if __name__ == "__main__":
    main()
