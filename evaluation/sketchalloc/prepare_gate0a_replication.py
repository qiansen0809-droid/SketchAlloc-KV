"""Prepare a fresh non-overlapping 24-prompt Gate 0A replication set.

Selection is label-free. It reuses the original MiniGate task families and
length rules, but excludes every legacy discovery prompt/context supplied via
--exclude-raw-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from evaluation.sketchalloc.prepare_minigate import (
    LONG_BENCH_FAMILY_SIZE,
    LONG_BENCH_MIN_PER_TASK,
    MULTI_DOC_TASKS,
    RULER_RETRIEVAL_TASKS,
    SINGLE_DOC_TASKS,
    make_record,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--exclude-raw-dir", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/sketchalloc/gate0a/confirm24.jsonl"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/sketchalloc/gate0a/confirm24_manifest.json"),
    )
    p.add_argument("--target-context-tokens", type=int, default=8192)
    p.add_argument("--min-context-tokens", type=int, default=6144)
    p.add_argument("--max-context-tokens", type=int, default=9216)
    p.add_argument("--ruler-revision", default="24adceac8a0e6532936e8d721cd9e9084d2e4686")
    p.add_argument("--longbench-revision", default="0ce23c4aa955accf17527097eb12a8f00e2743e6")
    p.add_argument("--ruler-local-parquet", type=Path, default=None)
    return p.parse_args()


def load_exclusions(raw_dir: Path) -> tuple[set[str], set[str], set[tuple[str, str, int]]]:
    prompt_ids: set[str] = set()
    context_hashes: set[str] = set()
    source_keys: set[tuple[str, str, int]] = set()

    for path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or "id" not in payload:
            continue
        prompt_ids.add(str(payload["id"]))
        if payload.get("context_sha256"):
            context_hashes.add(str(payload["context_sha256"]))
        if payload.get("source_dataset") is not None and payload.get("source_config") is not None and payload.get("source_index") is not None:
            source_keys.add(
                (
                    str(payload["source_dataset"]),
                    str(payload["source_config"]),
                    int(payload["source_index"]),
                )
            )
    if not prompt_ids:
        raise ValueError(f"no legacy payloads found under {raw_dir}")
    return prompt_ids, context_hashes, source_keys


def is_excluded(row: dict, prompt_ids: set[str], context_hashes: set[str], source_keys: set[tuple[str, str, int]]) -> bool:
    if row["id"] in prompt_ids:
        return True
    if row["context_sha256"] in context_hashes:
        return True
    key = (str(row["source_dataset"]), str(row["source_config"]), int(row["source_index"]))
    return key in source_keys


def eligible_sorted(records: list[dict], target: int, low: int, high: int, exclusions) -> list[dict]:
    prompt_ids, context_hashes, source_keys = exclusions
    out = [
        row for row in records
        if low <= row["context_tokens"] <= high
        and not is_excluded(row, prompt_ids, context_hashes, source_keys)
    ]
    out.sort(key=lambda row: (abs(row["context_tokens"] - target), row["source_index"]))
    deduped: list[dict] = []
    seen: set[str] = set()
    for row in out:
        if row["context_sha256"] in seen:
            continue
        deduped.append(row)
        seen.add(row["context_sha256"])
    return deduped


def _select_balanced_from_pools(
    pools: dict[str, list[dict]],
    tasks: list[str],
    target: int,
    total_size: int,
    min_rows_to_represent_task: int = LONG_BENCH_MIN_PER_TASK,
) -> tuple[list[dict], dict]:
    """Select a label-free, approximately balanced family sample.

    A task is represented only when enough fresh, in-window, distinct contexts
    remain after legacy exclusions. This avoids widening the token window or
    reusing old contexts merely to preserve a task quota that the fresh support
    cannot satisfy.
    """

    pool_sizes = {task: len(pools[task]) for task in tasks}
    eligible_tasks = [
        task for task in tasks
        if len(pools[task]) >= min_rows_to_represent_task
    ]
    excluded_tasks = {
        task: len(pools[task])
        for task in tasks
        if task not in eligible_tasks
    }

    if len(eligible_tasks) < 2:
        raise RuntimeError(
            "fresh LongBench support is too narrow after exclusions: "
            f"pool_sizes={pool_sizes}; need at least two tasks with "
            f">={min_rows_to_represent_task} fresh contexts each"
        )

    selected: list[dict] = []
    seen_contexts: set[str] = set()
    per_task_selected = {task: 0 for task in eligible_tasks}
    cursors = {task: 0 for task in eligible_tasks}

    while len(selected) < total_size:
        candidates: list[tuple[int, int, str, int, dict]] = []
        for task in eligible_tasks:
            cursor = cursors[task]
            while cursor < len(pools[task]) and pools[task][cursor]["context_sha256"] in seen_contexts:
                cursor += 1
            cursors[task] = cursor
            if cursor >= len(pools[task]):
                continue

            row = pools[task][cursor]
            candidates.append(
                (
                    per_task_selected[task],
                    abs(int(row["context_tokens"]) - target),
                    task,
                    int(row["source_index"]),
                    row,
                )
            )

        if not candidates:
            break

        _, _, task, _, row = min(candidates, key=lambda item: item[:4])
        selected.append(row)
        seen_contexts.add(row["context_sha256"])
        per_task_selected[task] += 1
        cursors[task] += 1

    if len(selected) < total_size:
        raise RuntimeError(
            f"only {len(selected)} fresh globally distinct LongBench contexts "
            f"available across eligible tasks {eligible_tasks}; need {total_size}"
        )

    diagnostics = {
        "pool_sizes_after_exclusion": pool_sizes,
        "eligible_tasks": eligible_tasks,
        "excluded_tasks_insufficient_fresh_support": excluded_tasks,
        "min_rows_to_represent_task": min_rows_to_represent_task,
        "selected_task_counts": per_task_selected,
        "selection_rule": (
            "keep the original 6144-9216 token window; exclude all legacy "
            "prompt/source/context overlaps; drop only tasks with fewer than "
            "two fresh in-window distinct contexts; fill the 8-prompt family "
            "approximately evenly across remaining tasks, breaking ties by "
            "distance to the 8192-token target and source index; no model "
            "outputs or answer quality are used"
        ),
    }
    return selected, diagnostics


def load_longbench_group(tokenizer, tasks, family, revision, target, low, high, exclusions):
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

    return _select_balanced_from_pools(
        pools,
        list(tasks),
        target=target,
        total_size=LONG_BENCH_FAMILY_SIZE,
    )

def load_ruler(tokenizer, revision, target, exclusions, local_parquet=None):
    if local_parquet is not None:
        ds = load_dataset("parquet", data_files={"test": str(local_parquet)}, split="test")
    else:
        ds = load_dataset("simonjegou/ruler", "8192", split="test", revision=revision)

    by_task = {task: [] for task in RULER_RETRIEVAL_TASKS}
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
    for task in RULER_RETRIEVAL_TASKS:
        if not by_task[task]:
            raise RuntimeError(f"RULER task {task} has no fresh row after exclusions")
        candidates = sorted(
            by_task[task],
            key=lambda row: (
                abs(row["context_tokens"] - target),
                row["source_index"],
            ),
        )
        selected.append(candidates[0])
    return selected


def main() -> None:
    args = parse_args()
    exclusions = load_exclusions(args.exclude_raw_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ruler = load_ruler(
        tokenizer,
        args.ruler_revision,
        args.target_context_tokens,
        exclusions,
        args.ruler_local_parquet,
    )
    single, single_diag = load_longbench_group(
        tokenizer,
        SINGLE_DOC_TASKS,
        "longbench_single",
        args.longbench_revision,
        args.target_context_tokens,
        args.min_context_tokens,
        args.max_context_tokens,
        exclusions,
    )
    multi, multi_diag = load_longbench_group(
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
    if len(records) != 24:
        raise AssertionError(f"expected 24 prompts, got {len(records)}")

    old_ids, old_hashes, old_sources = exclusions
    for row in records:
        if is_excluded(row, old_ids, old_hashes, old_sources):
            raise AssertionError(f"replication leakage detected: {row['id']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    family_counts: dict[str, int] = {}
    for row in records:
        family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1

    manifest = {
        "protocol": "SketchAlloc-KV Gate 0A fixed-action confirmatory replication",
        "selection_uses_model_outputs": False,
        "exclusion_source": str(args.exclude_raw_dir),
        "excluded_prompt_count": len(old_ids),
        "excluded_context_count": len(old_hashes),
        "num_prompts": len(records),
        "family_counts": family_counts,
        "target_context_tokens": args.target_context_tokens,
        "longbench_context_token_window": [args.min_context_tokens, args.max_context_tokens],
        "sampling_deviation_note": (
            "The original MiniGate reserved two prompts from every LongBench task. "
            "For confirmatory replication, legacy prompt/source/context overlap is forbidden. "
            "If a task has fewer than two fresh in-window distinct contexts after exclusion, "
            "that task is omitted rather than widening the token window or reusing old data."
        ),
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
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote replication data: {args.output}")
    print(f"Wrote manifest:         {args.manifest}")
    print("Family counts:", family_counts)
    for row in records:
        print(row["id"], f"context={row['context_tokens']}", f"query={row['query_tokens']}")


if __name__ == "__main__":
    main()
