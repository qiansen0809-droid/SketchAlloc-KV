from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


RULER_RETRIEVAL_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
]

SINGLE_DOC_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
]

MULTI_DOC_TASKS = [
    "hotpotqa",
    "2wikimqa",
    "musique",
]

LONG_BENCH_FAMILY_SIZE = 8
LONG_BENCH_MIN_PER_TASK = 2


def parse_args():
    p = argparse.ArgumentParser(description="Prepare the fixed 24-prompt Gate-0 MiniGate set.")
    p.add_argument("--model", required=True, help="Tokenizer path used for exact token counts.")
    p.add_argument("--output", type=Path, default=Path("results/gate0/minigate/minigate_24.jsonl"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/gate0/minigate/minigate_24_manifest.json"),
    )
    p.add_argument("--target-context-tokens", type=int, default=8192)
    p.add_argument("--min-context-tokens", type=int, default=6144)
    p.add_argument("--max-context-tokens", type=int, default=9216)
    p.add_argument("--ruler-revision", default="24adceac8a0e6532936e8d721cd9e9084d2e4686")
    p.add_argument(
        "--ruler-local-parquet",
        type=Path,
        default=None,
        help="Optional local copy of RULER 8192 test parquet; avoids flaky HF downloads.",
    )
    p.add_argument(
        "--longbench-revision",
        default="0ce23c4aa955accf17527097eb12a8f00e2743e6",
    )
    return p.parse_args()


def as_answers(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def token_len(tokenizer, text, add_special_tokens):
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def make_record(
    *,
    family,
    task,
    source_dataset,
    source_config,
    source_revision,
    source_index,
    row,
    tokenizer,
):
    context = str(row["context"])
    question = str(row.get("question", ""))
    answer_prefix = str(row.get("answer_prefix", ""))
    answers = as_answers(row.get("answer", row.get("answers")))

    if not answers:
        raise ValueError(f"{family}/{task}/{source_index} has no gold answer")

    context_tokens = token_len(tokenizer, context, add_special_tokens=True)
    query_tokens = token_len(tokenizer, question + answer_prefix, add_special_tokens=False)

    context_sha256 = hashlib.sha256(context.encode("utf-8")).hexdigest()

    return {
        "id": f"{family}__{task}__{source_index}",
        "context_sha256": context_sha256,
        "family": family,
        "task": task,
        "source_dataset": source_dataset,
        "source_config": source_config,
        "source_revision": source_revision,
        "source_index": int(source_index),
        "context": context,
        "question": question,
        "answer_prefix": answer_prefix,
        "answers": answers,
        "max_new_tokens": int(row.get("max_new_tokens", 64)),
        "context_tokens": int(context_tokens),
        "query_tokens": int(query_tokens),
        "prompt_tokens": int(context_tokens + query_tokens),
    }


def distinct_by_length(records, target, low, high):
    eligible = [
        r for r in records
        if low <= r["context_tokens"] <= high
    ]
    eligible.sort(
        key=lambda r: (
            abs(r["context_tokens"] - target),
            r["source_index"],
        )
    )

    selected = []
    seen_contexts = set()
    for row in eligible:
        if row["context_sha256"] in seen_contexts:
            continue
        selected.append(row)
        seen_contexts.add(row["context_sha256"])
    return selected


def load_longbench_group(
    tokenizer,
    tasks,
    family,
    revision,
    target,
    low,
    high,
    total_size=LONG_BENCH_FAMILY_SIZE,
    min_per_task=LONG_BENCH_MIN_PER_TASK,
):
    """
    Build one 8-prompt LongBench family without duplicate contexts.

    The selection is intentionally balanced but does not force brittle fixed
    quotas such as 3/3/2. We first reserve two distinct natural contexts from
    each task, then fill the remaining two slots from the globally closest
    unused contexts to the 8K target. Selection uses only task identity and
    tokenizer length, never model outputs or answer quality.
    """
    pools = {}
    for task in tasks:
        ds = load_dataset(
            "Xnhyacinth/LongBench",
            task,
            split="test",
            revision=revision,
        )

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
        pools[task] = distinct_by_length(records, target, low, high)

        if len(pools[task]) < min_per_task:
            raise RuntimeError(
                f"{family}/{task}: only {len(pools[task])} distinct contexts fall "
                f"in [{low}, {high}] context tokens; need at least {min_per_task}."
            )

    selected = []
    seen_contexts = set()

    # Balanced base: two distinct contexts from every task.
    for task in tasks:
        taken = 0
        for row in pools[task]:
            if row["context_sha256"] in seen_contexts:
                continue
            selected.append(row)
            seen_contexts.add(row["context_sha256"])
            taken += 1
            if taken == min_per_task:
                break
        if taken < min_per_task:
            raise RuntimeError(
                f"{family}/{task}: could not reserve {min_per_task} globally "
                "distinct contexts after cross-task deduplication."
            )

    # Fill the remaining family slots by closeness to the target length.
    remaining = []
    for task in tasks:
        for row in pools[task]:
            if row["context_sha256"] not in seen_contexts:
                remaining.append(row)

    remaining.sort(
        key=lambda r: (
            abs(r["context_tokens"] - target),
            r["task"],
            r["source_index"],
        )
    )

    for row in remaining:
        if len(selected) >= total_size:
            break
        if row["context_sha256"] in seen_contexts:
            continue
        selected.append(row)
        seen_contexts.add(row["context_sha256"])

    if len(selected) < total_size:
        raise RuntimeError(
            f"{family}: only {len(selected)} globally distinct contexts available "
            f"in [{low}, {high}] tokens; need {total_size}."
        )

    return selected[:total_size]


def load_ruler(tokenizer, revision, target, local_parquet=None):
    if local_parquet is not None:
        local_parquet = Path(local_parquet)
        if not local_parquet.exists():
            raise FileNotFoundError(f"RULER parquet not found: {local_parquet}")
        ds = load_dataset(
            "parquet",
            data_files={"test": str(local_parquet)},
            split="test",
        )
    else:
        ds = load_dataset(
            "simonjegou/ruler",
            "8192",
            split="test",
            revision=revision,
        )

    by_task = {task: [] for task in RULER_RETRIEVAL_TASKS}
    for i, row in enumerate(ds):
        task = str(row["task"])
        if task not in by_task:
            continue
        by_task[task].append(
            make_record(
                family="ruler_retrieval",
                task=task,
                source_dataset="simonjegou/ruler",
                source_config="8192",
                source_revision=revision,
                source_index=i,
                row=row,
                tokenizer=tokenizer,
            )
        )

    selected = []
    for task in RULER_RETRIEVAL_TASKS:
        if not by_task[task]:
            raise RuntimeError(f"RULER config 8192 has no rows for task={task}")

        # One prompt per official retrieval subtask. Length is the only ranking
        # criterion; no model outputs or gold-answer quality are consulted.
        candidates = sorted(
            by_task[task],
            key=lambda r: (
                abs(r["context_tokens"] - target),
                r["source_index"],
            ),
        )
        selected.append(candidates[0])

    return selected


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    ruler = load_ruler(
        tokenizer,
        revision=args.ruler_revision,
        target=args.target_context_tokens,
        local_parquet=args.ruler_local_parquet,
    )
    single = load_longbench_group(
        tokenizer,
        tasks=SINGLE_DOC_TASKS,
        family="longbench_single",
        revision=args.longbench_revision,
        target=args.target_context_tokens,
        low=args.min_context_tokens,
        high=args.max_context_tokens,
    )
    multi = load_longbench_group(
        tokenizer,
        tasks=MULTI_DOC_TASKS,
        family="longbench_multi",
        revision=args.longbench_revision,
        target=args.target_context_tokens,
        low=args.min_context_tokens,
        high=args.max_context_tokens,
    )

    records = ruler + single + multi
    counts = {}
    for row in records:
        counts[row["family"]] = counts.get(row["family"], 0) + 1

    expected = {
        "ruler_retrieval": 8,
        "longbench_single": 8,
        "longbench_multi": 8,
    }
    if counts != expected:
        raise AssertionError(f"MiniGate family counts mismatch: {counts} != {expected}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "protocol": "CausalDuel-KV Gate-0 MiniGate",
        "num_prompts": len(records),
        "family_counts": counts,
        "target_context_tokens": args.target_context_tokens,
        "longbench_context_token_window": [
            args.min_context_tokens,
            args.max_context_tokens,
        ],
        "ruler": {
            "dataset": "simonjegou/ruler",
            "config": "8192",
            "revision": args.ruler_revision,
            "tasks": RULER_RETRIEVAL_TASKS,
            "selection": "one row per retrieval subtask, closest context length to target",
            "local_parquet": str(args.ruler_local_parquet) if args.ruler_local_parquet else None,
        },
        "longbench": {
            "dataset": "Xnhyacinth/LongBench",
            "revision": args.longbench_revision,
            "single_doc_tasks": SINGLE_DOC_TASKS,
            "multi_doc_tasks": MULTI_DOC_TASKS,
            "family_size": LONG_BENCH_FAMILY_SIZE,
            "min_per_task": LONG_BENCH_MIN_PER_TASK,
            "selection": (
                "reserve two distinct natural contexts per task, then fill the "
                "remaining two family slots by closeness to the target length; "
                "fixed token window; no truncation; no model-output-based selection"
            ),
        },
        "samples": [
            {
                key: row[key]
                for key in (
                    "id",
                    "family",
                    "task",
                    "source_dataset",
                    "source_config",
                    "source_revision",
                    "source_index",
                    "context_sha256",
                    "context_tokens",
                    "query_tokens",
                    "prompt_tokens",
                    "answers",
                    "max_new_tokens",
                )
            }
            for row in records
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote MiniGate data: {args.output}")
    print(f"Wrote manifest:      {args.manifest}")
    print("Family counts:", counts)
    for row in records:
        print(
            row["id"],
            f"context={row['context_tokens']}",
            f"query={row['query_tokens']}",
            f"prompt={row['prompt_tokens']}",
        )


if __name__ == "__main__":
    main()
