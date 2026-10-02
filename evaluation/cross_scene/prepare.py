"""Prepare a small, context-disjoint LongBench probe for LU-KV step 1.

This is a *diagnostic* subset, not a benchmark result. The source is the
official THUDM/LongBench test split; selected contexts are divided into
calibration and held-out subsets before any oracle curve is collected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASK_TEMPLATES = {
    "lcc": (
        "Please complete the code given below. \n{context}",
        "{input}",
        "Next line of code:\n",
    ),
    "repobench-p": (
        "Please complete the code given below. \n{context}",
        "{input}",
        "Next line of code:\n",
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. "
        "Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\nThe following is an abstract.\n\n",
        '{input}\n\nPlease enter the number of the paragraph that the abstract is from. '
        'The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\n',
        "The answer is: ",
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n"
        "News:\n{context}\n\n",
        "Now, write a one-page summary of all the news.\n\n",
        "Summary:",
    ),
}


def format_row(task: str, row: dict) -> tuple[str, str, str]:
    context_template, question_template, answer_prefix = TASK_TEMPLATES[task]
    raw_context = row.get("context")
    if not isinstance(raw_context, str) or not raw_context.strip():
        raise ValueError(f"{task}: row lacks a nonempty context")
    raw_input = row.get("input", "")
    if not isinstance(raw_input, str):
        raise ValueError(f"{task}: row input must be a string")
    return (
        context_template.format(context=raw_context),
        question_template.format(input=raw_input),
        answer_prefix,
    )


def pick_rows(
    task: str,
    rows,
    tokenizer,
    *,
    count: int,
    min_tokens: int,
    max_tokens: int,
    scan_limit: int,
    seed: int,
    seen_hashes: set[str],
) -> list[dict]:
    candidates = []
    local_hashes = set()
    for row in rows:
        if scan_limit and len(local_hashes) >= scan_limit:
            break
        try:
            context, question, answer_prefix = format_row(task, row)
        except ValueError:
            continue
        # Hash the underlying source, not its task-specific prompt wrapper:
        # one document must not appear on both sides under different templates.
        digest = hashlib.sha256(row["context"].encode("utf-8")).hexdigest()
        if digest in seen_hashes or digest in local_hashes:
            continue
        local_hashes.add(digest)
        token_count = len(tokenizer.encode(context, add_special_tokens=True))
        if not min_tokens <= token_count <= max_tokens:
            continue
        order_key = hashlib.sha256(f"{seed}:{task}:{digest}".encode()).hexdigest()
        candidates.append({
            "context": context,
            "question": question,
            "answer_prefix": answer_prefix,
            "context_sha256": digest,
            "context_tokens": token_count,
            "order_key": order_key,
        })
    candidates.sort(key=lambda item: item["order_key"])
    if len(candidates) < count:
        raise ValueError(
            f"{task}: only {len(candidates)} unique contexts in {min_tokens}-{max_tokens} "
            f"tokens after scanning {len(local_hashes)}; need {count}. "
            "Widen the token range or increase --scan-limit; do not silently truncate."
        )
    chosen = candidates[:count]
    seen_hashes.update(item["context_sha256"] for item in chosen)
    return chosen


def prepare(args, *, dataset_loader=None, tokenizer=None) -> dict:
    if args.out_dir.exists():
        raise FileExistsError(f"Output already exists: {args.out_dir}")
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if len(tasks) < 2 or len(tasks) != len(set(tasks)):
        raise ValueError("--tasks requires at least two distinct tasks")
    if any(task not in TASK_TEMPLATES for task in tasks):
        raise ValueError(f"Supported tasks: {', '.join(TASK_TEMPLATES)}")
    if not 1 <= args.calibration_per_task < args.per_task:
        raise ValueError("Need 1 <= --calibration-per-task < --per-task")
    if args.min_tokens <= 0 or args.max_tokens < args.min_tokens:
        raise ValueError("Invalid context token range")

    if dataset_loader is None:
        from datasets import load_dataset
        dataset_loader = lambda task: load_dataset("THUDM/LongBench", task, split="test")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    samples = []
    manifest = []
    seen_hashes: set[str] = set()
    for task in tasks:
        selected = pick_rows(
            task, dataset_loader(task), tokenizer,
            count=args.per_task,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            scan_limit=args.scan_limit,
            seed=args.seed,
            seen_hashes=seen_hashes,
        )
        for index, item in enumerate(selected):
            sample_id = f"{task}_{index:03d}"
            split = "calibration" if index < args.calibration_per_task else "heldout"
            samples.append({
                "sample_id": sample_id,
                "task": task,
                "context": item["context"],
                "questions": [item["question"]],
                "answer_prefix": item["answer_prefix"],
            })
            manifest.append({
                "context_id": f"context_{sample_id}",
                "task": task,
                "split": split,
                "context_tokens": item["context_tokens"],
                "context_sha256": item["context_sha256"],
            })

    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "profile_samples.jsonl").open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    metadata = {
        "warning": "Exploratory LongBench test-split subset; no independent final benchmark.",
        "source": "THUDM/LongBench",
        "model_path": str(args.model_path),
        "seed": args.seed,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "records": manifest,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        default="lcc,repobench-p,passage_retrieval_en,multi_news",
    )
    parser.add_argument("--per-task", type=int, default=6)
    parser.add_argument("--calibration-per-task", type=int, default=3)
    parser.add_argument("--min-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--scan-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare(args)
    for task in sorted({record["task"] for record in result["records"]}):
        subset = [record for record in result["records"] if record["task"] == task]
        print(
            f"{task}: {len(subset)} contexts, "
            f"{sum(x['split'] == 'calibration' for x in subset)} calibration, "
            f"{sum(x['split'] == 'heldout' for x in subset)} heldout"
        )
    print(f"Wrote {args.out_dir / 'profile_samples.jsonl'}")
    print(f"Wrote {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
