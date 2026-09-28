from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a leakage-free LU-KV profiling JSONL from SQuALITY v1.3 train."
    )
    p.add_argument("--model", required=True, help="Local/HF tokenizer path used to measure token length.")
    p.add_argument(
        "--dataset",
        default="pszemraj/SQuALITY-v1.3",
        help="Public SQuALITY v1.3 mirror on Hugging Face.",
    )
    p.add_argument(
        "--revision",
        default="336f1743b13cb5ad3524bf45e72b8aef27ad4d8e",
        help="Pinned Hugging Face dataset revision for reproducibility.",
    )
    p.add_argument("--split", default="train")
    p.add_argument("--num-stories", type=int, default=6)
    p.add_argument("--questions-per-story", type=int, default=5)
    p.add_argument("--min-tokens", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=7168)
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate0/profile_data/squality_train_6x5.jsonl"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/gate0/profile_data/squality_train_6x5_manifest.json"),
    )
    return p.parse_args()


def metadata_value(metadata, *keys):
    if not isinstance(metadata, dict):
        return None
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def main():
    args = parse_args()

    if args.num_stories <= 0:
        raise ValueError("--num-stories must be positive")
    if args.questions_per_story <= 0:
        raise ValueError("--questions-per-story must be positive")
    if args.min_tokens <= 0 or args.max_tokens < args.min_tokens:
        raise ValueError("invalid token-length interval")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Do not let datasets auto-discover every JSON/JSONL artifact in the Hub
    # repository. Some mirrors contain auxiliary files with a different schema
    # (e.g. instruction/input/output), which can trigger DatasetGenerationCastError.
    # Pin and load exactly the canonical SQuALITY split file instead.
    split_filename = "dev.jsonl" if args.split == "validation" else f"{args.split}.jsonl"
    data_url = (
        f"https://huggingface.co/datasets/{args.dataset}/resolve/"
        f"{args.revision}/{split_filename}"
    )
    ds = load_dataset(
        "json",
        data_files={args.split: data_url},
        split=args.split,
    )

    eligible = []
    for row_idx, row in enumerate(ds):
        document = row.get("document", "")
        questions_raw = row.get("questions", []) or []

        questions = []
        for q in questions_raw:
            if isinstance(q, dict):
                text = q.get("question_text") or q.get("question") or q.get("text")
            else:
                text = str(q)
            if text:
                text = text.strip()
            if text and text not in questions:
                questions.append(text)

        if len(questions) < args.questions_per_story:
            continue

        token_count = len(
            tokenizer.encode(document, add_special_tokens=True)
        )
        if not (args.min_tokens <= token_count <= args.max_tokens):
            continue

        metadata = row.get("metadata", {}) or {}
        eligible.append(
            {
                "row_idx": row_idx,
                "document": document,
                "questions": questions[: args.questions_per_story],
                "token_count": token_count,
                "passage_id": metadata_value(metadata, "passage_id", "gutenberg_id"),
                "uid": metadata_value(metadata, "uid", "id"),
            }
        )

    if len(eligible) < args.num_stories:
        lengths = sorted(item["token_count"] for item in eligible)
        raise RuntimeError(
            f"Only {len(eligible)} SQuALITY {args.split} stories satisfy "
            f"{args.min_tokens} <= tokens <= {args.max_tokens}; "
            f"need {args.num_stories}. Eligible lengths: {lengths}"
        )

    rng = random.Random(args.seed)
    selected = rng.sample(eligible, args.num_stories)
    selected.sort(key=lambda x: (x["token_count"], x["row_idx"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in selected:
            record = {
                "context": item["document"],
                "questions": item["questions"],
                "task": "squality_profile",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "source_dataset": args.dataset,
        "source_split": args.split,
        "source_revision": args.revision,
        "source_file": data_url,
        "model_tokenizer": args.model,
        "seed": args.seed,
        "selection": {
            "num_stories": args.num_stories,
            "questions_per_story": args.questions_per_story,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "total_question_pairs": args.num_stories * args.questions_per_story,
        },
        "stories": [
            {
                "source_row": item["row_idx"],
                "passage_id": item["passage_id"],
                "uid": item["uid"],
                "token_count": item["token_count"],
                "questions": item["questions"],
            }
            for item in selected
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote profile JSONL: {args.output}")
    print(f"Wrote manifest: {args.manifest}")
    print(
        f"Selected {len(selected)} stories x {args.questions_per_story} questions "
        f"= {len(selected) * args.questions_per_story} profiling pairs."
    )
    print("Token lengths:", [item["token_count"] for item in selected])


if __name__ == "__main__":
    main()
