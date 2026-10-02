"""Evaluate real LongBench code-completion quality on the held-out cross-scene contexts.

This is the first downstream check after the proxy diagnostic.  It reuses the
exact held-out contexts selected by evaluation.cross_scene.prepare, regenerates
answers with several 80%-compression LU/SnapKV budget profiles, and scores the
predictions with LongBench's official code_sim_score.

The script intentionally evaluates only the held-out contexts from the manifest;
calibration contexts are never scored here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import pipeline

import kvpress  # noqa: F401 - registers the kv-press pipeline
from evaluation.benchmarks.longbench.calculate_metrics import code_sim_score
from kvpress import LUPress, SnapKVPress


DEFAULT_PROFILES = (
    "official_lu",
    "pooled",
    "task_lcc",
    "task_repobench-p",
)


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    if not records:
        raise ValueError("manifest contains no records")
    return data


def load_samples(path: Path) -> dict[str, dict]:
    samples = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        context_id = f"context_{item['sample_id']}"
        samples[context_id] = item
    if not samples:
        raise ValueError("profile_samples.jsonl contains no samples")
    return samples


def match_ground_truths(
    manifest: dict,
    samples: dict[str, dict],
    *,
    dataset_name: str,
    tasks: list[str],
) -> list[dict]:
    heldout = [
        item
        for item in manifest["records"]
        if item["split"] == "heldout" and item["task"] in tasks
    ]
    needed = defaultdict(dict)
    for item in heldout:
        needed[item["task"]][item["context_sha256"]] = item

    matched: dict[str, dict] = {}
    for task in tasks:
        if task not in needed:
            continue
        ds = load_dataset(
            dataset_name,
            task,
            split="test",
            trust_remote_code=True,
        )
        for row in ds:
            raw_context = row.get("context")
            if not isinstance(raw_context, str):
                continue
            digest = hashlib.sha256(raw_context.encode("utf-8")).hexdigest()
            record = needed[task].get(digest)
            if record is None:
                continue
            context_id = record["context_id"]
            sample = samples.get(context_id)
            if sample is None:
                raise KeyError(f"{context_id}: missing from profile_samples.jsonl")
            answers = row.get("answers")
            if not isinstance(answers, (list, tuple)) or not answers:
                raise ValueError(f"{context_id}: missing LongBench answers")
            matched[context_id] = {
                "context_id": context_id,
                "task": task,
                "context": sample["context"],
                "question": sample["questions"][0],
                "answer_prefix": sample["answer_prefix"],
                "answers": list(answers),
            }

    expected = {item["context_id"] for item in heldout}
    missing = sorted(expected - set(matched))
    if missing:
        raise ValueError(f"Could not recover LongBench ground truths for: {missing}")

    return [matched[item["context_id"]] for item in heldout]


def load_profile(path: Path, expected_shape: tuple[int, int] | None) -> np.ndarray:
    profile = np.load(path, allow_pickle=False)
    if profile.ndim != 2:
        raise ValueError(f"{path}: expected [layer, KV-head] profile, got {profile.shape}")
    if expected_shape is not None and profile.shape != expected_shape:
        raise ValueError(
            f"{path}: profile shape {profile.shape} != expected {expected_shape}"
        )
    if not np.isfinite(profile).all():
        raise ValueError(f"{path}: profile contains non-finite values")
    return np.asarray(profile, dtype=np.float64)


def make_press(profile: np.ndarray, compression: float, sink: int, window: int) -> LUPress:
    press = LUPress(
        press=SnapKVPress(compression_ratio=compression),
        sink=sink,
        window=window,
    )
    # LUPress indexes a 99-step curve by compression ratio.  Only the requested
    # row is used in this experiment; repeating the fixed profile over all rows
    # avoids manufacturing unrelated curve points.
    press._budget_curves = np.repeat(profile[None, ...], 99, axis=0)
    press.compression_ratio = compression
    return press


def score_prediction(prediction: str, answers: list[str]) -> float:
    return max(code_sim_score(prediction, answer) for answer in answers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="THUDM/LongBench")
    parser.add_argument("--tasks", default="lcc,repobench-p")
    parser.add_argument(
        "--profiles",
        default=",".join(DEFAULT_PROFILES),
        help="Comma-separated profile names; expects <name>_ratio_<pct>.npy",
    )
    parser.add_argument("--compression", type=float, default=0.80)
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    pct = int(round(args.compression * 100))
    if abs(args.compression * 100 - pct) > 1e-8:
        raise ValueError("--compression must be an integer percentage such as 0.80")

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]
    if not tasks or not profiles:
        raise ValueError("--tasks and --profiles must be nonempty")

    manifest = load_manifest(args.manifest)
    samples = load_samples(args.samples)
    heldout = match_ground_truths(
        manifest,
        samples,
        dataset_name=args.dataset_name,
        tasks=tasks,
    )

    profile_arrays = {}
    shape = None
    for name in profiles:
        path = args.profiles_dir / f"{name}_ratio_{pct}.npy"
        profile = load_profile(path, shape)
        shape = profile.shape if shape is None else shape
        profile_arrays[name] = profile

    dtype = getattr(torch, args.torch_dtype)
    pipe = pipeline(
        "kv-press-text-generation",
        model=args.model_path,
        model_kwargs={
            "torch_dtype": dtype,
            "attn_implementation": "eager",
        },
        device_map=args.device_map,
        trust_remote_code=True,
    )
    pipe.model.eval()

    rows = []
    task_scores = defaultdict(lambda: defaultdict(list))

    for profile_name in profiles:
        print(f"\n=== profile: {profile_name} ===", flush=True)
        press = make_press(
            profile_arrays[profile_name],
            compression=args.compression,
            sink=args.sink,
            window=args.window,
        )
        for index, item in enumerate(heldout, 1):
            output = pipe(
                item["context"],
                question=item["question"],
                answer_prefix=item["answer_prefix"],
                press=press,
                max_new_tokens=args.max_new_tokens,
                dataset_name="longbench",
                data_dir_name=item["task"],
            )
            prediction = output["answer"]
            score = score_prediction(prediction, item["answers"])
            task_scores[item["task"]][profile_name].append(score)
            rows.append({
                "task": item["task"],
                "context_id": item["context_id"],
                "profile": profile_name,
                "code_sim_score": float(score),
                "prediction": prediction,
                "answers_json": json.dumps(item["answers"], ensure_ascii=False),
            })
            print(
                f"{index:02d}/{len(heldout)} {item['context_id']} "
                f"score={score * 100:.2f}",
                flush=True,
            )
            torch.cuda.empty_cache()

    summary = {
        "warning": (
            "Held-out LongBench subset only; this is downstream code_sim_score, "
            "not an independent final benchmark."
        ),
        "settings": {
            "manifest": str(args.manifest),
            "samples": str(args.samples),
            "profiles_dir": str(args.profiles_dir),
            "model_path": args.model_path,
            "compression": args.compression,
            "sink": args.sink,
            "window": args.window,
            "max_new_tokens": args.max_new_tokens,
        },
        "scores": {},
    }
    for task in tasks:
        summary["scores"][task] = {}
        for profile_name in profiles:
            values = np.asarray(task_scores[task][profile_name], dtype=np.float64)
            if values.size:
                summary["scores"][task][profile_name] = {
                    "mean_code_sim_score": float(values.mean() * 100.0),
                    "per_context_code_sim_score": [float(x * 100.0) for x in values],
                    "contexts": int(values.size),
                }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "task",
                "context_id",
                "profile",
                "code_sim_score",
                "prediction",
                "answers_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "quality_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== REAL CODE QUALITY (LongBench code_sim_score, higher is better) ===")
    for task in tasks:
        if task not in summary["scores"]:
            continue
        print(f"[{task}]")
        for name, result in summary["scores"][task].items():
            print(f"  {name:22s} {result['mean_code_sim_score']:.2f}")
    print(f"Saved: {args.output_dir / 'quality_results.json'}")
    print(f"Saved: {args.output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
