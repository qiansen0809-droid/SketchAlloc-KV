"""Equal-sample comparison for task-specific vs mixed LU-KV calibration profiles.

The current pooled profile averages all ten calibration contexts (5 LCC + 5
RepoBench-P), while each task-specific profile uses only five.  This script
removes that sample-count advantage by enumerating every five-context mixed
subset from the same ten calibration contexts and evaluating each profile on
the untouched held-out contexts.

No model inference is run; only the already-collected oracle/scorer arrays are
used.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from evaluation.cross_scene.analyze import prepare_pairs
from evaluation.active_calibration.gate0 import evaluate_profile


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="snapkv")
    parser.add_argument("--compression", type=float, default=0.80)
    parser.add_argument("--threshold", type=int, default=99)
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--subset-size", type=int, default=5)
    parser.add_argument(
        "--tasks",
        default="lcc,repobench-p",
        help="Exactly two task names used for the mixed calibration pool.",
    )
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    if len(tasks) != 2 or len(set(tasks)) != 2:
        raise ValueError("--tasks must contain exactly two distinct tasks")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [
        item for item in manifest["records"]
        if item["task"] in tasks
    ]
    if not records:
        raise ValueError("No requested tasks found in manifest")

    grouped, _ = prepare_pairs(
        records,
        args.raw_root,
        method=args.method,
        sink=args.sink,
        window=args.window,
        compression=args.compression,
        threshold=args.threshold,
    )

    calibration = []
    for task in tasks:
        for pair in grouped[task]["calibration"]:
            calibration.append((task, pair))

    if len(calibration) < args.subset_size:
        raise ValueError("Not enough calibration contexts")
    for task in tasks:
        if len(grouped[task]["calibration"]) != args.subset_size:
            raise ValueError(
                f"{task}: expected exactly {args.subset_size} calibration contexts "
                "for the equal-sample task-specific baseline"
            )

    task_profiles = {
        task: np.mean(
            [pair.prune_ratio for pair in grouped[task]["calibration"]],
            axis=0,
        )
        for task in tasks
    }
    task_losses = {
        test_task: evaluate_profile(
            task_profiles[test_task],
            grouped[test_task]["heldout"],
            sink=args.sink,
            window=args.window,
        )["mean_proxy_loss"]
        for test_task in tasks
    }

    rows = []
    for chosen_indices in itertools.combinations(
        range(len(calibration)),
        args.subset_size,
    ):
        chosen = [calibration[i] for i in chosen_indices]
        counts = {
            task: sum(chosen_task == task for chosen_task, _ in chosen)
            for task in tasks
        }

        # "Mixed" means the subset actually contains both scenes.
        if any(counts[task] == 0 for task in tasks):
            continue

        profile = np.mean(
            [pair.prune_ratio for _, pair in chosen],
            axis=0,
        )
        row = {
            "subset_ids": "|".join(pair.pair_id for _, pair in chosen),
            f"n_{tasks[0]}": counts[tasks[0]],
            f"n_{tasks[1]}": counts[tasks[1]],
        }
        for test_task in tasks:
            loss = evaluate_profile(
                profile,
                grouped[test_task]["heldout"],
                sink=args.sink,
                window=args.window,
            )["mean_proxy_loss"]
            row[f"loss_{test_task}"] = float(loss)
            row[f"delta_vs_task_specific_{test_task}"] = float(
                loss - task_losses[test_task]
            )
        rows.append(row)

    if not rows:
        raise ValueError("No mixed subsets were produced")

    summary = {
        "warning": (
            "Offline proxy comparison only. Every mixed profile and every "
            "task-specific profile uses exactly the same number of calibration contexts."
        ),
        "settings": {
            "manifest": str(args.manifest),
            "raw_root": str(args.raw_root),
            "tasks": tasks,
            "subset_size": args.subset_size,
            "compression": args.compression,
            "mixed_subsets": len(rows),
        },
        "task_specific_baseline": {
            task: {
                "calibration_contexts": args.subset_size,
                "mean_proxy_loss": float(task_losses[task]),
            }
            for task in tasks
        },
        "mixed_equal_sample": {},
    }

    for test_task in tasks:
        losses = [row[f"loss_{test_task}"] for row in rows]
        deltas = [row[f"delta_vs_task_specific_{test_task}"] for row in rows]
        balanced_rows = [
            row
            for row in rows
            if abs(row[f"n_{tasks[0]}"] - row[f"n_{tasks[1]}"]) == 1
        ]
        balanced_deltas = [
            row[f"delta_vs_task_specific_{test_task}"] for row in balanced_rows
        ]
        summary["mixed_equal_sample"][test_task] = {
            "loss_distribution": summarize(losses),
            "delta_vs_task_specific_distribution": summarize(deltas),
            "fraction_mixed_beats_task_specific": float(
                np.mean(np.asarray(deltas) < 0)
            ),
            "balanced_2_3_or_3_2_subsets": len(balanced_rows),
            "balanced_fraction_beats_task_specific": float(
                np.mean(np.asarray(balanced_deltas) < 0)
            ),
            "balanced_delta_distribution": summarize(balanced_deltas),
        }

    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "equal_sample_subsets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "equal_sample_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== EQUAL-SAMPLE MIXED-vs-TASK-SPECIFIC ===")
    for task in tasks:
        item = summary["mixed_equal_sample"][task]
        print(
            f"{task}: task-specific={task_losses[task]:.6f}, "
            f"mixed median={item['loss_distribution']['median']:.6f}, "
            f"mixed beats task-specific="
            f"{item['fraction_mixed_beats_task_specific'] * 100:.1f}%, "
            f"balanced beats="
            f"{item['balanced_fraction_beats_task_specific'] * 100:.1f}%"
        )
    print(f"Saved: {args.output_dir / 'equal_sample_results.json'}")
    print(f"Saved: {args.output_dir / 'equal_sample_subsets.csv'}")


if __name__ == "__main__":
    main()
