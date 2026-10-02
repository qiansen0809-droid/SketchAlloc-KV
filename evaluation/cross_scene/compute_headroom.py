"""Compute post-hoc budget-vs-token-scorer headroom for cross-scene heldout contexts.

This diagnostic separates two possible bottlenecks:
1) fixed_scorer_budget_headroom:
   keep SnapKV's within-head token ranking fixed, but replace the official
   cross-head budget with the heldout context's own LU-solved budget.
2) token_scorer_headroom_at_official_budget:
   keep the official per-head budget fixed, but select tokens post-hoc by the
   oracle utility.  This is a non-deployable fixed-budget upper bound.

If token-scorer headroom is much larger than budget headroom, head-budget
allocation is probably not the dominant bottleneck.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.cross_scene.analyze import (
    load_manifest,
    oracle_headroom,
    prepare_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="snapkv")
    parser.add_argument("--compression", type=float, default=0.80)
    parser.add_argument("--threshold", type=int, default=99)
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--tasks", default="lcc,repobench-p")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    manifest_records = [
        item
        for item in load_manifest(args.manifest)
        if item["task"] in tasks
    ]
    if not manifest_records:
        raise ValueError("No requested tasks found in manifest")

    grouped, shape = prepare_pairs(
        manifest_records,
        args.raw_root,
        method=args.method,
        sink=args.sink,
        window=args.window,
        compression=args.compression,
        threshold=args.threshold,
    )

    official_curve = np.load(args.official_profile, allow_pickle=False)
    index = int(round(args.compression * 100)) - 1
    if (
        official_curve.ndim != 3
        or official_curve.shape[0] <= index
        or official_curve.shape[1:] != shape
    ):
        raise ValueError(
            f"Official curve shape {official_curve.shape} does not match {shape}"
        )
    official = np.asarray(official_curve[index], dtype=np.float64)

    result = oracle_headroom(
        grouped,
        official,
        args.raw_root,
        sink=args.sink,
        window=args.window,
    )

    output = {
        "warning": (
            "Post-hoc offline diagnostic. In-sample LU budget uses each heldout "
            "context's oracle and is not deployable. Oracle token selection is an "
            "upper bound at the official head budget."
        ),
        "settings": {
            "manifest": str(args.manifest),
            "raw_root": str(args.raw_root),
            "official_profile": str(args.official_profile),
            "compression": args.compression,
            "sink": args.sink,
            "window": args.window,
        },
        "tasks": result,
    }

    args.output_dir.mkdir(parents=True)
    out = args.output_dir / "headroom_results.json"
    out.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== BUDGET-vs-TOKEN-SCORER HEADROOM ===")
    for task in tasks:
        item = result[task]
        budget = item["mean_fixed_scorer_budget_headroom"]
        token = item["mean_token_scorer_headroom_at_official_budget"]
        ratio = None if abs(budget) < 1e-12 else token / budget
        print(
            f"{task}: budget_headroom={budget:+.6f}, "
            f"token_scorer_headroom={token:+.6f}, "
            f"token/budget={'N/A' if ratio is None else f'{ratio:.2f}x'}"
        )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
