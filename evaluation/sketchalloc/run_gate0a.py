"""One-command runner for SketchAlloc-KV Gate 0A.

This wrapper imports legacy CausalDuel answer-onset JSON files, reconstructs
stable action IDs, audits rectangular prompt-by-action coverage, and runs the
leakage-aware adaptive-necessity analysis. It does not require a GPU.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from evaluation.sketchalloc.analyze_gate0a import evaluate_gate0a
from evaluation.sketchalloc.normalize_legacy_results import (
    audit_coverage,
    load_payload_files,
    normalize_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sketchalloc/gate0a"),
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads, source_files = load_payload_files(args.raw_dir)

    rows: list[dict] = []
    for payload, source_file in zip(payloads, source_files):
        rows.extend(normalize_payload(payload, source_file))

    frame = pd.DataFrame(rows).sort_values(["prompt_id", "action_id"])
    audit = audit_coverage(frame, source_files)
    summary = evaluate_gate0a(
        frame,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions_path = args.output_dir / "legacy24_actions.csv"
    coverage_path = args.output_dir / "legacy24_coverage.json"
    summary_path = args.output_dir / "legacy24_gate0a.json"

    frame.to_csv(actions_path, index=False)
    coverage_path.write_text(
        json.dumps(asdict(audit), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Actions:  {actions_path}")
    print(f"Coverage: {coverage_path}")
    print(f"Gate 0A:  {summary_path}")
    print(json.dumps(asdict(audit), ensure_ascii=False, indent=2))
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
