from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml


def _numeric_metrics(obj: dict) -> list[tuple[str, float]]:
    out = []
    for key, value in obj.items():
        if isinstance(value, (int, float)):
            out.append((str(key), float(value)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Collect LU-KV generalization stress-test results.")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()

    rows = []
    missing = []
    with args.manifest.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    for rec in records:
        run_root = Path(rec["run_root"])
        metric_files = sorted(run_root.rglob("metrics.json"))
        if not metric_files:
            missing.append(rec["run_id"])
            continue

        # One run root should normally contain one evaluation directory.
        metrics_path = metric_files[-1]
        config_path = metrics_path.with_name("config.yaml")
        config = {}
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        for metric_name, score in _numeric_metrics(metrics):
            rows.append(
                {
                    "run_id": rec["run_id"],
                    "profile_name": rec["profile_name"],
                    "curve_path": rec["curve_path"],
                    "dimension": rec["dimension"],
                    "dataset": rec["dataset"],
                    "data_dir": rec["data_dir"],
                    "task_family": rec["task_family"],
                    "context_length": rec.get("context_length"),
                    "compression_ratio": rec["compression_ratio"],
                    "metric": metric_name,
                    "score": score,
                    "metrics_path": str(metrics_path),
                    "seed": config.get("seed"),
                }
            )

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    group_summary = []
    if not df.empty:
        grouped = df.groupby(
            ["profile_name", "dimension", "task_family", "compression_ratio"],
            dropna=False,
        )["score"]
        for key, series in grouped:
            group_summary.append(
                {
                    "profile_name": key[0],
                    "dimension": key[1],
                    "task_family": key[2],
                    "compression_ratio": float(key[3]),
                    "n": int(series.count()),
                    "mean_score": float(series.mean()),
                    "std_score": float(series.std(ddof=0)),
                }
            )

    payload = {
        "num_manifest_runs": len(records),
        "num_completed_runs": len(set(df["run_id"])) if not df.empty else 0,
        "missing_run_ids": missing,
        "group_summary": group_summary,
        "note": (
            "Raw score differences alone do not establish profile generalization failure. "
            "For confirmatory claims, compare a source profile with an independently "
            "target-calibrated LU profile on the same target test set."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(df)} metric rows to {args.output_csv}")
    print(f"Wrote summary to {args.output_json}")
    if missing:
        print(f"Missing runs: {len(missing)}")


if __name__ == "__main__":
    main()
