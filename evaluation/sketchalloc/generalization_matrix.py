from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

TASK_FAMILIES = {
    "single_doc_qa": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multi_doc_qa": ["hotpotqa", "2wikimqa", "musique"],
    "summarization": ["gov_report", "qmsum", "multi_news"],
    "few_shot": ["trec", "triviaqa", "samsum"],
    "synthetic": ["passage_count", "passage_retrieval_en"],
    "code": ["lcc", "repobench-p"],
}

SCREENING_TASKS = [
    "qasper",
    "hotpotqa",
    "gov_report",
    "trec",
    "passage_retrieval_en",
    "repobench-p",
]

SCREENING_LENGTHS = [8192, 16384, 32768]
FULL_LENGTHS = [8192, 16384, 32768, 65536]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    profile_name: str
    curve_path: str
    dimension: str
    dataset: str
    data_dir: str
    compression_ratio: float
    context_length: int | None
    task_family: str
    run_root: str


def family_for_task(task: str) -> str:
    for family, tasks in TASK_FAMILIES.items():
        if task in tasks:
            return family
    return "unknown"


def parse_profile(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--profile must be NAME=/path/to/curve.npy")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--profile must be NAME=/path/to/curve.npy")
    return name, path


def _safe_ratio(ratio: float) -> str:
    return str(ratio).replace(".", "p")


def build_specs(
    profiles: Iterable[tuple[str, str]],
    output_root: Path,
    dimension: str,
    mode: str,
) -> list[RunSpec]:
    dims = {"domain", "length", "compression"} if dimension == "all" else {dimension}
    tasks = SCREENING_TASKS if mode == "screening" else [t for ts in TASK_FAMILIES.values() for t in ts]
    lengths = SCREENING_LENGTHS if mode == "screening" else FULL_LENGTHS

    specs: list[RunSpec] = []
    for profile_name, curve_path in profiles:
        if "domain" in dims:
            for task in tasks:
                ratio = 0.8
                run_id = f"{profile_name}__domain__{task}__cr{_safe_ratio(ratio)}"
                specs.append(
                    RunSpec(
                        run_id, profile_name, curve_path, "domain", "longbench", task,
                        ratio, None, family_for_task(task), str(output_root / "raw" / run_id)
                    )
                )

        if "compression" in dims:
            for task in tasks:
                for ratio in (0.5, 0.8, 0.9):
                    run_id = f"{profile_name}__compression__{task}__cr{_safe_ratio(ratio)}"
                    specs.append(
                        RunSpec(
                            run_id, profile_name, curve_path, "compression", "longbench", task,
                            ratio, None, family_for_task(task), str(output_root / "raw" / run_id)
                        )
                    )

        if "length" in dims:
            for length in lengths:
                for ratio in (0.8, 0.9):
                    run_id = f"{profile_name}__length__{length}__cr{_safe_ratio(ratio)}"
                    specs.append(
                        RunSpec(
                            run_id, profile_name, curve_path, "length", "ruler", str(length),
                            ratio, length, "ruler", str(output_root / "raw" / run_id)
                        )
                    )
    return specs


def build_command(
    spec: RunSpec,
    model: str,
    device: str,
    fraction: float,
) -> list[str]:
    return [
        sys.executable,
        "evaluation/evaluate.py",
        "--dataset", spec.dataset,
        "--data_dir", spec.data_dir,
        "--model", model,
        "--device", device,
        "--press_name", "lu_snapkv",
        "--compression_ratio", str(spec.compression_ratio),
        "--output_dir", spec.run_root,
        "--budget_curve_path", spec.curve_path,
        "--fraction", str(fraction),
        "--seed", "20260928",
    ]


def build_env(longbench_path: str | None, ruler_path: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if longbench_path:
        env["SKETCHALLOC_LONGBENCH_PATH"] = longbench_path
    if ruler_path:
        env["SKETCHALLOC_RULER_PATH"] = ruler_path
    return env


def append_manifest(
    path: Path,
    spec: RunSpec,
    command: list[str],
    longbench_path: str | None,
    ruler_path: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(spec)
    payload["command"] = command
    payload["longbench_path"] = longbench_path
    payload["ruler_path"] = ruler_path
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Plan or execute LU-KV generalization stress tests.")
    p.add_argument("--model", required=True)
    p.add_argument("--profile", action="append", type=parse_profile, required=True,
                   help="Repeatable NAME=/path/to/curve.npy")
    p.add_argument("--longbench-path", help="Override the LongBench dataset root for this run.")
    p.add_argument("--ruler-path", help="Override the RULER dataset root for this run.")
    p.add_argument("--dimension", choices=["domain", "length", "compression", "all"], default="all")
    p.add_argument("--mode", choices=["screening", "full"], default="screening")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--output-root", type=Path, default=Path("results/generalization"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-runs", type=int, default=None,
                   help="Optional smoke-test cap. Do not use for final reported results.")
    args = p.parse_args()

    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1]")

    specs = build_specs(args.profile, args.output_root, args.dimension, args.mode)
    if args.max_runs is not None:
        specs = specs[: args.max_runs]

    manifest = args.output_root / "run_manifest.jsonl"
    if args.execute and manifest.exists():
        manifest.unlink()

    env = build_env(args.longbench_path, args.ruler_path)

    print(f"Planned runs: {len(specs)}")
    if args.longbench_path:
        print(f"LongBench root: {args.longbench_path}")
    if args.ruler_path:
        print(f"RULER root: {args.ruler_path}")

    for idx, spec in enumerate(specs, 1):
        cmd = build_command(
            spec,
            model=args.model,
            device=args.device,
            fraction=args.fraction,
        )
        print(f"[{idx:03d}/{len(specs):03d}] {spec.run_id}")
        print("  " + " ".join(cmd))
        if args.execute:
            append_manifest(
                manifest,
                spec,
                cmd,
                longbench_path=args.longbench_path,
                ruler_path=args.ruler_path,
            )
            Path(spec.run_root).mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True, env=env)

    if args.execute:
        print(f"Manifest: {manifest}")
    else:
        print("Dry run only. Add --execute after checking paths and run count.")


if __name__ == "__main__":
    main()
