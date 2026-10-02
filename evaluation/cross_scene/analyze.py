"""Cross-test LU-KV head-budget profiles on unseen *contexts*, by task.

All losses are offline oracle/scorer proxies. Positive improvement over the
official static profile is a candidate failure signal, not answer-quality proof.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.active_calibration.gate0 import (
    PairPaths, evaluate_profile, keep_counts, load_pair,
)


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest requires nonempty records")
    ids = set()
    hashes = set()
    grouped = defaultdict(lambda: defaultdict(int))
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("Manifest record must be an object")
        context_id = item.get("context_id")
        task = item.get("task")
        split = item.get("split")
        digest = item.get("context_sha256")
        if not context_id or not task or split not in {"calibration", "heldout"} or not digest:
            raise ValueError("Manifest record lacks context_id/task/split/context_sha256")
        if context_id in ids or digest in hashes:
            raise ValueError("Duplicate context ID or identical context across splits/tasks")
        ids.add(context_id)
        hashes.add(digest)
        grouped[task][split] += 1
    if len(grouped) < 2:
        raise ValueError("Need at least two tasks")
    for task, counts in grouped.items():
        if counts["calibration"] < 1 or counts["heldout"] < 1:
            raise ValueError(f"{task}: requires both calibration and heldout contexts")
    return records


def prepare_pairs(
    records: list[dict], raw_root: Path, *, method: str, sink: int,
    window: int, compression: float, threshold: int,
):
    grouped = defaultdict(lambda: {"calibration": [], "heldout": []})
    shape = None
    index = int(round(compression * 100)) - 1
    for number, item in enumerate(records, 1):
        context_dir = raw_root / item["context_id"]
        oracle_path = context_dir / "question_0.npy"
        scorer_path = context_dir / f"{method}.npy"
        if not oracle_path.is_file() or not scorer_path.is_file():
            raise FileNotFoundError(
                f"Missing raw pair in {context_dir}; step 1 must finish every manifest context"
            )
        pair = load_pair(
            PairPaths(item["context_id"], oracle_path, scorer_path),
            sink=sink,
            window=window,
            compression_index=index,
            threshold=threshold,
            expected_heads=shape,
        )
        shape = pair.prune_ratio.shape
        grouped[item["task"]][item["split"]].append(pair)
        print(f"Loaded {number}/{len(records)}: {item['context_id']}", flush=True)
    return grouped, shape


def build_profiles(grouped, official_profile: np.ndarray) -> dict[str, np.ndarray]:
    profiles = {"official_lu": official_profile}
    all_calibration = []
    for task in sorted(grouped):
        pairs = grouped[task]["calibration"]
        profiles[f"task_{task}"] = np.mean(
            [pair.prune_ratio for pair in pairs], axis=0
        )
        all_calibration.extend(pairs)
    profiles["pooled"] = np.mean(
        [pair.prune_ratio for pair in all_calibration], axis=0
    )
    return profiles


def transfer_matrix(
    grouped, profiles: dict[str, np.ndarray], *, compression: float,
    sink: int, window: int, max_budget_gap_fraction: float,
) -> tuple[dict, list[dict]]:
    profiles = dict(profiles)
    profiles["uniform"] = np.full_like(profiles["official_lu"], compression)
    matrix = {}
    rows = []
    for task in sorted(grouped):
        heldout = grouped[task]["heldout"]
        matrix[task] = {}
        for profile_name, profile in sorted(profiles.items()):
            result = evaluate_profile(profile, heldout, sink=sink, window=window)
            matrix[task][profile_name] = {
                "mean_proxy_loss": result["mean_proxy_loss"],
                "mean_kept_entries": result["mean_kept_entries"],
            }
            for pair, loss in zip(heldout, result["per_pair_proxy_loss"]):
                kept = int(keep_counts(profile, pair.context_length).sum())
                rows.append({
                    "task": task,
                    "context_id": pair.pair_id,
                    "profile": profile_name,
                    "proxy_loss": float(loss),
                    "kept_entries": kept,
                })

    budgets = defaultdict(list)
    for row in rows:
        budgets[row["context_id"]].append(row["kept_entries"])
    unfair = {
        context_id: max(values) - min(values)
        for context_id, values in budgets.items()
        if (max(values) - min(values)) / max(values) > max_budget_gap_fraction
    }
    if unfair:
        raise ValueError(
            f"Profiles use materially different total KV budgets: {unfair}. "
            "Do not interpret a loss gap as allocation quality."
        )
    return matrix, rows


def paired_difference(rows: list[dict], task: str, profile_a: str, profile_b: str):
    by_context = defaultdict(dict)
    for row in rows:
        if row["task"] == task:
            by_context[row["context_id"]][row["profile"]] = row["proxy_loss"]
    return [
        values[profile_a] - values[profile_b]
        for values in by_context.values()
    ]


def oracle_headroom(grouped, official_profile, raw_root: Path, *, sink: int, window: int):
    """Post-hoc, non-deployable budget candidate and token-ranking bound."""
    output = {}
    for task in sorted(grouped):
        records = []
        for pair in grouped[task]["heldout"]:
            official_loss = evaluate_profile(
                official_profile, [pair], sink=sink, window=window
            )["mean_proxy_loss"]
            own_loss = evaluate_profile(
                pair.prune_ratio, [pair], sink=sink, window=window
            )["mean_proxy_loss"]
            oracle = np.load(
                raw_root / pair.pair_id / "question_0.npy", allow_pickle=False
            )
            if oracle.shape[:2] != official_profile.shape or oracle.shape[-1] != pair.context_length:
                raise ValueError(f"{pair.pair_id}: oracle shape changed")
            counts = keep_counts(official_profile, pair.context_length)
            # Post-hoc upper bound at the exact official per-head budgets:
            # select directly by oracle over all cached positions. This stays
            # valid when a transferred profile maps below sink+window.
            sorted_oracle = np.sort(oracle, axis=-1)[..., ::-1]
            ideal_prefix = np.concatenate((
                np.zeros((*official_profile.shape, 1), dtype=np.float64),
                np.cumsum(sorted_oracle, axis=-1, dtype=np.float64),
            ), axis=-1)
            ideal_kept = np.take_along_axis(
                ideal_prefix, counts[..., None], axis=-1
            )[..., 0].sum()
            total = pair.prefix_utility[..., -1].sum()
            ideal_token_loss = float(1.0 - ideal_kept / total)
            records.append({
                "context_id": pair.pair_id,
                "official_proxy_loss": float(official_loss),
                "in_sample_lu_budget_proxy_loss": float(own_loss),
                "oracle_token_selection_proxy_loss": ideal_token_loss,
                "fixed_scorer_budget_headroom": float(official_loss - own_loss),
                "token_scorer_headroom_at_official_budget": float(
                    official_loss - ideal_token_loss
                ),
                "in_sample_lu_budget_kept_entries": int(
                    keep_counts(pair.prune_ratio, pair.context_length).sum()
                ),
                "official_kept_entries": int(counts.sum()),
            })
        output[task] = {
            "mean_fixed_scorer_budget_headroom": float(np.mean([
                item["fixed_scorer_budget_headroom"] for item in records
            ])),
            "mean_token_scorer_headroom_at_official_budget": float(np.mean([
                item["token_scorer_headroom_at_official_budget"] for item in records
            ])),
            "per_context": records,
        }
    return output


def analyze(args) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    if not (0 < args.compression <= args.threshold / 100):
        raise ValueError("Compression must be positive and <= threshold/100")
    if abs(args.compression * 100 - round(args.compression * 100)) > 1e-8:
        raise ValueError("Compression must be an integer percentage")
    if args.sink < 0 or args.window < 0 or args.max_budget_gap_fraction < 0:
        raise ValueError("sink/window/budget-gap fraction must be nonnegative")
    records = load_manifest(args.manifest)
    grouped, shape = prepare_pairs(
        records, args.raw_root, method=args.method,
        sink=args.sink, window=args.window,
        compression=args.compression, threshold=args.threshold,
    )
    official = np.load(args.official_profile, allow_pickle=False)
    index = int(round(args.compression * 100)) - 1
    if official.ndim != 3 or official.shape[0] <= index or official.shape[1:] != shape:
        raise ValueError(f"Official curve shape {official.shape} does not match {shape}")
    official_at_ratio = np.asarray(official[index], dtype=np.float64)
    if not np.isfinite(official_at_ratio).all():
        raise ValueError("Official profile contains non-finite values")
    profiles = build_profiles(grouped, official_at_ratio)
    matrix, rows = transfer_matrix(
        grouped, profiles, compression=args.compression,
        sink=args.sink, window=args.window,
        max_budget_gap_fraction=args.max_budget_gap_fraction,
    )
    bounds = oracle_headroom(
        grouped, official_at_ratio, args.raw_root,
        sink=args.sink, window=args.window,
    )
    comparisons = {}
    for task in sorted(grouped):
        key = f"task_{task}"
        improvement = paired_difference(rows, task, "official_lu", key)
        pooled_improvement = paired_difference(rows, task, "pooled", key)
        comparisons[task] = {
            "task_profile_vs_official_mean_proxy_loss_reduction": float(np.mean(improvement)),
            "task_profile_vs_pooled_mean_proxy_loss_reduction": float(np.mean(pooled_improvement)),
            "task_profile_beats_official_contexts": int(np.sum(np.asarray(improvement) > 0)),
            "heldout_contexts": len(improvement),
            "per_context_reduction_vs_official": improvement,
        }
    result = {
        "warning": (
            "Offline LU-KV oracle/scorer proxy only. Task-specific profiles use "
            "few calibration contexts from LongBench test split; no answer quality "
            "or independent benchmark has been measured."
        ),
        "settings": {
            "manifest": str(args.manifest),
            "raw_root": str(args.raw_root),
            "official_profile": str(args.official_profile),
            "method": args.method,
            "compression": args.compression,
            "sink": args.sink,
            "window": args.window,
        },
        "counts": {
            task: {
                split: len(grouped[task][split])
                for split in ("calibration", "heldout")
            }
            for task in sorted(grouped)
        },
        "matrix": matrix,
        "comparisons": comparisons,
        "posthoc_oracle_headroom_not_deployable": bounds,
        "max_abs_prune_ratio_difference_lcc_vs_repobench": (
            float(np.max(np.abs(profiles["task_lcc"] - profiles["task_repobench-p"])))
            if "task_lcc" in profiles and "task_repobench-p" in profiles else None
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "transfer_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / "per_context.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["task", "context_id", "profile", "proxy_loss", "kept_entries"]
        )
        writer.writeheader()
        writer.writerows(rows)
    for profile_name, profile in profiles.items():
        if profile_name != "uniform":
            np.save(args.output_dir / f"{profile_name}_ratio_{index + 1}.npy", profile)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="snapkv")
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--compression", type=float, default=0.80)
    parser.add_argument("--threshold", type=int, default=99)
    parser.add_argument("--max-budget-gap-fraction", type=float, default=0.001)
    args = parser.parse_args()
    result = analyze(args)
    print("Proxy loss matrix (lower is better; held-out contexts only):")
    for task, profiles in result["matrix"].items():
        print(task + ": " + ", ".join(
            f"{name}={data['mean_proxy_loss']:.4f}"
            for name, data in profiles.items()
        ))
    for task, comparison in result["comparisons"].items():
        print(
            f"{task}: task-vs-official proxy loss reduction="
            f"{comparison['task_profile_vs_official_mean_proxy_loss_reduction']:+.5f}, "
            f"wins={comparison['task_profile_beats_official_contexts']}/"
            f"{comparison['heldout_contexts']}"
        )
    print(f"Wrote {args.output_dir / 'transfer_results.json'}")
    print("STOP: these data alone cannot establish downstream answer-quality gain.")


if __name__ == "__main__":
    main()
