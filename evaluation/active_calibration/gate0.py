"""Test whether the choice of LU-KV calibration questions matters at all.

This is an *offline proxy* experiment. It reuses the per-question full-attention
arrays produced by evaluation/curve_data/step1_llama.py and the unchanged LU-KV
per-question allocator. It does not measure generated-answer quality.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluation.curve_data.step2_compute_curve import (
    compute_optimal_budget_for_pair,
    exact_runtime_keep_counts,
)


@dataclass(frozen=True)
class PairPaths:
    pair_id: str
    oracle_path: Path
    scorer_path: Path


@dataclass
class PairData:
    pair_id: str
    prune_ratio: np.ndarray  # [layer, KV head], one target compression level
    prefix_utility: np.ndarray  # [layer, KV head, runtime-ranked tokens + 1]
    context_length: int


def discover_pairs(raw_root: Path, method: str) -> list[PairPaths]:
    """Require a scorer file for every discovered question; never silently drop it."""
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw curve directory does not exist: {raw_root}")

    pairs: list[PairPaths] = []
    for context_dir in sorted(raw_root.glob("context_*")):
        if not context_dir.is_dir():
            continue
        question_files = sorted(context_dir.glob("question_*.npy"))
        scorer_path = context_dir / f"{method}.npy"
        if question_files and not scorer_path.is_file():
            raise FileNotFoundError(
                f"{context_dir} has question files but no {method}.npy; "
                "run LU-KV step 1 with the same method first"
            )
        for oracle_path in question_files:
            pairs.append(
                PairPaths(
                    pair_id=f"{context_dir.name}/{oracle_path.stem}",
                    oracle_path=oracle_path,
                    scorer_path=scorer_path,
                )
            )
    if not pairs:
        raise FileNotFoundError(
            f"No context_*/question_*.npy found under {raw_root}. "
            "A static *_avg_ratio.npy file is NOT per-question raw data."
        )
    return pairs


def load_pair(
    paths: PairPaths,
    *,
    sink: int,
    window: int,
    compression_index: int,
    threshold: int,
    expected_heads: tuple[int, int] | None,
) -> PairData:
    oracle = np.load(paths.oracle_path, allow_pickle=False)
    scorer = np.load(paths.scorer_path, allow_pickle=False)
    if oracle.ndim != 3 or oracle.shape != scorer.shape:
        raise ValueError(
            f"{paths.pair_id}: oracle/scorer must have matching [layer, head, token] "
            f"shapes, got {oracle.shape} and {scorer.shape}"
        )
    if expected_heads is not None and oracle.shape[:2] != expected_heads:
        raise ValueError(
            f"{paths.pair_id}: head shape {oracle.shape[:2]} != {expected_heads}"
        )
    if not np.isfinite(oracle).all() or not np.isfinite(scorer).all():
        raise ValueError(f"{paths.pair_id}: non-finite oracle/scorer values")
    if (oracle < 0).any():
        raise ValueError(f"{paths.pair_id}: oracle utility must be nonnegative")

    length = int(oracle.shape[-1])
    if length <= sink + window:
        raise ValueError(
            f"{paths.pair_id}: context length {length} <= sink + window {sink + window}"
        )

    curve = compute_optimal_budget_for_pair(
        oracle,
        scorer,
        sink_size=sink,
        window_size=window,
        prune_threshold_int=threshold,
        layerwise=True,
    )
    if curve is None:
        raise ValueError(f"{paths.pair_id}: LU-KV could not compute an allocation curve")

    # Replay the actual LUPress ranking over all positions. Sink/window are
    # score boosts at runtime, not a hard per-head keep floor. When a static
    # prune-ratio profile is transferred to another sequence length, its
    # integer keep count may legitimately fall below sink+window.
    runtime_scorer = scorer.copy()
    safe_sink = min(sink, length)
    window_start = max(0, length - window)
    for layer in range(runtime_scorer.shape[0]):
        layer_max = runtime_scorer[layer].max()
        if safe_sink:
            runtime_scorer[layer, :, :safe_sink] = layer_max
        if window:
            runtime_scorer[layer, :, window_start:] = layer_max

    # Match torch.argsort(..., descending=True, stable=True) in LUPress.
    order = np.argsort(-runtime_scorer, axis=-1, kind="stable")
    aligned_oracle = np.take_along_axis(oracle, order, axis=-1)
    prefix_utility = np.concatenate(
        (
            np.zeros((*oracle.shape[:2], 1), dtype=np.float64),
            np.cumsum(aligned_oracle, axis=-1, dtype=np.float64),
        ),
        axis=-1,
    )
    if prefix_utility[..., -1].sum() <= 0:
        raise ValueError(f"{paths.pair_id}: all-token oracle utilities are zero")

    return PairData(
        pair_id=paths.pair_id,
        prune_ratio=np.asarray(curve[compression_index], dtype=np.float64),
        prefix_utility=prefix_utility,
        context_length=length,
    )


def keep_counts(profile: np.ndarray, length: int) -> np.ndarray:
    return np.stack(
        [exact_runtime_keep_counts(layer, length) for layer in profile], axis=0
    )


def evaluate_profile(
    profile: np.ndarray, heldout: list[PairData], *, sink: int, window: int
) -> dict:
    """Return middle-token oracle loss after ranking tokens with the fixed scorer."""
    losses = []
    kept_counts = []
    for pair in heldout:
        counts = keep_counts(profile, pair.context_length)
        max_keep = pair.prefix_utility.shape[-1] - 1
        if np.any(counts < 1) or np.any(counts > max_keep):
            raise ValueError(
                f"{pair.pair_id}: runtime keep count is outside [1, {max_keep}]"
            )
        kept = np.take_along_axis(
            pair.prefix_utility, counts[..., None], axis=-1
        )[..., 0].sum()
        total = pair.prefix_utility[..., -1].sum()
        losses.append(float(1.0 - kept / total))
        kept_counts.append(int(counts.sum()))
    return {
        "mean_proxy_loss": float(np.mean(losses)),
        "per_pair_proxy_loss": losses,
        "mean_kept_entries": float(np.mean(kept_counts)),
    }


def parse_sizes(raw: str) -> list[int]:
    try:
        sizes = [int(part.strip()) for part in raw.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from exc
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must be unique positive integers")
    return sorted(sizes)


def run(args: argparse.Namespace) -> dict:
    if not (1 <= args.threshold <= 99):
        raise ValueError("threshold must be an integer from 1 to 99")
    if not (0 < args.compression <= args.threshold / 100):
        raise ValueError("compression must be positive and no greater than threshold/100")
    pct = args.compression * 100
    if abs(pct - round(pct)) > 1e-8:
        raise ValueError("compression must be an integer percentage, e.g. 0.80")
    if args.sink < 0 or args.window < 0:
        raise ValueError("sink/window must be nonnegative")
    if args.repeats < 1:
        raise ValueError("repeats must be >= 1")
    if args.calibration_size < 2:
        raise ValueError("calibration-size must be >= 2")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {args.output_dir}; choose a new run name"
        )
    sizes = parse_sizes(args.subset_sizes)
    if max(sizes) >= args.calibration_size:
        raise ValueError("each subset size must be smaller than calibration-size")

    paths = discover_pairs(args.raw_root, args.method)
    if len(paths) <= args.calibration_size:
        raise ValueError(
            f"Need more than {args.calibration_size} raw pairs for an untouched holdout; "
            f"found {len(paths)}"
        )

    compression_index = int(round(pct)) - 1
    loaded: list[PairData] = []
    head_shape = None
    for index, item in enumerate(paths, 1):
        pair = load_pair(
            item,
            sink=args.sink,
            window=args.window,
            compression_index=compression_index,
            threshold=args.threshold,
            expected_heads=head_shape,
        )
        head_shape = pair.prune_ratio.shape
        loaded.append(pair)
        print(f"Loaded {index}/{len(paths)}: {pair.pair_id}", flush=True)

    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(loaded))
    calibration = [loaded[i] for i in permutation[: args.calibration_size]]
    heldout = [loaded[i] for i in permutation[args.calibration_size :]]

    full_profile = np.mean([pair.prune_ratio for pair in calibration], axis=0)
    uniform_profile = np.full_like(full_profile, args.compression)
    full_result = evaluate_profile(full_profile, heldout, sink=args.sink, window=args.window)
    uniform_result = evaluate_profile(
        uniform_profile, heldout, sink=args.sink, window=args.window
    )

    records = []
    example_profiles = {}
    for size in sizes:
        for repeat in range(args.repeats):
            chosen = sorted(int(i) for i in rng.choice(args.calibration_size, size, replace=False))
            profile = np.mean([calibration[i].prune_ratio for i in chosen], axis=0)
            evaluated = evaluate_profile(profile, heldout, sink=args.sink, window=args.window)
            records.append(
                {
                    "subset_size": size,
                    "repeat": repeat,
                    "chosen_calibration_ids": [calibration[i].pair_id for i in chosen],
                    "proxy_loss": evaluated["mean_proxy_loss"],
                    "delta_proxy_loss_vs_full": evaluated["mean_proxy_loss"]
                    - full_result["mean_proxy_loss"],
                    "mean_abs_prune_ratio_difference": float(
                        np.mean(np.abs(profile - full_profile))
                    ),
                    "mean_kept_entries": evaluated["mean_kept_entries"],
                }
            )
            if repeat == 0:
                example_profiles[size] = profile

    summary = {}
    for size in sizes:
        values = np.array(
            [row["delta_proxy_loss_vs_full"] for row in records if row["subset_size"] == size]
        )
        distances = np.array(
            [row["mean_abs_prune_ratio_difference"] for row in records if row["subset_size"] == size]
        )
        summary[str(size)] = {
            "median_delta_proxy_loss": float(np.median(values)),
            "p90_delta_proxy_loss": float(np.quantile(values, 0.9)),
            "best_delta_proxy_loss": float(np.min(values)),
            "median_abs_prune_ratio_difference": float(np.median(distances)),
        }

    output = {
        "warning": (
            "Offline LU-KV oracle/scorer proxy only; not answer quality or a "
            "validated active-selection algorithm. All questions from one context "
            "cannot establish cross-context generalization."
        ),
        "settings": {
            "raw_root": str(args.raw_root.resolve()),
            "method": args.method,
            "sink": args.sink,
            "window": args.window,
            "compression": args.compression,
            "threshold": args.threshold,
            "seed": args.seed,
            "calibration_size": args.calibration_size,
            "heldout_size": len(heldout),
            "repeats": args.repeats,
        },
        "calibration_ids": [pair.pair_id for pair in calibration],
        "heldout_ids": [pair.pair_id for pair in heldout],
        "full_calibration": full_result,
        "uniform_budget": uniform_result,
        "random_subset_summary": summary,
        "random_subset_records": records,
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "full_calibration_profile.npy", full_profile)
    np.save(args.output_dir / "uniform_profile.npy", uniform_profile)
    for size, profile in example_profiles.items():
        np.save(args.output_dir / f"random_{size}_repeat0_profile.npy", profile)
    (args.output_dir / "gate0_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Full calibration proxy loss: {full_result['mean_proxy_loss']:.6f}")
    print(f"Uniform budget proxy loss:  {uniform_result['mean_proxy_loss']:.6f}")
    for size in sizes:
        item = summary[str(size)]
        print(
            f"Random {size}: median Δloss={item['median_delta_proxy_loss']:+.6f}, "
            f"p90 Δloss={item['p90_delta_proxy_loss']:+.6f}, "
            f"median |Δprune ratio|={item['median_abs_prune_ratio_difference']:.6f}"
        )
    print(f"Saved: {args.output_dir / 'gate0_results.json'}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="snapkv")
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--compression", type=float, default=0.80)
    parser.add_argument("--threshold", type=int, default=99)
    parser.add_argument("--calibration-size", type=int, default=20)
    parser.add_argument("--subset-sizes", default="4,8,16")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

