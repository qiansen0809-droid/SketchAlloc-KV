import argparse
import glob
import heapq
import multiprocessing as mp
import os

import numpy as np
from tqdm import tqdm


def compute_lower_convex_hull(y_values):
    n = len(y_values)
    stack = [0]
    for i in range(1, n):
        while len(stack) >= 2:
            p1, p2, p3 = stack[-2], stack[-1], i
            x1, y1 = p1, y_values[p1]
            x2, y2 = p2, y_values[p2]
            x3, y3 = p3, y_values[p3]
            cross_product = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if cross_product <= 0:
                stack.pop()
            else:
                break
        stack.append(i)
    return stack


def apply_convex_hull_smoothing(aligned_gt):
    rows, cols, length = aligned_gt.shape
    smoothed_gt = np.zeros_like(aligned_gt)
    for r in range(rows):
        for c in range(cols):
            raw_vals = aligned_gt[r, c]
            vals_to_discard = raw_vals[::-1]
            cum_loss = np.concatenate(([0], np.cumsum(vals_to_discard)))
            hull_indices = compute_lower_convex_hull(cum_loss)
            reconstructed_vals = np.zeros(length)
            for i in range(len(hull_indices) - 1):
                idx_start, idx_end = hull_indices[i], hull_indices[i + 1]
                slope = (cum_loss[idx_end] - cum_loss[idx_start]) / (idx_end - idx_start)
                reconstructed_vals[idx_start:idx_end] = slope
            smoothed_gt[r, c] = reconstructed_vals[::-1]
    return smoothed_gt


def compute_priority_scores(data_gt, data_pred, sink_size, window_size):
    """
    Align oracle long-horizon utility by the token scorer ranking and apply the
    same convex-hull relaxation used by LU-KV's global budget solver.

    Returned priority_scores are ordered from highest scorer rank to lowest.
    If k compressible tokens are currently kept in a head:
      - removing one more token costs priority_scores[k - 1]
      - adding one token back gains priority_scores[k]
    """
    rows, cols, length = data_gt.shape
    eff_len = length - sink_size - window_size
    if eff_len <= 0:
        return None, 0

    mid_gt = data_gt[:, :, sink_size : length - window_size]
    mid_pred = data_pred[:, :, sink_size : length - window_size]
    aligned_mid_gt = np.zeros_like(mid_gt)

    for r in range(rows):
        for c in range(cols):
            sort_indices = np.argsort(mid_pred[r, c])[::-1]
            aligned_mid_gt[r, c] = mid_gt[r, c][sort_indices]

    return apply_convex_hull_smoothing(aligned_mid_gt), eff_len


def compute_optimal_budget_for_pair(
    data_gt,
    data_pred,
    sink_size,
    window_size,
    prune_threshold_int=99,
    layerwise=True,
):
    """Returns the 99-step allocation curve for a single question pairing."""
    limit_ratio = prune_threshold_int / 100.0
    rows, cols, length = data_gt.shape

    priority_scores_mid, eff_len = compute_priority_scores(
        data_gt,
        data_pred,
        sink_size=sink_size,
        window_size=window_size,
    )
    if priority_scores_mid is None:
        return None

    results = np.zeros((99, rows, cols), dtype=np.float64)
    user_min_keep = int(round((1.0 - limit_ratio) * length))
    min_keep_limit = max(user_min_keep, sink_size + window_size)

    if layerwise:
        total_elements = rows * cols * length
        current_keep_count = np.full((rows, cols), length, dtype=np.int32)
        total_discarded = 0
        heap = []

        for r in range(rows):
            for c in range(cols):
                if eff_len - 1 >= 0:
                    val = priority_scores_mid[r, c, eff_len - 1]
                    heapq.heappush(heap, (val, r, c, eff_len - 1))

        for i in range(99):
            target_d = int(round(((i + 1) / 100.0) * total_elements))
            global_x = (i + 1) / 100.0
            if global_x > limit_ratio:
                results[i] = np.full((rows, cols), global_x)
                continue

            while total_discarded < target_d and heap:
                val, r, c, idx_in_mid = heapq.heappop(heap)
                if current_keep_count[r, c] > min_keep_limit:
                    current_keep_count[r, c] -= 1
                    total_discarded += 1
                    next_idx = idx_in_mid - 1
                    if next_idx >= 0:
                        heapq.heappush(
                            heap,
                            (priority_scores_mid[r, c, next_idx], r, c, next_idx),
                        )
            results[i] = 1.0 - (current_keep_count / length)
    else:
        layer_elements = cols * length
        for r in range(rows):
            current_keep_count_layer = np.full(cols, length, dtype=np.int32)
            total_discarded_layer = 0
            heap = []

            for c in range(cols):
                if eff_len - 1 >= 0:
                    val = priority_scores_mid[r, c, eff_len - 1]
                    heapq.heappush(heap, (val, c, eff_len - 1))

            for i in range(99):
                target_d_layer = int(round(((i + 1) / 100.0) * layer_elements))
                global_x = (i + 1) / 100.0
                if global_x > limit_ratio:
                    results[i, r, :] = global_x
                    continue

                while total_discarded_layer < target_d_layer and heap:
                    val, c, idx_in_mid = heapq.heappop(heap)
                    if current_keep_count_layer[c] > min_keep_limit:
                        current_keep_count_layer[c] -= 1
                        total_discarded_layer += 1
                        next_idx = idx_in_mid - 1
                        if next_idx >= 0:
                            heapq.heappush(
                                heap,
                                (priority_scores_mid[r, c, next_idx], c, next_idx),
                            )
                results[i, r, :] = 1.0 - (current_keep_count_layer / length)

    return results


def exact_runtime_keep_counts(local_prune_ratios, length):
    """
    Reproduce LUPress._curve_keep_counts for one layer.

    LU-KV stores static per-head prune ratios. At runtime those ratios are
    converted to integer keep counts by flooring each head and distributing
    the rounded layer-level remainder to the largest fractional parts.
    """
    ratios = np.asarray(local_prune_ratios, dtype=np.float64)
    ideal = (1.0 - ratios) * length
    total_keep_target = int(np.round(ideal.sum()))
    keep_counts = np.floor(ideal).astype(np.int64)

    remainder = total_keep_target - int(keep_counts.sum())
    if remainder > 0:
        fractional = ideal - keep_counts
        order = np.argsort(-fractional, kind="stable")
        keep_counts[order[: min(remainder, len(keep_counts))]] += 1

    return np.clip(keep_counts, 1, length)


def compute_boundary_marginals_for_pair(
    data_gt,
    data_pred,
    static_budget_curve,
    sink_size,
    window_size,
    marginal_step_tokens=1,
    prune_threshold_int=99,
):
    """
    Measure LU-KV remove-cost and next-gain at the *final static averaged budget*
    for this calibration pair.

    This is an additional Gate-0 export. It does not change LU-KV's solver.
    It reuses the exact scorer-aligned, convex-hull-smoothed oracle utilities
    already used by the solver, then samples the local boundary around each
    static budget point.

    remove_cost[i,l,h]:
        oracle utility lost by removing marginal_step_tokens from unit (l,h)

    next_gain[i,l,h]:
        oracle utility recovered by adding marginal_step_tokens to unit (l,h)

    Invalid boundary moves are NaN.
    """
    if marginal_step_tokens <= 0:
        raise ValueError("marginal_step_tokens must be positive")

    limit_ratio = prune_threshold_int / 100.0
    rows, cols, length = data_gt.shape
    if static_budget_curve.shape != (99, rows, cols):
        raise ValueError(
            "static_budget_curve shape mismatch: "
            f"expected {(99, rows, cols)}, got {static_budget_curve.shape}"
        )

    priority_scores_mid, eff_len = compute_priority_scores(
        data_gt,
        data_pred,
        sink_size=sink_size,
        window_size=window_size,
    )
    if priority_scores_mid is None:
        return None

    remove_cost = np.full((99, rows, cols), np.nan, dtype=np.float64)
    next_gain = np.full((99, rows, cols), np.nan, dtype=np.float64)

    user_min_keep = int(round((1.0 - limit_ratio) * length))
    min_keep_limit = max(user_min_keep, sink_size + window_size)
    protected = sink_size + window_size

    for i in range(99):
        global_x = (i + 1) / 100.0
        if global_x > limit_ratio:
            continue

        # Mirror the exact integer conversion used by runtime LUPress,
        # including the per-layer fractional remainder distribution.
        keep_counts_by_layer = np.stack(
            [
                exact_runtime_keep_counts(static_budget_curve[i, r], length)
                for r in range(rows)
            ],
            axis=0,
        )

        for r in range(rows):
            for c in range(cols):
                keep_count = int(keep_counts_by_layer[r, c])

                # priority_scores_mid excludes sink/window positions. If the
                # transferred static profile falls inside that protected span,
                # the local marginal cannot be recovered from this mid-token
                # utility vector, so leave the corresponding entry NaN.
                mid_keep = keep_count - protected

                if (
                    keep_count - marginal_step_tokens >= min_keep_limit
                    and mid_keep >= marginal_step_tokens
                    and mid_keep <= eff_len
                ):
                    start = mid_keep - marginal_step_tokens
                    remove_cost[i, r, c] = float(
                        priority_scores_mid[r, c, start:mid_keep].sum()
                    )

                if (
                    keep_count >= protected
                    and keep_count + marginal_step_tokens <= length
                    and mid_keep + marginal_step_tokens <= eff_len
                ):
                    stop = mid_keep + marginal_step_tokens
                    next_gain[i, r, c] = float(
                        priority_scores_mid[r, c, mid_keep:stop].sum()
                    )

    return remove_cost, next_gain


def worker_task(task_args):
    q_file, pred_path, sink_size, window_size, threshold, layerwise = task_args
    try:
        data_gt = np.load(q_file)
        data_pred = np.load(pred_path)

        res = compute_optimal_budget_for_pair(
            data_gt,
            data_pred,
            sink_size=sink_size,
            window_size=window_size,
            prune_threshold_int=threshold,
            layerwise=layerwise,
        )
        return res
    except Exception as e:
        print(f"\n[Worker Error] Failed to process {q_file}: {e}")
        return None


def marginal_worker_task(task_args):
    (
        q_file,
        pred_path,
        static_budget_curve,
        sink_size,
        window_size,
        marginal_step_tokens,
        threshold,
    ) = task_args

    try:
        data_gt = np.load(q_file)
        data_pred = np.load(pred_path)
        return compute_boundary_marginals_for_pair(
            data_gt,
            data_pred,
            static_budget_curve=static_budget_curve,
            sink_size=sink_size,
            window_size=window_size,
            marginal_step_tokens=marginal_step_tokens,
            prune_threshold_int=threshold,
        )
    except Exception as e:
        print(f"\n[Marginal Worker Error] Failed to process {q_file}: {e}")
        return None


def collect_method_tasks(context_dirs, method):
    tasks = []
    for c_dir in context_dirs:
        pred_path = os.path.join(c_dir, f"{method}.npy")
        if not os.path.exists(pred_path):
            continue

        q_files = glob.glob(os.path.join(c_dir, "question_*.npy"))
        for q_file in q_files:
            tasks.append((q_file, pred_path))
    return tasks


def average_nan_arrays(total, count, values):
    mask = np.isfinite(values)
    total[mask] += values[mask]
    count[mask] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to step 1 temporary folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store the final curve files",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help="Prefix naming for the output files",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Method configs formatted as method:sink:window (e.g. ea:4:1 snapkv:4:32)",
    )
    parser.add_argument("--threshold", type=int, default=99)
    parser.add_argument(
        "--layerwise",
        action="store_true",
        help="Enable global budget allocation across layers",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=30,
        help="Number of concurrent processes",
    )
    parser.add_argument(
        "--export_marginals",
        action="store_true",
        help=(
            "Gate-0 extension: export remove-cost / next-gain at the final "
            "static LU-KV budget boundary."
        ),
    )
    parser.add_argument(
        "--marginal_step_tokens",
        type=int,
        default=1,
        help=(
            "Number of ranked KV entries moved by one Gate-0 logical swap. "
            "Use the same value as Gate-0 --swap-size."
        ),
    )
    parser.add_argument(
        "--static_budget_curve_path",
        type=str,
        default=None,
        help=(
            "Optional existing LU-KV static allocation curve (.npy) whose "
            "boundary should define Gate-0 marginals. Use this when Gate-0 "
            "runtime uses the official bundled LU-KV curve, so donor/receiver "
            "marginals are sampled at exactly the same allocation profile."
        ),
    )
    args = parser.parse_args()

    method_configs = {}
    for cfg in args.configs:
        parts = cfg.split(":")
        if len(parts) == 3:
            method_configs[parts[0]] = {
                "sink": int(parts[1]),
                "window": int(parts[2]),
            }
        else:
            print(
                f"Warning: Configuration '{cfg}' is invalid. "
                "Must be format 'method:sink:window'."
            )

    context_dirs = glob.glob(os.path.join(args.input_dir, "context_*"))
    if not context_dirs:
        print(f"Error: No context_* directories found in {args.input_dir}!")
        raise SystemExit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    mode_str = "Global-Layerwise" if args.layerwise else "Layer-Independent"
    print(
        "Starting Convex Hull & Averaging "
        f"| Mode: {mode_str} | Workers: {args.num_workers}"
    )

    for method, config in method_configs.items():
        sink_size = config["sink"]
        window_size = config["window"]
        pair_tasks = collect_method_tasks(context_dirs, method)

        total_tasks = len(pair_tasks)
        if total_tasks == 0:
            print(f"[{method}] Skipped: No valid data/questions found.")
            continue

        tasks = [
            (
                q_file,
                pred_path,
                sink_size,
                window_size,
                args.threshold,
                args.layerwise,
            )
            for q_file, pred_path in pair_tasks
        ]

        total_sum = None
        valid_count = 0
        desc_str = f"Processing [{method}] (sink={sink_size}, win={window_size})"

        with mp.Pool(processes=args.num_workers) as pool:
            for res in tqdm(
                pool.imap_unordered(worker_task, tasks),
                total=total_tasks,
                desc=desc_str,
                unit="q",
            ):
                if res is not None:
                    if total_sum is None:
                        total_sum = np.zeros_like(res, dtype=np.float64)
                    total_sum += res
                    valid_count += 1

        if valid_count == 0:
            print(f"[{method}] Failed: All tasks returned None.\n")
            continue

        avg_data = total_sum / valid_count
        out_path = os.path.join(
            args.output_dir,
            f"{args.output_prefix}_{method}_sink{sink_size}_win{window_size}.npy",
        )
        np.save(out_path, avg_data)
        print(
            f"[{method}] Completed: Merged {valid_count} samples. "
            f"Saved allocation curve to: {out_path}"
        )

        if not args.export_marginals:
            print()
            continue

        marginal_budget_curve = avg_data
        marginal_budget_source = out_path

        if args.static_budget_curve_path is not None:
            marginal_budget_curve = np.load(args.static_budget_curve_path)
            if marginal_budget_curve.shape != avg_data.shape:
                raise ValueError(
                    "external static budget curve shape mismatch: "
                    f"expected {avg_data.shape}, got {marginal_budget_curve.shape}"
                )
            marginal_budget_source = args.static_budget_curve_path
            print(
                f"[{method}] Gate-0 marginals will use external runtime "
                f"allocation curve: {marginal_budget_source}"
            )

        marginal_tasks = [
            (
                q_file,
                pred_path,
                marginal_budget_curve,
                sink_size,
                window_size,
                args.marginal_step_tokens,
                args.threshold,
            )
            for q_file, pred_path in pair_tasks
        ]

        remove_sum = np.zeros_like(marginal_budget_curve, dtype=np.float64)
        gain_sum = np.zeros_like(marginal_budget_curve, dtype=np.float64)
        remove_count = np.zeros_like(marginal_budget_curve, dtype=np.int32)
        gain_count = np.zeros_like(marginal_budget_curve, dtype=np.int32)
        marginal_valid_pairs = 0

        with mp.Pool(processes=args.num_workers) as pool:
            for res in tqdm(
                pool.imap_unordered(marginal_worker_task, marginal_tasks),
                total=total_tasks,
                desc=f"Marginals [{method}] step={args.marginal_step_tokens}",
                unit="q",
            ):
                if res is None:
                    continue

                remove_cost, next_gain = res
                average_nan_arrays(remove_sum, remove_count, remove_cost)
                average_nan_arrays(gain_sum, gain_count, next_gain)
                marginal_valid_pairs += 1

        avg_remove = np.full_like(remove_sum, np.nan, dtype=np.float64)
        avg_gain = np.full_like(gain_sum, np.nan, dtype=np.float64)

        remove_mask = remove_count > 0
        gain_mask = gain_count > 0
        avg_remove[remove_mask] = remove_sum[remove_mask] / remove_count[remove_mask]
        avg_gain[gain_mask] = gain_sum[gain_mask] / gain_count[gain_mask]

        marginal_path = os.path.join(
            args.output_dir,
            (
                f"{args.output_prefix}_{method}_sink{sink_size}_win{window_size}"
                f"_marginal_step{args.marginal_step_tokens}.npz"
            ),
        )
        np.savez_compressed(
            marginal_path,
            remove_cost=avg_remove,
            next_gain=avg_gain,
            valid_remove_count=remove_count,
            valid_gain_count=gain_count,
            budget_prune_ratio=marginal_budget_curve,
            static_budget_curve_path=np.array(marginal_budget_source),
            global_compression_ratio=np.arange(1, 100, dtype=np.float64) / 100.0,
            marginal_step_tokens=np.array(args.marginal_step_tokens, dtype=np.int64),
            sink_size=np.array(sink_size, dtype=np.int64),
            window_size=np.array(window_size, dtype=np.int64),
            calibration_pairs=np.array(marginal_valid_pairs, dtype=np.int64),
        )

        print(
            f"[{method}] Gate-0 marginal profile saved to: {marginal_path}\n"
        )
