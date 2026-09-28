"""Structural Gate 0 for a prompt-by-action KV allocation utility matrix.

This script intentionally evaluates an optimistic upper bound: at test time it
uses the true utilities of a small set of selected sentinel actions to recover
the latent row coordinate. A real no-answer probe must pass a later gate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RankGateSummary:
    num_train_prompts: int
    num_test_prompts: int
    num_actions: int
    rank: int
    num_sentinels: int
    sentinel_action_ids: list[str]
    heldout_variance_explained: float
    heldout_rmse: float
    median_within_prompt_spearman: float
    best_fixed_mean_gain: float
    selected_mean_gain: float
    oracle_mean_gain: float
    selected_mean_regret: float
    headroom_recovery: float | None


def _build_complete_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    required = {"prompt_id", "action_id", "gain"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    duplicated = frame.duplicated(["prompt_id", "action_id"], keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, ["prompt_id", "action_id"]].head().to_dict("records")
        raise ValueError(f"duplicate prompt/action rows: {examples}")

    pivot = frame.pivot(index="prompt_id", columns="action_id", values="gain").sort_index().sort_index(axis=1)
    if pivot.isna().any().any():
        missing_count = int(pivot.isna().sum().sum())
        raise ValueError(f"utility matrix is incomplete: {missing_count} prompt/action cells are missing")
    return pivot.to_numpy(dtype=np.float64), list(pivot.index.astype(str)), list(pivot.columns.astype(str))


def _split_prompts(frame: pd.DataFrame, seed: int) -> tuple[set[str], set[str]]:
    prompt_ids = frame["prompt_id"].astype(str)
    if "split" in frame.columns:
        split_by_prompt = frame.assign(prompt_id=prompt_ids).groupby("prompt_id")["split"].agg(lambda x: set(x))
        inconsistent = split_by_prompt[split_by_prompt.map(len) != 1]
        if not inconsistent.empty:
            raise ValueError("each prompt_id must belong to exactly one split")
        labels = split_by_prompt.map(lambda x: str(next(iter(x))).lower())
        train = set(labels[labels.isin({"train", "discovery", "calibration"})].index.astype(str))
        test = set(labels[labels.isin({"test", "heldout", "held-out"})].index.astype(str))
        if train and test:
            return train, test

    unique = np.array(sorted(prompt_ids.unique()))
    if len(unique) < 4:
        raise ValueError("at least four prompts are required when no explicit train/test split is present")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    cut = max(2, int(round(0.7 * len(unique))))
    cut = min(cut, len(unique) - 1)
    return set(unique[:cut]), set(unique[cut:])


def _greedy_d_optimal_rows(basis: np.ndarray, count: int, ridge: float = 1e-8) -> list[int]:
    """Select informative action rows for recovering latent coordinates."""

    if count < basis.shape[1]:
        raise ValueError("num_sentinels must be at least the requested rank")
    count = min(count, basis.shape[0])
    gram = ridge * np.eye(basis.shape[1], dtype=np.float64)
    selected: list[int] = []
    remaining = set(range(basis.shape[0]))
    for _ in range(count):
        best_idx = None
        best_score = -np.inf
        for idx in remaining:
            row = basis[idx]
            sign, logdet = np.linalg.slogdet(gram + np.outer(row, row))
            score = logdet if sign > 0 else -np.inf
            if score > best_score:
                best_idx = idx
                best_score = score
        assert best_idx is not None
        selected.append(best_idx)
        row = basis[best_idx]
        gram += np.outer(row, row)
        remaining.remove(best_idx)
    return selected


def _row_spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual_rank = pd.Series(actual).rank(method="average").to_numpy(dtype=np.float64)
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(actual_rank) == 0 or np.std(predicted_rank) == 0:
        return 0.0
    return float(np.corrcoef(actual_rank, predicted_rank)[0, 1])


def evaluate_rank_gate(
    train: np.ndarray,
    test: np.ndarray,
    action_ids: list[str],
    rank: int,
    num_sentinels: int,
    ridge: float = 1e-6,
) -> RankGateSummary:
    if rank <= 0 or rank > min(train.shape):
        raise ValueError("rank is outside the valid range for the training matrix")

    static_prior = train.mean(axis=0)
    train_residual = train - static_prior
    _, _, vt = np.linalg.svd(train_residual, full_matrices=False)
    basis = vt[:rank].T
    sentinel_idx = _greedy_d_optimal_rows(basis, num_sentinels)

    observation = basis[sentinel_idx]
    lhs = observation.T @ observation + ridge * np.eye(rank)
    decoder = np.linalg.solve(lhs, observation.T)

    test_residual = test - static_prior
    latent = test_residual[:, sentinel_idx] @ decoder.T
    predicted = static_prior + latent @ basis.T

    error = test - predicted
    denominator = float(np.sum((test - test.mean(axis=0, keepdims=True)) ** 2))
    variance_explained = 1.0 - float(np.sum(error**2)) / denominator if denominator > 0 else 0.0
    rmse = float(np.sqrt(np.mean(error**2)))
    spearman = float(np.median([_row_spearman(a, p) for a, p in zip(test, predicted)]))

    best_fixed_idx = int(np.argmax(static_prior))
    selected_idx = np.argmax(predicted, axis=1)
    oracle_idx = np.argmax(test, axis=1)
    row_idx = np.arange(test.shape[0])
    best_fixed_gain = test[:, best_fixed_idx]
    selected_gain = test[row_idx, selected_idx]
    oracle_gain = test[row_idx, oracle_idx]

    numerator = float(np.mean(selected_gain - best_fixed_gain))
    headroom = float(np.mean(oracle_gain - best_fixed_gain))
    recovery = numerator / headroom if headroom > 0 else None

    return RankGateSummary(
        num_train_prompts=int(train.shape[0]),
        num_test_prompts=int(test.shape[0]),
        num_actions=int(train.shape[1]),
        rank=int(rank),
        num_sentinels=len(sentinel_idx),
        sentinel_action_ids=[action_ids[idx] for idx in sentinel_idx],
        heldout_variance_explained=float(variance_explained),
        heldout_rmse=rmse,
        median_within_prompt_spearman=spearman,
        best_fixed_mean_gain=float(np.mean(best_fixed_gain)),
        selected_mean_gain=float(np.mean(selected_gain)),
        oracle_mean_gain=float(np.mean(oracle_gain)),
        selected_mean_regret=float(np.mean(oracle_gain - selected_gain)),
        headroom_recovery=None if recovery is None else float(recovery),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with prompt_id, action_id, gain, and optional split")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--num-sentinels", type=int, default=6)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    frame["action_id"] = frame["action_id"].astype(str)
    matrix, prompt_ids, action_ids = _build_complete_matrix(frame)
    train_ids, test_ids = _split_prompts(frame, args.seed)
    train_mask = np.array([prompt_id in train_ids for prompt_id in prompt_ids])
    test_mask = np.array([prompt_id in test_ids for prompt_id in prompt_ids])
    if not train_mask.any() or not test_mask.any():
        raise ValueError("train and test splits must both contain at least one prompt")

    summary = evaluate_rank_gate(
        matrix[train_mask],
        matrix[test_mask],
        action_ids,
        rank=args.rank,
        num_sentinels=args.num_sentinels,
        ridge=args.ridge,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
