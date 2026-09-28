from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Summarize CausalDuel-KV Gate-0 MiniGate outputs.")
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/gate0/minigate/summary"))
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260928)
    return p.parse_args()


def sgn(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def sign_accuracy(rows, predictor_key):
    valid = [r for r in rows if sgn(r["answer_nll_gain"]) != 0]
    if not valid:
        return None, 0
    correct = sum(
        sgn(r[predictor_key]) == sgn(r["answer_nll_gain"])
        for r in valid
    )
    return correct / len(valid), len(valid)


def bootstrap_prompt_difference(rows, a_key, b_key, n_boot, seed):
    cluster_ids = sorted({r.get("context_cluster", r["id"]) for r in rows})
    if not cluster_ids:
        return None

    grouped = {
        cid: [
            r for r in rows
            if r.get("context_cluster", r["id"]) == cid
            and sgn(r["answer_nll_gain"]) != 0
        ]
        for cid in cluster_ids
    }
    cluster_ids = [cid for cid in cluster_ids if grouped[cid]]
    if not cluster_ids:
        return None

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        boot_rows = []
        for cid in sampled:
            boot_rows.extend(grouped[str(cid)])
        a, _ = sign_accuracy(boot_rows, a_key)
        b, _ = sign_accuracy(boot_rows, b_key)
        diffs.append(a - b)

    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "mean": float(np.mean(diffs)),
        "ci95": [float(lo), float(hi)],
    }


def load_results(raw_dir: Path):
    payloads = []
    for path in sorted(raw_dir.glob("*.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def flatten_pairs(payloads):
    rows = []
    for p in payloads:
        lu = next(x for x in p["conditions"] if x["name"] == "lu_baseline")
        candidates = [
            x for x in p["conditions"]
            if x["name"] not in {"fullkv", "lu_baseline"}
        ]

        for cand in candidates:
            for probe_len in p["probe_lens"]:
                key = str(probe_len)
                probe = cand["probes"][key]
                rows.append(
                    {
                        "id": p["id"],
                        "context_cluster": p.get("context_sha256") or p["id"],
                        "family": p["family"],
                        "task": p["task"],
                        "candidate": cand["name"],
                        "category": cand["category"],
                        "probe_len": int(probe_len),
                        "lu_delta": float(cand["lu_marginal_delta"]),
                        "answer_nll_gain": float(cand["answer_nll_gain_vs_lu"]),
                        "task_score_gain": cand.get("task_score_gain_vs_lu"),
                        "delta_overlap": float(probe["delta_overlap_vs_lu"]),
                        "kl_gain": float(probe["kl_gain_vs_lu"]),
                        "js_gain": float(probe["js_gain_vs_lu"]),
                        "top10_gain": float(probe["top10_gain_vs_lu"]),
                        "probe_nll_gain": float(probe["probe_nll_gain_vs_lu"]),
                        "lu_answer_nll": float(lu["answer_nll"]),
                        "candidate_answer_nll": float(cand["answer_nll"]),
                    }
                )
    return rows


def subset_summary(rows, bootstrap, seed):
    out = {
        "pairs": len(rows),
        "prompts": len({r["id"] for r in rows}),
    }
    predictors = [
        "delta_overlap",
        "kl_gain",
        "js_gain",
        "top10_gain",
        "probe_nll_gain",
        "lu_delta",
    ]
    for key in predictors:
        acc, n = sign_accuracy(rows, key)
        out[f"{key}_sign_accuracy"] = acc
        out[f"{key}_sign_n"] = n

    out["overlap_minus_lu_sign_accuracy"] = (
        None
        if out["delta_overlap_sign_accuracy"] is None
        or out["lu_delta_sign_accuracy"] is None
        else out["delta_overlap_sign_accuracy"] - out["lu_delta_sign_accuracy"]
    )
    out["overlap_minus_lu_bootstrap"] = bootstrap_prompt_difference(
        rows,
        "delta_overlap",
        "lu_delta",
        n_boot=bootstrap,
        seed=seed,
    )
    return out


def selection_summary(payloads, probe_len):
    per_prompt = []

    for p in payloads:
        fullkv = next(x for x in p["conditions"] if x["name"] == "fullkv")
        lu = next(x for x in p["conditions"] if x["name"] == "lu_baseline")
        candidates = [
            x for x in p["conditions"]
            if x["name"] not in {"fullkv", "lu_baseline"}
        ]

        probe_key = str(probe_len)
        best_behavior = max(
            candidates,
            key=lambda x: x["probes"][probe_key]["delta_overlap_vs_lu"],
        )
        best_behavior_delta = best_behavior["probes"][probe_key]["delta_overlap_vs_lu"]

        if best_behavior_delta > 0:
            selected = best_behavior
            fallback = False
            selected_gain = float(selected["answer_nll_gain_vs_lu"])
        else:
            selected = lu
            fallback = True
            selected_gain = 0.0

        candidate_gains = [float(x["answer_nll_gain_vs_lu"]) for x in candidates]
        local_oracle_gain = max([0.0] + candidate_gains)

        if local_oracle_gain > 0:
            local_recovery = selected_gain / local_oracle_gain
        else:
            local_recovery = None

        fullkv_gap = float(lu["answer_nll"]) - float(fullkv["answer_nll"])
        fullkv_recovery = selected_gain / fullkv_gap if fullkv_gap > 0 else None

        per_prompt.append(
            {
                "id": p["id"],
                "family": p["family"],
                "task": p["task"],
                "selected": selected["name"],
                "fallback": fallback,
                "selected_overlap_delta": float(best_behavior_delta),
                "selected_answer_nll_gain": selected_gain,
                "local_oracle_answer_nll_gain": local_oracle_gain,
                "local_oracle_recovery": local_recovery,
                "fullkv_gap_recovery": fullkv_recovery,
                "fullkv_task_score": fullkv.get("task_score"),
                "lu_task_score": lu.get("task_score"),
                "selected_task_score": selected.get("task_score"),
            }
        )

    def finite_mean(values):
        vals = [float(v) for v in values if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    return {
        "probe_len": int(probe_len),
        "prompts": len(per_prompt),
        "fallback_rate": (
            sum(x["fallback"] for x in per_prompt) / len(per_prompt)
            if per_prompt else None
        ),
        "prompts_with_beneficial_swap_rate": (
            sum(x["local_oracle_answer_nll_gain"] > 0 for x in per_prompt)
            / len(per_prompt)
            if per_prompt else None
        ),
        "selected_improvement_rate": (
            sum(x["selected_answer_nll_gain"] > 0 for x in per_prompt)
            / len(per_prompt)
            if per_prompt else None
        ),
        "selected_harm_rate": (
            sum(x["selected_answer_nll_gain"] < 0 for x in per_prompt)
            / len(per_prompt)
            if per_prompt else None
        ),
        "mean_selected_answer_nll_gain": finite_mean(
            [x["selected_answer_nll_gain"] for x in per_prompt]
        ),
        "mean_local_oracle_recovery": finite_mean(
            [x["local_oracle_recovery"] for x in per_prompt]
        ),
        "mean_fullkv_gap_recovery": finite_mean(
            [x["fullkv_gap_recovery"] for x in per_prompt]
        ),
        "mean_fullkv_task_score": finite_mean(
            [x["fullkv_task_score"] for x in per_prompt]
        ),
        "mean_lu_task_score": finite_mean(
            [x["lu_task_score"] for x in per_prompt]
        ),
        "mean_selected_task_score": finite_mean(
            [x["selected_task_score"] for x in per_prompt]
        ),
        "per_prompt": per_prompt,
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    payloads = load_results(args.raw_dir)
    if not payloads:
        raise RuntimeError(f"no MiniGate JSON files found in {args.raw_dir}")

    pairs = flatten_pairs(payloads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "pairs.csv", pairs)

    probe_lens = sorted({r["probe_len"] for r in pairs})
    summary = {
        "num_prompt_files": len(payloads),
        "num_candidate_probe_pairs": len(pairs),
        "probe_lens": probe_lens,
        "by_probe": {},
        "selection": {},
    }

    for probe_len in probe_lens:
        probe_rows = [r for r in pairs if r["probe_len"] == probe_len]
        by_family = {
            "all": subset_summary(probe_rows, args.bootstrap, args.seed),
        }
        for family in sorted({r["family"] for r in probe_rows}):
            family_rows = [r for r in probe_rows if r["family"] == family]
            by_family[family] = subset_summary(
                family_rows,
                args.bootstrap,
                args.seed,
            )
        summary["by_probe"][str(probe_len)] = by_family
        summary["selection"][str(probe_len)] = selection_summary(
            payloads,
            probe_len,
        )

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
