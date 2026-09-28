from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Summarize Gate-0-v2 answer-onset discovery outputs."
    )
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/gate0/answer_onset/discovery24_summary.json"),
    )
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260928)
    return p.parse_args()


def sgn(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def load_payloads(raw_dir: Path):
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(raw_dir.glob("*.json"))
    ]


def flatten(payloads):
    rows = []
    for p in payloads:
        for cand in p["conditions"]:
            if cand["name"] in {"fullkv", "lu_baseline"}:
                continue
            for length in p["trace_lens"]:
                m = cand["answer_trace"][str(length)]
                rows.append(
                    {
                        "id": p["id"],
                        "context_cluster": p.get("context_sha256") or p["id"],
                        "family": p["family"],
                        "task": p["task"],
                        "candidate": cand["name"],
                        "category": cand["category"],
                        "trace_len": int(length),
                        "answer_nll_gain": float(cand["answer_nll_gain_vs_lu"]),
                        "lu_delta": float(cand["lu_marginal_delta"]),
                        "delta_overlap": float(m["delta_overlap_vs_lu"]),
                        "kl_gain": float(m["kl_gain_vs_lu"]),
                        "js_gain": float(m["js_gain_vs_lu"]),
                        "top10_gain": float(m["top10_gain_vs_lu"]),
                        "pseudo_nll_gain": float(m["pseudo_nll_gain_vs_lu"]),
                    }
                )
    return rows


def sign_accuracy(rows, key):
    valid = [r for r in rows if sgn(r["answer_nll_gain"]) != 0]
    if not valid:
        return None
    return sum(
        sgn(r[key]) == sgn(r["answer_nll_gain"])
        for r in valid
    ) / len(valid)


def bootstrap_diff(rows, a_key, b_key, n_boot, seed):
    clusters = sorted({r["context_cluster"] for r in rows})
    grouped = {
        cid: [
            r for r in rows
            if r["context_cluster"] == cid and sgn(r["answer_nll_gain"]) != 0
        ]
        for cid in clusters
    }
    clusters = [c for c in clusters if grouped[c]]
    if not clusters:
        return None

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot = []
        for cid in sampled:
            boot.extend(grouped[str(cid)])
        diffs.append(sign_accuracy(boot, a_key) - sign_accuracy(boot, b_key))

    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "mean": float(np.mean(diffs)),
        "ci95": [float(lo), float(hi)],
    }


def subset_summary(rows, n_boot, seed):
    keys = [
        "delta_overlap",
        "kl_gain",
        "js_gain",
        "top10_gain",
        "pseudo_nll_gain",
        "lu_delta",
    ]
    out = {
        "pairs": len(rows),
        "prompts": len({r["id"] for r in rows}),
    }
    for key in keys:
        out[f"{key}_sign_accuracy"] = sign_accuracy(rows, key)

    out["overlap_minus_lu"] = (
        None
        if out["delta_overlap_sign_accuracy"] is None
        else out["delta_overlap_sign_accuracy"] - out["lu_delta_sign_accuracy"]
    )
    out["overlap_minus_lu_bootstrap"] = bootstrap_diff(
        rows,
        "delta_overlap",
        "lu_delta",
        n_boot,
        seed,
    )
    return out


def selection_summary(payloads, trace_len):
    rows = []
    for p in payloads:
        lu = next(x for x in p["conditions"] if x["name"] == "lu_baseline")
        candidates = [
            x for x in p["conditions"]
            if x["name"] not in {"fullkv", "lu_baseline"}
        ]
        key = str(trace_len)

        best = max(
            candidates,
            key=lambda x: x["answer_trace"][key]["delta_overlap_vs_lu"],
        )
        best_signal = float(best["answer_trace"][key]["delta_overlap_vs_lu"])

        if best_signal > 0:
            selected = best
            selected_gain = float(selected["answer_nll_gain_vs_lu"])
            fallback = False
        else:
            selected = lu
            selected_gain = 0.0
            fallback = True

        oracle_gain = max(
            [0.0] + [float(x["answer_nll_gain_vs_lu"]) for x in candidates]
        )

        rows.append(
            {
                "id": p["id"],
                "family": p["family"],
                "selected": selected["name"],
                "fallback": fallback,
                "selected_answer_nll_gain": selected_gain,
                "local_oracle_answer_nll_gain": oracle_gain,
            }
        )

    n = len(rows)
    return {
        "prompts": n,
        "fallback_rate": sum(r["fallback"] for r in rows) / n,
        "prompts_with_beneficial_swap_rate": (
            sum(r["local_oracle_answer_nll_gain"] > 0 for r in rows) / n
        ),
        "selected_improvement_rate": (
            sum(r["selected_answer_nll_gain"] > 0 for r in rows) / n
        ),
        "selected_harm_rate": (
            sum(r["selected_answer_nll_gain"] < 0 for r in rows) / n
        ),
        "mean_selected_answer_nll_gain": float(
            np.mean([r["selected_answer_nll_gain"] for r in rows])
        ),
        "mean_local_oracle_answer_nll_gain": float(
            np.mean([r["local_oracle_answer_nll_gain"] for r in rows])
        ),
        "per_prompt": rows,
    }


def main():
    args = parse_args()
    payloads = load_payloads(args.raw_dir)
    if not payloads:
        raise RuntimeError(f"no answer-onset JSON files in {args.raw_dir}")

    flat = flatten(payloads)
    trace_lens = sorted({r["trace_len"] for r in flat})

    summary = {
        "protocol": "Gate-0-v2 answer-onset discovery",
        "num_prompts": len(payloads),
        "trace_lens": trace_lens,
        "by_trace": {},
        "selection_by_overlap": {},
    }

    for length in trace_lens:
        rows = [r for r in flat if r["trace_len"] == length]
        block = {
            "all": subset_summary(rows, args.bootstrap, args.seed),
        }
        for family in sorted({r["family"] for r in rows}):
            block[family] = subset_summary(
                [r for r in rows if r["family"] == family],
                args.bootstrap,
                args.seed,
            )
        summary["by_trace"][str(length)] = block
        summary["selection_by_overlap"][str(length)] = selection_summary(
            payloads,
            length,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
