"""Zero-GPU diagnostic for the intersection of legacy Gate 0A actions.

This is intentionally NOT a formal Gate 0A pass when the original action
matrix is incomplete. It asks a narrower question: among actions that were
measured for every prompt, is there already evidence that prompt-local oracle
choices beat leakage-free static choices?
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from evaluation.sketchalloc.analyze_gate0a import evaluate_gate0a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260928)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    frame["action_id"] = frame["action_id"].astype(str)

    num_prompts = int(frame["prompt_id"].nunique())
    counts = frame.groupby("action_id")["prompt_id"].nunique().sort_index()
    common_actions = list(counts[counts == num_prompts].index.astype(str))
    if not common_actions:
        raise ValueError("no action is shared by all prompts")

    common = frame[frame["action_id"].isin(common_actions)].copy()
    common = common.sort_values(["prompt_id", "action_id"])
    summary = evaluate_gate0a(common, n_bootstrap=args.bootstrap, seed=args.seed)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    common.to_csv(args.output_csv, index=False)

    payload = {
        "scope": "common_action_intersection_diagnostic_only",
        "formal_gate0a_pass_allowed": False,
        "reason": (
            "The legacy full action matrix is incomplete. This diagnostic uses "
            "only actions observed for every prompt and cannot by itself satisfy "
            "the preregistered Gate 0A."
        ),
        "num_prompts": num_prompts,
        "num_common_actions": len(common_actions),
        "common_action_ids": common_actions,
        "summary": asdict(summary),
    }
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
