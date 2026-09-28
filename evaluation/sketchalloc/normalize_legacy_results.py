"""Normalize CausalDuel answer-onset JSON files for SketchAlloc Gate 0A.

The old experiment generated candidates independently for every prompt.  The
candidate names therefore cannot be assumed to denote a shared action.  This
converter reconstructs an action ID from the donor, receiver, and transferred
budget and emits an explicit coverage audit before any cross-prompt comparison
is attempted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


BASELINE_NAMES = {"fullkv", "lu_baseline"}


@dataclass
class CoverageAudit:
    num_payloads: int
    num_prompts: int
    num_rows: int
    num_unique_actions: int
    num_common_actions: int
    expected_rectangular_cells: int
    observed_rectangular_cells: int
    missing_rectangular_cells: int
    complete_action_matrix: bool
    action_counts: dict[str, int]
    source_files: list[str]


def _require_int(unit: dict, key: str, label: str) -> int:
    if key not in unit:
        raise ValueError(f"{label} is missing {key!r}: {unit}")
    return int(unit[key])


def stable_action_id(candidate: dict) -> str:
    """Build an action ID independent of the legacy candidate label."""

    donor = candidate.get("donor")
    receiver = candidate.get("receiver")
    if not isinstance(donor, dict) or not isinstance(receiver, dict):
        raise ValueError(f"candidate lacks donor/receiver metadata: {candidate.get('name')}")
    dl = _require_int(donor, "layer", "donor")
    dh = _require_int(donor, "head", "donor")
    rl = _require_int(receiver, "layer", "receiver")
    rh = _require_int(receiver, "head", "receiver")
    amount = int(candidate["amount"])
    return f"L{dl}H{dh}->L{rl}H{rh}@{amount}"


def normalize_payload(payload: dict, source_file: str = "") -> list[dict]:
    prompt_id = str(payload["id"])
    conditions = payload.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError(f"payload {prompt_id} has no conditions list")

    lu_rows = [row for row in conditions if row.get("name") == "lu_baseline"]
    if len(lu_rows) != 1:
        raise ValueError(f"payload {prompt_id} must contain exactly one lu_baseline")
    lu_nll = float(lu_rows[0]["answer_nll"])

    rows: list[dict] = []
    for candidate in conditions:
        if candidate.get("name") in BASELINE_NAMES:
            continue
        action_nll = float(candidate["answer_nll"])
        recorded_gain = candidate.get("answer_nll_gain_vs_lu")
        derived_gain = lu_nll - action_nll
        if recorded_gain is not None and abs(float(recorded_gain) - derived_gain) > 1e-6:
            raise ValueError(
                f"payload {prompt_id}, candidate {candidate.get('name')}: "
                "recorded and derived answer-NLL gains disagree"
            )
        donor = candidate["donor"]
        receiver = candidate["receiver"]
        rows.append(
            {
                "prompt_id": prompt_id,
                "context_cluster": str(payload.get("context_sha256") or prompt_id),
                "family": str(payload.get("family", "unknown")),
                "task": str(payload.get("task", "unknown")),
                "action_id": stable_action_id(candidate),
                "legacy_candidate_name": str(candidate.get("name", "")),
                "category": str(candidate.get("category", "unknown")),
                "donor_layer": int(donor["layer"]),
                "donor_head": int(donor["head"]),
                "receiver_layer": int(receiver["layer"]),
                "receiver_head": int(receiver["head"]),
                "amount": int(candidate["amount"]),
                "lu_remove_cost": float(candidate.get("lu_remove_cost", float("nan"))),
                "lu_next_gain": float(candidate.get("lu_next_gain", float("nan"))),
                "lu_marginal_delta": float(candidate.get("lu_marginal_delta", float("nan"))),
                "lu_answer_nll": lu_nll,
                "action_answer_nll": action_nll,
                "gain": derived_gain,
                "context_tokens": int(payload.get("context_tokens", -1)),
                "compression_ratio": float(payload.get("compression_ratio", float("nan"))),
                "source_file": source_file,
            }
        )
    if not rows:
        raise ValueError(f"payload {prompt_id} contains no intervention candidates")
    return rows


def audit_coverage(frame: pd.DataFrame, source_files: list[str]) -> CoverageAudit:
    duplicated = frame.duplicated(["prompt_id", "action_id"], keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, ["prompt_id", "action_id"]].head().to_dict("records")
        raise ValueError(f"duplicate prompt/action rows after normalization: {examples}")

    counts = frame.groupby("action_id")["prompt_id"].nunique().sort_index()
    num_prompts = int(frame["prompt_id"].nunique())
    num_actions = int(len(counts))
    common = int((counts == num_prompts).sum())
    expected = num_prompts * num_actions
    observed = int(len(frame))
    return CoverageAudit(
        num_payloads=len(source_files),
        num_prompts=num_prompts,
        num_rows=observed,
        num_unique_actions=num_actions,
        num_common_actions=common,
        expected_rectangular_cells=expected,
        observed_rectangular_cells=observed,
        missing_rectangular_cells=expected - observed,
        complete_action_matrix=bool(expected == observed and common == num_actions),
        action_counts={str(k): int(v) for k, v in counts.items()},
        source_files=source_files,
    )


def load_payload_files(raw_dir: Path) -> tuple[list[dict], list[str]]:
    payloads: list[dict] = []
    source_files: list[str] = []
    for path in sorted(raw_dir.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or "conditions" not in value or "id" not in value:
            continue
        payloads.append(value)
        source_files.append(str(path))
    if not payloads:
        raise ValueError(f"no answer-onset payload JSON files found under {raw_dir}")
    return payloads, source_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads, source_files = load_payload_files(args.raw_dir)
    rows: list[dict] = []
    for payload, source_file in zip(payloads, source_files):
        rows.extend(normalize_payload(payload, source_file))
    frame = pd.DataFrame(rows).sort_values(["prompt_id", "action_id"])
    audit = audit_coverage(frame, source_files)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    args.audit_json.write_text(
        json.dumps(asdict(audit), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(asdict(audit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
