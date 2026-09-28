# SketchAlloc-KV

## Gate 0A quick start

The first gate audits whether the legacy CausalDuel interventions form a
shared prompt-by-action matrix and whether prompt-local oracle actions really
beat a leakage-free fixed action.

```bash
python -m evaluation.sketchalloc.normalize_legacy_results \
  --raw-dir results/legacy_causalduel/gate0/answer_onset/discovery24 \
  --output-csv results/sketchalloc/gate0a/legacy24_actions.csv \
  --audit-json results/sketchalloc/gate0a/legacy24_coverage.json

python -m evaluation.sketchalloc.analyze_gate0a \
  --input results/sketchalloc/gate0a/legacy24_actions.csv \
  --output-json results/sketchalloc/gate0a/legacy24_gate0a.json
```

See `docs/gate0a_runbook.md` for the stop/go interpretation.  The analyzer
refuses to compare actions when the old per-prompt candidate sets do not form a
complete rectangular matrix.

Research scaffold for **sparse active identification of prompt-conditioned KV-cache allocation utility**.

Status: Gate 0 design and legacy migration. No SOTA or physical-memory claim is made yet.

## Core question

LU-KV provides a strong static long-horizon utility profile. SketchAlloc-KV asks whether the prompt-specific residual around that profile:

1. contains real headroom beyond the best fixed or task-family-static action;
2. has a low-dimensional structure on held-out prompts;
3. can be identified using only a few actively selected mechanical probes.

If any answer is no, the project stops or falls back to static profile calibration.

## Repository lineage

This repository is derived from the public KVPress/LU-KV code lineage and preserves its `LICENSE`, `CITATION.cff`, package layout, and attribution. It also imports selected engineering assets from the abandoned CausalDuel-KV Gate-0 branch.

Reusable assets include exact per-KV-head budget overrides, LU marginal export, budget-preserving swaps, MiniGate data preparation, gold-answer NLL evaluation, and prompt-clustered statistics.

The former probability-overlap / KL / answer-onset selector is not part of the new method. It is kept under `legacy/causalduel/` as negative evidence and for reproducibility.

## Documents

- `docs/idea_report.md`: Part 1 and Part 2 research design.
- `docs/gate0_sketchalloc_kv.md`: preregistered Gate 0.
- `docs/collision_check_2026-09-28.md`: novelty collision check.
- `docs/legacy_reuse_manifest.md`: exact keep/archive/rewrite decisions.

## First executable check

Normalize the legacy prompt-action results into a CSV with columns:

```text
prompt_id,action_id,gain,split
```

where `gain = NLL_LU - NLL_action` and `split` is `train` or `test`. Then run:

```bash
python -m evaluation.sketchalloc.analyze_utility_rank \
  --input results/sketchalloc/utility_matrix.csv \
  --rank 3 \
  --num-sentinels 6 \
  --output results/sketchalloc/rank_gate0.json
```

This command is only a structural upper bound: it assumes the true utility of the selected sentinel actions is observable. Passing it does not prove that an online no-answer probe exists.

## Current limitations

- Server result artifacts have not yet been migrated into this local repository.
- The inherited LU path uses logical masking; physical KV-page savings remain a separate Gate -1.
- The active local probe extractor is not implemented until the structural gates pass.
- GitHub private remote creation is pending because no authenticated GitHub connector/CLI is available in the current environment.

## Tests

```bash
pytest tests/test_sketchalloc_rank.py tests/test_gate0_candidates.py tests/test_gate0_marginals.py tests/test_gate0_metrics.py
```

