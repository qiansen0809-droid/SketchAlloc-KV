# Gate 0B half-scale runbook

Date: 2026-09-28

Status: frozen before any Gate 0B Answer-NLL label is computed.

This runbook is a compute-saving diagnostic before the formal 96-prompt x
24-action Gate 0B in `docs/gate0_sketchalloc_kv.md`. Passing this half-scale
diagnostic is **not** a formal Gate 0B pass. It only authorizes spending compute
on the full preregistered structural gate.

## Question

Gate 0A established that local LU-neighborhood action utility is prompt
dependent. Gate 0B asks a different question:

> After subtracting action-static and task-family-static structure, does the
> prompt-specific utility residual lie close to a low-dimensional subspace?

The utility cell is fixed as:

```text
U[prompt, action] = NLL_LU - NLL_action
```

Positive values mean the action improves Answer-NLL relative to LU.

## Half-scale dataset

Use 48 fresh prompts:

- 16 RULER retrieval
- 16 LongBench single-document QA
- 16 LongBench multi-document QA

Split:

- discovery: 24 total, 8 per broad family
- calibration: 12 total, 4 per broad family
- held-out: 12 total, 4 per broad family

All legacy discovery24 and Gate 0A confirm24 prompt IDs, source rows, and
context hashes are excluded.

RULER task/template families are split-disjoint: 4 retrieval tasks belong only
to discovery, 2 only to calibration, and 2 only to held-out. LongBench natural
contexts are unique across splits.

The 6144-9216-token LongBench window and approximately 8K target are unchanged.

## Frozen 12-action dictionary

The half-scale dictionary is built only after the fresh 48-prompt manifest is
created and before any Gate 0B model label is computed.

Every action must be legal for every one of the 48 prompt lengths. Selection
uses only:

- LU static budget curve
- LU exported remove-cost / next-gain marginal profile
- prompt token length for legality

It must not use Gate 0B Answer-NLL or task scores.

Fixed mix:

- 4 strongest LU-promising shared-legal actions
- 3 closest non-negative LU-boundary actions
- 3 closest negative LU-boundary actions
- 2 most LU-unfavored shared-legal controls

A soft donor/receiver use cap of 3 encourages action diversity. If a category
cannot be filled under that cap, it is deterministically relaxed without
changing category counts.

## Matrix execution

Run one LU baseline plus the same 12 frozen actions for every prompt.

Expected intervention matrix:

```text
48 prompts x 12 actions = 576 utility cells
```

LU adds 48 baseline evaluations, for 624 scored conditions total.

The runner must pass a full 48 x 12 legality audit before loading the model and
must clear LU keep-count overrides between prompts.

## Centering modes

Analyze all three:

1. raw utility `U`
2. action-centered residual: subtract discovery action mean
3. family-centered residual: subtract discovery family x action mean

Only family-centered held-out structure counts toward the primary half-scale
decision. Raw/action-centered structure is diagnostic because it can be
explained by static action quality or broad task-family identity.

## Rank fitting and selection

Fit right-singular-vector bases on discovery only for ranks 1-8.

Calibration chooses the rank without looking at held-out labels:

- consider ranks <= 4
- find the best calibration residual R2 in that range
- select the **smallest** rank reaching at least 95% of that best calibration R2
- if every calibration R2 is non-positive, choose the rank with the best
  calibration R2, using Spearman/regret only as deterministic tie-breaks

Held-out row coordinates are then obtained by projecting the **true held-out
residual** onto the discovery basis. This is intentionally optimistic and is
only a structural upper bound. It is not an online method.

## Primary half-scale expansion checks

Expand to the formal 96 x 24 Gate 0B only if the selected family-centered rank
satisfies all four:

- selected rank <= 4
- held-out family-centered variance explained >= 70%
- held-out median within-prompt Spearman >= 0.60
- held-out oracle-coordinate top-action regret <= 30% of
  oracle-over-best-fixed headroom

Also inspect, but do not retune against, two supporting diagnostics:

- permutation parallel analysis of discovery singular values
- prompt-bootstrap right-singular-vector subspace stability

A weak parallel-analysis signal or unstable bootstrap basis should be treated
as a warning even if the four primary checks narrowly pass.

If the four primary checks fail, do not increase predictor complexity or start
Gate 0C. Reinspect or stop the low-rank hypothesis.

## Interpretation boundary

A half-scale pass means only:

> A larger formal Gate 0B is worth running.

It does not show:

- sparse sentinels can recover latent coordinates
- an online no-answer selector exists
- SketchAlloc beats LU-KV on end-task metrics
- real physical KV memory is saved

Those remain Gate 0C and Gate -1 questions.


## Frozen half48 sampling realization

Before any Gate 0B Answer-NLL label was computed, the label-free sampler
resolved to the following concrete composition after excluding all legacy
discovery24 and Gate 0A confirm24 prompt/source/context overlap:

- RULER retrieval: 16 prompts, with task/template families split-disjoint
  - discovery: niah_single_1, niah_single_2, niah_single_3, niah_multikey_1
  - calibration: niah_multikey_2, niah_multikey_3
  - heldout: niah_multivalue, niah_multiquery
  - two fresh prompts per RULER task
- LongBench single-document QA: 8 Qasper + 8 MultiFieldQA-en
  - per split: discovery 4+4, calibration 2+2, heldout 2+2
  - NarrativeQA had insufficient fresh in-window support after exclusions
- LongBench multi-document QA: 8 HotpotQA + 8 2WikiMQA
  - per split: discovery 4+4, calibration 2+2, heldout 2+2
  - MuSiQue had insufficient fresh in-window support after exclusions

Overall:

- 48 prompts total
- 16 prompts per broad family
- 24 discovery / 12 calibration / 12 heldout
- each broad family contributes 8 / 4 / 4 to those splits

This realized composition is frozen before Gate 0B model execution. It narrows
the LongBench task diversity relative to the formal 96 x 24 plan, so a
half-scale positive result authorizes expansion only; it is not evidence of
task-wide generalization.
