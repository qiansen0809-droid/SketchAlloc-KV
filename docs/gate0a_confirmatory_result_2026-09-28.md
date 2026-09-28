# Gate 0A confirmatory result (2026-09-28)

## Decision

**GO to Gate 0B.**

This decision applies to the preregistered Gate 0A question: whether prompt-specific
KV-allocation utility exists beyond a leakage-aware fixed action or a task-family-static
action. It is not evidence that the full SketchAlloc method works.

## Confirmatory setup

- 24 fresh prompts, disjoint from the legacy discovery prompts/contexts
- 3 broad families: RULER retrieval, LongBench single-document QA, LongBench multi-document QA
- 5 actions frozen before confirmatory Answer-NLL labels
- LU fallback always available
- fixed 20% retention / 80% compression
- fixed 16-token donor->receiver transfer
- complete 24 x 5 prompt-action matrix
- no prompt-specific candidate generation from confirmatory labels

The single-document family contains 4 Qasper + 4 MultiFieldQA-en because no fresh
NarrativeQA context remained inside the frozen 6144-9216 token window after leakage
exclusion. This sampling amendment was frozen before confirmatory labels.

## Confirmatory statistics

```text
oracle_mean_gain                     0.0191889082
random_action_mean_gain              0.0030829350
lopo_best_fixed_mean_gain           -0.0049787710
lopo_family_static_mean_gain         0.0083121757
lofo_best_fixed_mean_gain           -0.0001274198

oracle_minus_best_fixed_mean         0.0241676793
oracle_minus_best_fixed_ci95        [0.0147506842, 0.0352965066]

oracle_minus_family_static_mean      0.0108767326
oracle_minus_family_static_ci95     [0.0058203693, 0.0165357720]

best_fixed_headroom_recovery        -0.2594608796
family_static_headroom_recovery      0.4331760602

pairwise_action_crossover_rate       1.0000000000
prompt_crossover_participation       1.0000000000
oracle_action_entropy_normalized     0.9454524971
```

Oracle action counts:

```text
LU                         2
L0H6->L25H0@16             6
L17H6->L16H6@16            5
L18H5->L24H0@16            3
L22H2->L5H2@16             6
L27H7->L25H0@16            2
```

## Gate 0A criteria

- oracle - best-fixed bootstrap 95% CI lower bound > 0: **PASS**
- family-static fails to recover >=70% of oracle headroom: **PASS** (43.3% recovered)
- pairwise crossover rate >= 0.30: **PASS** (1.00)
- no single oracle action dominates; high normalized entropy: **supporting evidence**
- leave-one-family-out fixed policy has approximately zero aggregate gain: **supporting evidence**
- drop-one-family direction remains positive for every omitted family: **PASS**
  - drop LongBench multi: oracle - LOPO best-fixed = 0.0189563893
  - drop LongBench single: oracle - LOPO best-fixed = 0.0177398138
  - drop RULER retrieval: oracle - LOPO best-fixed = 0.0179192573

The confirmatory result reproduces the qualitative discovery pattern: a prompt-local
oracle has positive headroom, while leakage-aware fixed choices do not explain it and
action rankings cross over strongly across prompts. The direction is unchanged after
dropping any one of the three broad task families, completing the preregistered Gate 0A
sensitivity check.

## Interpretation boundary

Allowed conclusion:

> There is confirmatory evidence, in this Gate 0 setting, that the utility of small
> budget-preserving LU-neighborhood actions is prompt-dependent and cannot be explained
> by a single fixed action or the tested family-static rule.

Not allowed yet:

- SketchAlloc sparse probes work
- utility residuals are low rank
- the effect generalizes to other models, lengths, or compression ratios
- physical KV memory is saved
- the method beats LU-KV on end-task metrics

Those questions belong to Gate 0B, Gate 0C, and Gate -1.

## Next step

Proceed to Gate 0B. Follow the preregistered compute-saving order:

1. freeze a shared 12-action dictionary using LU-profile information only;
2. prepare 48 fresh prompts, 16 per family, disjoint by source/context cluster;
3. run the complete 48 x 12 utility matrix at the same 8K / 20% retention setting;
4. inspect raw, action-centered, and family-centered spectra;
5. continue to the full 96 x 24 Gate 0B only if the half-scale matrix shows stable
   low-rank structure under held-out reconstruction diagnostics.
