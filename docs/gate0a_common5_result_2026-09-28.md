# Gate 0A common-action diagnostic result (2026-09-28)

## Status

**Screening signal: GO to an independent fixed-action replication.**

This is **not** a formal Gate 0A pass. The legacy 24-prompt experiment used
10 unique actions in total and only 5 actions were shared by all 24 prompts.
The full legacy prompt-by-action matrix is therefore incomplete. The analysis
below uses only the 5-action intersection.

## Data

- prompts: 24
- task families: 3
- common actions: 5
- complete matrix on common-action intersection: 24 x 5
- missing cells in the common-action intersection: 0

Common actions:

```text
L0H6->L25H0@16
L17H6->L16H6@16
L18H5->L24H0@16
L22H2->L5H2@16
L27H7->L25H0@16
```

## Screening statistics

```text
oracle_mean_gain                     0.0129357030
lopo_best_fixed_mean_gain            0.0000000000
lopo_family_static_mean_gain        -0.0070725456
lofo_best_fixed_mean_gain            0.0000000000

oracle_minus_best_fixed_mean         0.0129357030
oracle_minus_best_fixed_ci95        [0.0070285073, 0.0198393946]

oracle_minus_family_static_mean      0.0200082486
oracle_minus_family_static_ci95     [0.0102251482, 0.0313223009]

best_fixed_headroom_recovery         0.0000000000
family_static_headroom_recovery     -0.5467461326

pairwise_action_crossover_rate       1.0000000000
prompt_crossover_participation       1.0000000000
oracle_action_entropy_normalized     0.9438427789
```

Oracle action counts:

```text
LU                         6
L0H6->L25H0@16             5
L17H6->L16H6@16            4
L18H5->L24H0@16            1
L22H2->L5H2@16             5
L27H7->L25H0@16            3
```

## Interpretation

Within the five actions that were actually measured for every prompt, the
prompt-local oracle has positive headroom over a leakage-aware fixed policy,
and the cluster-bootstrap 95% interval for oracle minus LOPO best-fixed is
strictly above zero.

All pairwise action pairs exhibit at least one ordering reversal somewhere in
the 24-prompt set, and every prompt participates in at least one crossover.
The oracle choice is also spread across LU and multiple actions rather than
being dominated by one universal intervention.

These results are strong screening evidence that the old headroom is not
explained by a single fixed action or a simple family-static rule.

However, this remains discovery-only evidence because:

1. the original 24x6 legacy matrix is not rectangular;
2. the five-action intersection was identified after inspecting legacy
   coverage;
3. the 24 prompts are old exploratory data rather than a fresh confirmatory
   split;
4. only three task families are represented.

## Next preregistered action

Freeze the shared action dictionary before looking at any new labels and run an
independent replication on fresh prompts. The safest minimal replication is the
five universally observed actions above plus LU fallback. A sixth action may
only be added if it is selected by a label-free LU-profile/legality rule and
is verified legal on every replication prompt before answer-NLL labels are
computed.

The replication should preserve document/context clusters and use the same
model, scorer, 20% retention, 16-token transfer size, and Answer-NLL definition.
The primary confirmatory statistics remain:

- oracle minus leakage-aware best-fixed cluster-bootstrap 95% CI;
- oracle minus family-static;
- pairwise crossover rate;
- leave-one-family-out direction;
- oracle action entropy and action counts.

Do not start Gate 0B until this independent fixed-action replication is
complete.


## Confirmatory sampling amendment frozen before replication labels

The fresh confirmatory sampler preserves the 6144-9216 LongBench context-token
window and excludes all legacy prompt IDs, source rows, and context hashes.

After applying those exclusions, NarrativeQA has zero fresh distinct contexts
inside the frozen token window. The confirmatory single-document family is
therefore sampled as 4 Qasper + 4 MultiFieldQA-en rather than widening the
window or reusing legacy contexts. The multi-document family remains balanced
at 3 HotpotQA + 3 2WikiMQA + 2 MuSiQue. RULER retains one prompt from each of
the eight retrieval subtasks.

This amendment is based only on fresh-sample availability and token length. It
was frozen before running the confirmatory model or observing any confirmatory
Answer-NLL labels. The confirmatory claim must therefore be phrased at the
three-family level; it does not establish a NarrativeQA-specific replication.
