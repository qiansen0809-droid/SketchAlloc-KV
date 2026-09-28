# Gate 0B half-scale result (2026-09-28)

## Decision

**NO-GO for expansion to the formal 96 x 24 Gate 0B under the current low-rank hypothesis.**

This is a half-scale structural diagnostic, not the formal Gate 0B itself. The
pre-frozen rule was to expand only if all four primary family-centered held-out
checks passed. Only the rank cap passed.

## Frozen setup

- 48 fresh prompts
- 12 fixed shared actions
- 24 discovery / 12 calibration / 12 held-out
- 16 prompts per broad family
- 20% retention / 80% compression
- fixed 16-token budget-preserving swaps
- actions and data frozen before Gate 0B Answer-NLL labels
- primary object: family-centered utility residual
- calibration chooses rank from ranks <= 4
- held-out row coordinates use the true held-out residual projection, so this
  is an optimistic structural upper bound rather than an online selector

## Primary result

Calibration selected rank 4.

Held-out family-centered rank-4 metrics:

```text
variance explained                0.5803559561   FAIL (< 0.70)
median within-prompt Spearman     0.4965034965   FAIL (< 0.60)
mean top-action regret            0.0096064111
oracle-over-best-fixed headroom   0.0184544424
regret / headroom                 0.5205473479   FAIL (> 0.30)
```

Primary half-scale checks:

```text
selected_rank_le_4                         PASS
heldout_variance_explained_ge_0_70         FAIL
heldout_median_spearman_ge_0_60            FAIL
heldout_regret_fraction_le_0_30             FAIL
```

Therefore the preregistered half-scale recommendation is:

```text
do_not_expand_yet_reinspect_low_rank_hypothesis
```

## Important secondary observations

The discovery family-centered spectrum looks compact in-sample:

```text
rank 1 cumulative variance   0.4417
rank 2                       0.6699
rank 3                       0.7636
rank 4                       0.8498
```

However, that discovery subspace does not transfer strongly enough to held-out
prompts at rank <= 4.

Higher ranks improve held-out reconstruction:

```text
rank 5: R2 0.6360, Spearman 0.6608, regret/headroom 0.4812
rank 6: R2 0.7270, Spearman 0.7413, regret/headroom 0.4152
rank 7: R2 0.7685, Spearman 0.7517, regret/headroom 0.2225
rank 8: R2 0.8259, Spearman 0.7832, regret/headroom 0.1524
```

So the utility surface is not structureless. But useful held-out action
reconstruction appears to require about 7-8 dimensions for this 12-action
dictionary. That is not the low-rank <=4 structure preregistered for
SketchAlloc, and it weakens the later <=6-sentinel design.

Permutation parallel analysis finds only two discovery singular components
above the 95th-percentile shuffled null. The rank-4 bootstrap subspace affinity
is moderately high (mean 0.829, 95% interval 0.687-0.962), but bootstrap
stability within discovery does not rescue the weaker held-out transfer.

## Interpretation

Allowed conclusion:

> Prompt-conditioned action utility exists (Gate 0A), but in this half-scale
> experiment its family-centered residual is not sufficiently stable and
> low-dimensional at rank <= 4 to justify the current SketchAlloc low-rank
> reconstruction hypothesis.

Do not conclude that utility is pure noise: higher-rank held-out projections
are informative. The failure is specifically the **small, stable latent
dimension** required by the current method.

## Next step

Do not run the formal 96 x 24 Gate 0B and do not start Gate 0C under the current
method.

Use only the already-computed 48 x 12 matrix for a zero-GPU forensic analysis:
per-family held-out reconstruction, per-action variance/noise, singular-vector
alignment across splits, and influence/outlier diagnostics. This analysis may
support a reframe (for example, static/family calibration or a different
structured model), but it must not retroactively change the Gate 0B thresholds
or convert this result into a pass.
