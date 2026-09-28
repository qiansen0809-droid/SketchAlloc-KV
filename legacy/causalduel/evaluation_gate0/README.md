# CausalDuel-KV Gate 0 POC

This branch implements the first feasibility check for the pre-registered CausalDuel-KV idea on top of the official LU-KV codebase.

## Goal

Before building cache-fork, page sharing, successive racing, or serving integration, test one narrow question:

> Starting from the LU-KV allocation, does a small budget-preserving donor/receiver swap that improves prompt-tail behavior also improve an offline answer-related objective?

The branch intentionally separates:

- **Gate 0:** scientific signal validation using LU-KV's existing logical head masking;
- **Gate -1:** physical ragged/page-cache validation and actual memory movement.

This branch does not yet claim physical cache-page savings.

## 1. Exact per-KV-head budget override

`kvpress/presses/LU_press.py` now accepts:

```python
keep_counts_override: dict[int, list[int]]
```

Mapping:

```text
layer_idx -> [keep_count_head0, keep_count_head1, ...]
```

If a layer is overridden, those exact keep counts replace LU-KV's static profile for that layer. The wrapped token scorer is unchanged, so Gate 0 varies only budget allocation.

The implementation validates:

- one count per runtime KV head;
- donor budgets cannot cross the protected sink + recent-window floor;
- receiver budgets cannot exceed prefix length;
- the current head-wise masking path uses batch size 1.

The exact counts used during prefill are exposed through `last_keep_counts`.

## 2. LU-KV boundary marginal export

The original LU-KV solver already computes scorer-aligned oracle utilities and applies convex-hull smoothing, but its repository only saves the final static allocation table.

For Gate 0, `evaluation/curve_data/step2_compute_curve.py` now has an optional export:

```text
--export_marginals
--marginal_step_tokens N
```

It performs a second offline pass and samples the convex-hull-smoothed long-horizon utility exactly at the **final averaged static LU-KV budget boundary**.

For every global compression ratio, layer, and KV head it exports:

- `remove_cost`: long-horizon oracle utility lost by removing one Gate-0 step;
- `next_gain`: long-horizon oracle utility recovered by adding one Gate-0 step;
- the matching static budget curve and metadata.

The output is an `.npz` file such as:

```text
gate0_lu_global_snapkv_sink4_win32_marginal_step16.npz
```

This export reuses LU-KV's own scorer alignment and convex-hull relaxation. It is an additional Gate-0 diagnostic; it does not modify the LU-KV solver or claim to be a new allocation algorithm.

## 3. Build the marginal profile

Use a **separate offline LU profiling dataset**. Do not build this profile from MiniGate, calibration, or held-out test prompts.

For the CausalDuel-KV Gate-0 run, the fixed public profiling source is **SQuALITY v1.3 train**:

- 6 distinct train stories;
- 5 human-written questions per story;
- 30 context/question pairs total;
- Llama-3.1 tokenizer length restricted to 4096-7168 tokens;
- deterministic story selection with seed 20260928;
- all five SQuALITY questions are retained, giving one broad plot query plus four query-focused prompts per story.

This is intentionally separate from the planned RULER/LongBench Gate-0 evaluation. It also keeps the query count at 30, matching the scale reported for LU-KV's original offline calibration, while replacing the unreleased AI-generated novel/questions with reproducible public data.

The complete preparation + marginal export can be run with:

```bash
MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct \
bash evaluation/gate0/run_squality_lu_profile.sh
```

The script writes both the exact selected profiling JSONL and a manifest containing source row IDs, token lengths, questions, seed, and selection parameters.

```bash
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B-Instruct \
DATASET_PATH=/path/to/separate_lu_profile.jsonl \
CUDA_DEVICE=0 \
SWAP_SIZE=16 \
NUM_WORKERS=8 \
bash evaluation/gate0/build_lu_marginal_profile.sh
```

The script runs LU-KV's Step 1 recorder and the modified Step 2 solver/export.

Important: the author repository references a local `novel.jsonl`, but that file is not included in the public fork. You therefore need to provide a separate profiling JSONL with the same basic fields expected by `step1_llama.py`, for example:

```json
{"context": "...long context...", "questions": ["...question..."], "task": "profile"}
```

## 4. Formal Gate-0 candidate construction

`build_candidates.py` implements the registered neighborhood policy.

### Donor pool

Among units that can legally donate one step:

```text
sort by LU remove_cost ascending
take 3-4 donors
```

### Receiver pool

Among units that can legally receive one step:

```text
sort by LU next_gain descending
take 3-4 receivers
```

### Six to eight swaps

Candidates are selected to cover:

- 2-3 LU-favored swaps with largest `next_gain - remove_cost`;
- 3-4 near-boundary swaps with smallest absolute marginal delta;
- 1 low-priority but legal control swap.

For six candidates this becomes:

```text
2 promising + 3 near-boundary + 1 control
```

For eight candidates:

```text
3 promising + 4 near-boundary + 1 control
```

Every candidate is checked for exact global budget conservation.

## 5. Prompt-tail replay

`suffix_replay.py`:

1. prefills the prefix;
2. optionally applies LU-KV / an overridden LU-KV budget;
3. teacher-forces the known prompt suffix;
4. returns suffix logits.

The same suffix is replayed for FullKV, the LU-KV baseline, and every swap candidate.

## 6. Metrics

`metrics.py` currently implements:

- full-vocabulary probability overlap;
- teacher-to-candidate KL;
- Jensen-Shannon divergence;
- top-k agreement;
- teacher-forced continuation NLL.

For the first engineering POC, suffix-token NLL is a dense sanity target.

For the registered MiniGate, the primary offline direction label must be **gold-answer NLL**. The answer is never allowed to choose the online budget; it is only an offline truth signal.

## 7. Run the formal six-swap POC

After exporting the marginal profile:

```bash
python -m evaluation.gate0.run_poc \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --budget-curve-path evaluation/curve_data/llama-3.1-8b/snapkv_maxpool_sink4_win_32_llama_avg_ratio.npy \
  --marginal-profile-path results/gate0/lu_profile/gate0_lu_global_snapkv_sink4_win32_marginal_step16.npz \
  --text-file /path/to/one_long_prompt.txt \
  --compression-ratio 0.80 \
  --suffix-len 64 \
  --swap-size 16 \
  --num-swaps 6 \
  --max-tokens 8192 \
  --dtype bfloat16 \
  --output results/gate0/poc_sample_001.json
```

`compression-ratio=0.80` follows LU-KV's convention: 80% pruning / roughly 20% retention.

If `--marginal-profile-path` is omitted, `run_poc.py` falls back to the old three-swap engineering smoke test. That fallback is not the formal Gate-0 candidate policy.

## 8. Engineering sanity checks

Before interpreting any scientific result, verify:

1. FullKV compared with itself gives overlap approximately 1.
2. LU-KV baseline is deterministic across repeated runs.
3. Every candidate has the same total keep count as LU-KV.
4. A donor loses exactly `swap-size` entries and the receiver gains exactly the same amount.
5. No donor crosses the sink/recent-window minimum.
6. All candidates use the same SnapKV token scorer.
7. Formal candidates record finite LU remove cost / next gain.
8. The marginal profile `marginal_step_tokens` exactly matches `--swap-size`.

## 9. Current limitation

The upstream LU-KV head-wise path stores the original K/V tensors and records pruned positions in `module.masked_key_indices`. During decoding, the attention patch replaces masked keys with fake keys so they receive approximately zero attention.

Therefore this branch is suitable for **Gate 0 signal validation**, but it does not establish that donor/receiver swaps move physical cache pages or reduce peak memory.

That remains the separate Gate -1 paged/ragged-cache implementation audit.

## 10. Next milestone

Once the two-sample engineering check passes:

- run 24 independent 8K MiniGate prompts;
- use six swaps per prompt;
- evaluate 32- and 64-token probes;
- add attention mass, entropy, and local reconstruction proxies;
- add gold-answer NLL as the primary dense offline label;
- compute prompt-clustered sign accuracy and within-prompt Spearman correlation.

Stop if the behavior signal does not exceed the pre-registered MiniGate threshold or does not beat the strongest cheap proxy.


## 5. MiniGate: fixed 24-prompt signal test

After the LU marginal profile and the global 6-swap candidate policy are frozen,
prepare the fixed MiniGate set before looking at any model outcomes.

The protocol contains 24 prompts:

- 8 RULER retrieval prompts: one from each of the eight NIAH retrieval subtasks
  in the public `simonjegou/ruler` 8192-token configuration;
- 8 LongBench single-document QA prompts drawn from NarrativeQA, Qasper, and
  MultiFieldQA-en;
- 8 LongBench multi-document QA prompts drawn from HotpotQA, 2WikiMQA, and
  MuSiQue.

LongBench uses the public `Xnhyacinth/LongBench` conversion created by the
LU-KV evaluation code, so context/question/answer-prefix separation matches the
upstream LU-KV inference protocol. Natural LongBench contexts are selected by Llama-3.1 tokenizer length near 8K;
benchmark contexts are not silently truncated. Within each 8-prompt LongBench
family, the sampler first reserves two distinct contexts per task, then fills
the remaining two slots with the closest unused contexts to the 8K target.
Selection depends only on task identity and tokenizer length, never on model
outputs or answer quality.

Prepare and inspect the set:

```bash
python -m evaluation.gate0.prepare_minigate \
  --model /path/to/Meta-Llama-3.1-8B-Instruct
```

The generated JSONL is local experiment data and should not be committed to the
public repository. The manifest records dataset revisions, source indices,
tasks, answers, and exact tokenizer lengths.

### MiniGate runtime semantics

For every condition, only the benchmark **context** is compressed. The question
and answer prefix are appended afterwards without another compression step,
matching the upstream LU-KV evaluation protocol.

Each prompt evaluates:

1. FullKV;
2. the frozen LU-KV baseline;
3. six same-total-budget donor->receiver swaps:
   - 2 strongly LU-favored,
   - 3 genuine near-zero LU-marginal probes with both signs when possible,
   - 1 strongly LU-unfavored control.

Gold answers are never used for candidate construction or online behavioral
selection. They are used only after the candidate has been fixed, as an offline
label through teacher-forced answer NLL and the benchmark task metric.

Behavioral probes default to 8, 16, and 32 tokens from the tail of the
question+answer-prefix. The 16-token probe is the primary Gate-0 signal; 8 and
32 are sensitivity checks. This was fixed before model outcomes were inspected,
because benchmark query metadata showed that many LongBench questions are
shorter than 32 tokens, making a 64-token probe redundant. When a query is
shorter than a requested probe, the full query is used and the actual probe
length is recorded. The expensive context prefill is shared across probe
lengths for each condition.

A one-prompt engineering pilot can be run with:

```bash
python -m evaluation.gate0.run_minigate \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --data results/gate0/minigate/minigate_24.jsonl \
  --budget-curve-path evaluation/curve_data/llama-3.1-8b/snapkv_maxpool_sink4_win_32_llama_avg_ratio.npy \
  --marginal-profile-path results/gate0/lu_profile/gate0_lu_global_snapkv_sink4_win32_marginal_step16.npz \
  --limit 1
```

After the pilot passes, remove `--limit 1` for the frozen 24-prompt run.

Summarize completed per-prompt JSON outputs with:

```bash
python -m evaluation.gate0.summarize_minigate \
  --raw-dir results/gate0/minigate/raw
```

The summary reports behavioral-vs-answer-NLL sign accuracy, LU-marginal sign
accuracy, a prompt-cluster bootstrap comparison, fallback rate, local-oracle
gap recovery, FullKV gap recovery, and task-score aggregates.


## 6. Gate-0-v2 exploratory diagnosis: answer-onset duel

The preregistered query-tail MiniGate should be analyzed as-is. If its primary
16-token query-tail overlap signal fails to improve reliably over the LU prior,
do not retune the old suffix signal on the same 24 prompts.

A separate exploratory diagnosis tests a future-facing signal at answer onset.
The FullKV teacher greedily produces a short pseudo-answer trajectory with no
gold answer. LU and every fixed candidate teacher-force exactly those same
pseudo-answer tokens. Behavioral metrics are computed over the first 1, 4, and
8 answer positions.

This changes the question from "can the compressed cache reconstruct the end of
the already-seen query?" to "does the compressed cache preserve FullKV behavior
where answer generation begins?" Gold answers remain offline labels only.

Run the exploratory discovery set:

```bash
python -m evaluation.gate0.run_answer_onset \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --data results/gate0/minigate/minigate_24.jsonl \
  --budget-curve-path evaluation/curve_data/llama-3.1-8b/snapkv_maxpool_sink4_win_32_llama_avg_ratio.npy \
  --marginal-profile-path results/gate0/lu_profile/gate0_lu_global_snapkv_sink4_win32_marginal_step16.npz \
  --output-dir results/gate0/answer_onset/discovery24 \
  --trace-lens 1 4 8
```

Summarize with:

```bash
python -m evaluation.gate0.summarize_answer_onset \
  --raw-dir results/gate0/answer_onset/discovery24 \
  --output results/gate0/answer_onset/discovery24_summary.json
```

The original 24 prompts are a discovery/diagnosis set for this post-hoc
hypothesis. If an answer-onset signal is promising, freeze one signal and one
trace length, then evaluate it on a fresh non-overlapping confirmatory set.
Do not report a same-set best-of-1/4/8 choice as confirmatory evidence.
