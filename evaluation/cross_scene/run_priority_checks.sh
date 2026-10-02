#!/bin/bash
set -euo pipefail

cd /root/autodl-tmp/CrossScene-Gate

MODEL="/root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct"
MANIFEST="results/cross_scene/selection_v2_code20/manifest.json"
SAMPLES="results/cross_scene/selection_v2_code20/profile_samples.jsonl"
RAW="results/cross_scene/raw_v2_code20"
PROFILES="results/cross_scene/analysis_quick_v2_code20"
OFFICIAL="evaluation/curve_data/llama-3.1-8b/snapkv_maxpool_sink4_win_32_llama_avg_ratio.npy"

echo "============================================================"
echo "1/3 REAL CODE COMPLETION QUALITY"
echo "============================================================"
python -m evaluation.cross_scene.evaluate_code_quality \
  --manifest "${MANIFEST}" \
  --samples "${SAMPLES}" \
  --profiles-dir "${PROFILES}" \
  --model-path "${MODEL}" \
  --output-dir results/cross_scene/code_quality_v2_code20 \
  --tasks lcc,repobench-p \
  --profiles official_lu,pooled,task_lcc,task_repobench-p \
  --compression 0.80 \
  --sink 4 \
  --window 32 \
  --max-new-tokens 64

echo "============================================================"
echo "2/3 EQUAL-SAMPLE MIXED PROFILE COMPARISON"
echo "============================================================"
python -m evaluation.cross_scene.equal_sample_compare \
  --manifest "${MANIFEST}" \
  --raw-root "${RAW}" \
  --output-dir results/cross_scene/equal_sample_v2_code20 \
  --tasks lcc,repobench-p \
  --subset-size 5 \
  --method snapkv \
  --compression 0.80 \
  --sink 4 \
  --window 32

echo "============================================================"
echo "3/3 BUDGET-vs-TOKEN-SCORER HEADROOM"
echo "============================================================"
python -m evaluation.cross_scene.compute_headroom \
  --manifest "${MANIFEST}" \
  --raw-root "${RAW}" \
  --official-profile "${OFFICIAL}" \
  --output-dir results/cross_scene/headroom_v2_code20 \
  --tasks lcc,repobench-p \
  --method snapkv \
  --compression 0.80 \
  --sink 4 \
  --window 32

echo "============================================================"
echo "DONE"
echo "============================================================"
echo "Quality:"
echo "  results/cross_scene/code_quality_v2_code20/quality_results.json"
echo "Equal sample:"
echo "  results/cross_scene/equal_sample_v2_code20/equal_sample_results.json"
echo "Headroom:"
echo "  results/cross_scene/headroom_v2_code20/headroom_results.json"
