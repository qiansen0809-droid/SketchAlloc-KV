#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CURVE_DIR="$ROOT_DIR/evaluation/curve_data"

: "${MODEL_PATH:?Set MODEL_PATH to Llama-3.1-8B-Instruct (local path or HF id)}"
: "${DATASET_PATH:?Set DATASET_PATH to a separate LU profiling JSONL file}"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
SWAP_SIZE="${SWAP_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
ANSWER_PREFIX="${ANSWER_PREFIX:-Answer:}"
PROFILE_SEED="${PROFILE_SEED:-20260928}"
MID_DATA_DIR="${MID_DATA_DIR:-$ROOT_DIR/results/gate0/lu_profile_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results/gate0/lu_profile}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-gate0_lu_global}"
STATIC_BUDGET_CURVE="${STATIC_BUDGET_CURVE:-$CURVE_DIR/llama-3.1-8b/snapkv_maxpool_sink4_win_32_llama_avg_ratio.npy}"

# Raw context/question traces are a complete run artifact. Remove any stale
# context_* directories so a smaller pilot run cannot accidentally mix with
# data from an earlier profile.
rm -rf "$MID_DATA_DIR"
mkdir -p "$MID_DATA_DIR" "$OUTPUT_DIR"

if [[ ! -f "$STATIC_BUDGET_CURVE" ]]; then
  echo "Static LU-KV budget curve not found: $STATIC_BUDGET_CURVE" >&2
  exit 1
fi

echo "[Gate0] Step 1/2: extract LU-KV long-horizon oracle/scorer data"
python "$CURVE_DIR/step1_llama.py" \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$MID_DATA_DIR" \
  --cuda_device "$CUDA_DEVICE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --answer_prefix "$ANSWER_PREFIX" \
  --seed "$PROFILE_SEED" \
  --methods snapkv

echo "[Gate0] Step 2/2: solve static LU curve and export boundary marginals"
python "$CURVE_DIR/step2_compute_curve.py" \
  --input_dir "$MID_DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --output_prefix "$OUTPUT_PREFIX" \
  --configs snapkv:4:32 \
  --threshold 99 \
  --layerwise \
  --num_workers "$NUM_WORKERS" \
  --export_marginals \
  --marginal_step_tokens "$SWAP_SIZE" \
  --static_budget_curve_path "$STATIC_BUDGET_CURVE"

echo
echo "Expected outputs:"
echo "  $OUTPUT_DIR/${OUTPUT_PREFIX}_snapkv_sink4_win32.npy"
echo "  $OUTPUT_DIR/${OUTPUT_PREFIX}_snapkv_sink4_win32_marginal_step${SWAP_SIZE}.npz"
echo "Marginal boundary source:"
echo "  $STATIC_BUDGET_CURVE"
echo
echo "Important: DATASET_PATH is offline LU profiling data. Do not use Gate-0 MiniGate,"
echo "calibration, or held-out test prompts to build this marginal profile."
