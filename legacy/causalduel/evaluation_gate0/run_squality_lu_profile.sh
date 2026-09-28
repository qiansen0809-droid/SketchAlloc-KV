#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the local Llama-3.1-8B-Instruct directory}"

NUM_STORIES="${NUM_STORIES:-6}"
QUESTIONS_PER_STORY="${QUESTIONS_PER_STORY:-5}"
MIN_TOKENS="${MIN_TOKENS:-4096}"
MAX_TOKENS="${MAX_TOKENS:-7168}"
PROFILE_SEED="${PROFILE_SEED:-20260928}"
SWAP_SIZE="${SWAP_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

PROFILE_DIR="${PROFILE_DIR:-$ROOT_DIR/results/gate0/profile_data}"
PROFILE_JSONL="${PROFILE_JSONL:-$PROFILE_DIR/squality_train_${NUM_STORIES}x${QUESTIONS_PER_STORY}.jsonl}"
PROFILE_MANIFEST="${PROFILE_MANIFEST:-$PROFILE_DIR/squality_train_${NUM_STORIES}x${QUESTIONS_PER_STORY}_manifest.json}"

mkdir -p "$PROFILE_DIR"

echo "[Gate0] Preparing public SQuALITY v1.3 train profiling set"
python "$ROOT_DIR/evaluation/gate0/prepare_squality_profile.py" \
  --model "$MODEL_PATH" \
  --num-stories "$NUM_STORIES" \
  --questions-per-story "$QUESTIONS_PER_STORY" \
  --min-tokens "$MIN_TOKENS" \
  --max-tokens "$MAX_TOKENS" \
  --seed "$PROFILE_SEED" \
  --output "$PROFILE_JSONL" \
  --manifest "$PROFILE_MANIFEST"

echo
echo "[Gate0] Building LU boundary marginal profile around the official runtime curve"
MODEL_PATH="$MODEL_PATH" \
DATASET_PATH="$PROFILE_JSONL" \
CUDA_DEVICE="$CUDA_DEVICE" \
SWAP_SIZE="$SWAP_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
PROFILE_SEED="$PROFILE_SEED" \
ANSWER_PREFIX="Answer:" \
bash "$ROOT_DIR/evaluation/gate0/build_lu_marginal_profile.sh"

echo
echo "[Gate0] Done."
echo "Profile JSONL: $PROFILE_JSONL"
echo "Manifest:      $PROFILE_MANIFEST"
echo "Marginals:     $ROOT_DIR/results/gate0/lu_profile/gate0_lu_global_snapkv_sink4_win32_marginal_step${SWAP_SIZE}.npz"
