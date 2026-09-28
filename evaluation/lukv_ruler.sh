#!/bin/bash
set -euo pipefail

# ==============================================================
# Configuration
# ==============================================================

# Short name used for directory organisation (no slashes)
MODEL_NAME="llama-3.1-8b"

# Full path to the model weights
MODEL_PATH="/ssd1/models/Llama-3.1-8B-Instruct"

# LU-KV methods to evaluate in order: lu_snapkv -> lu_keydiff -> lu_ea
declare -A PRESS_CURVES
PRESS_CURVES["lu_snapkv"]="curve_data/llama-3.1-8b/snapkv_maxpool_sink4_win_32_llama_avg_ratio.npy"
PRESS_CURVES["lu_keydiff"]="curve_data/llama-3.1-8b/keydiff_avg_ratio.npy"
PRESS_CURVES["lu_ea"]="curve_data/llama-3.1-8b/ea_0.02_sink4_win1_llama_avg_ratio.npy"

PRESS_ORDER=("lu_snapkv" "lu_keydiff" "lu_ea")

# RULER context-length tasks to evaluate
TASKS=("4096")

# Compression ratios to sweep
RATIOS=(0.8)

# GPU index
TARGET_GPU=1

# Root directory for all results (relative to this script's location)
RESULTS_ROOT="results"

# ==============================================================
# Execution
# ==============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for press in "${PRESS_ORDER[@]}"; do

    BUDGET_CURVE_PATH="${PRESS_CURVES[$press]}"

    for ratio in "${RATIOS[@]}"; do
        for task in "${TASKS[@]}"; do

            OUT_DIR="${SCRIPT_DIR}/${RESULTS_ROOT}/${MODEL_NAME}/${press}/${ratio}"
            mkdir -p "${OUT_DIR}"

            LOG_FILE="${OUT_DIR}/${task}.log"

            echo "=================================================="
            echo "  Model  : ${MODEL_NAME}"
            echo "  Press  : ${press}"
            echo "  Task   : ${task}"
            echo "  Ratio  : ${ratio}"
            echo "  Output : ${OUT_DIR}"
            echo "  Log    : ${LOG_FILE}"
            echo "=================================================="

            python3 "${SCRIPT_DIR}/evaluate.py" \
                --dataset ruler \
                --data_dir "${task}" \
                --model "${MODEL_PATH}" \
                --device "cuda:${TARGET_GPU}" \
                --press_name "${press}" \
                --compression_ratio "${ratio}" \
                --output_dir "${OUT_DIR}" \
                --budget_curve_path "${BUDGET_CURVE_PATH}" \
                > "${LOG_FILE}" 2>&1

            echo "Done: ${MODEL_NAME}/${press}/${ratio}/${task}"
        done
    done
done
