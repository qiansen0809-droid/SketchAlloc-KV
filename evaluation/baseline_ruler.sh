#!/bin/bash
set -euo pipefail

# ==============================================================
# Configuration
# ==============================================================

MODEL_NAME="llama-3.1-8b"
MODEL_PATH="/ssd1/models/Llama-3.1-8B-Instruct"
TARGET_GPU=3
RESULTS_ROOT="results"

# Methods that require a compression ratio sweep (no budget curve needed)
PRESS_NAMES=(
    "snapkv"
    "keydiff"
    "expected_attention_e2"
    "adakv_snapkv"
    "adakv_keydiff"
    "adakv_expected_attention_e2"
    "pyramidkv_snapkv"
    "pyramidkv_keydiff"
    "pyramidkv_ea_e2"
)

RATIOS=(0.8)

TASKS=("4096")

# ==============================================================
# Execution
# ==============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for press in "${PRESS_NAMES[@]}"; do
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
                > "${LOG_FILE}" 2>&1

            echo "Done: ${press}/${ratio}/${task}"
        done
    done
done

# no_press: compression_ratio is irrelevant (forced to 0.0 internally), run once
echo "=================================================="
echo "  Running no_press baseline"
echo "=================================================="
for task in "${TASKS[@]}"; do
    OUT_DIR="${SCRIPT_DIR}/${RESULTS_ROOT}/${MODEL_NAME}/no_press/0.0"
    mkdir -p "${OUT_DIR}"
    LOG_FILE="${OUT_DIR}/${task}.log"

    python3 "${SCRIPT_DIR}/evaluate.py" \
        --dataset ruler \
        --data_dir "${task}" \
        --model "${MODEL_PATH}" \
        --device "cuda:${TARGET_GPU}" \
        --press_name no_press \
        --compression_ratio 0.0 \
        --output_dir "${OUT_DIR}" \
        > "${LOG_FILE}" 2>&1

    echo "Done: no_press/0.0/${task}"
done

echo "All baseline ruler evaluations finished."
