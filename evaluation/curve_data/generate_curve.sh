#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e 

# ==================== 1. Configuration Area ====================

# [Choose Model Type: llama, mistral, or qwen]
MODEL_TYPE="llama" 

# [Paths & Devices]
MODEL_PATH="/ssd1/models/llama-3.1-8b"
DATASET_PATH="curve_data/novel.jsonl"
CUDA_DEVICE="2"

# [Temporary directory for Step 1 raw features]
MID_DATA_DIR="curve_data/${MODEL_TYPE}_novel"

# [Method configurations format -> method_name:sink_size:window_size]
# Example: snapkv uses window 32, others use window 1
METHOD_CONFIGS="ea:4:1 snapkv:4:32 keydiff:4:1"

# Leave empty "" for Layer-Independent, or use "--layerwise" for Global-Layerwise
LAYERWISE_FLAG="--layerwise" 

# [Final Output Settings]
FINAL_OUTPUT_DIR="${MID_DATA_DIR}_results"
FINAL_OUTPUT_PREFIX="avg_curve_global"


# ==================== 2. Execution Logic ====================

echo "========================================================="
echo "  🚀 Starting End-to-End Attention Analysis ($MODEL_TYPE) "
echo "========================================================="

# Determine Python script
if [ "$MODEL_TYPE" == "llama" ]; then
    STEP1_SCRIPT="step1_llama.py"
elif [ "$MODEL_TYPE" == "mistral" ]; then
    STEP1_SCRIPT="step1_mistral.py"
elif [ "$MODEL_TYPE" == "qwen" ]; then
    STEP1_SCRIPT="step1_qwen.py"
else
    echo "Error: Unknown MODEL_TYPE '$MODEL_TYPE'. Use llama, mistral, or qwen."
    exit 1
fi

echo "---------------------------------------------------------"
echo "[1/2] Extracting Attention & Feature Matrices ($STEP1_SCRIPT) ..."
echo "---------------------------------------------------------"
python3 curve_data/$STEP1_SCRIPT \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$MID_DATA_DIR" \
    --cuda_device "$CUDA_DEVICE"

echo "---------------------------------------------------------"
echo "[2/2] Computing Convex Hull & Averages In-Memory ..."
echo "      Configurations: $METHOD_CONFIGS"
echo "---------------------------------------------------------"
python3 curve_data/step2_compute_curve.py \
    --input_dir "$MID_DATA_DIR" \
    --output_dir "$FINAL_OUTPUT_DIR" \
    --output_prefix "$FINAL_OUTPUT_PREFIX" \
    --configs $METHOD_CONFIGS \
    $LAYERWISE_FLAG
