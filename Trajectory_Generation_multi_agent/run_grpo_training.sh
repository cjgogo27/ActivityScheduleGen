#!/bin/bash

# Stage 3: GRPO Training Launcher
# This script activates the environment and runs GRPO training with monitoring

set -e  # Exit on error

# ==================== Environment Setup ====================

echo "🚀 Stage 3: GRPO Training Launcher"
echo "===================================="
echo ""

# Activate conda environment
echo "🔧 Activating conda environment..."
for _conda_init in /data2/miniconda3/etc/profile.d/conda.sh \
                   ~/miniconda3/etc/profile.d/conda.sh \
                   /opt/conda/etc/profile.d/conda.sh; do
    [ -f "$_conda_init" ] && source "$_conda_init" && break
done
conda activate trajlla
echo "  ✓ Environment: $(conda info --envs | grep \* | awk '{print $1}')"
echo "  ✓ Python: $(which python)"
echo ""

# ==================== Configuration ====================

WORK_DIR="/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent"
cd "$WORK_DIR"

LOG_DIR="$WORK_DIR/stage3_grpo_output/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/grpo_training_${TIMESTAMP}.log"

echo "📁 Working directory: $WORK_DIR"
echo "📝 Log file: $LOG_FILE"
echo "📊 TensorBoard logs: $WORK_DIR/stage3_grpo_output/tensorboard_logs"
echo ""

# ==================== Prerequisites Check ====================

echo "🔍 Checking prerequisites..."

# Check training data
if [ ! -f "$WORK_DIR/stage1_training_data_3/metadata.jsonl" ]; then
    echo "❌ Error: Training data not found!"
    echo "   Expected: stage1_training_data_3/metadata.jsonl"
    exit 1
fi
echo "  ✓ Training data exists"

# Check SFT model
if [ ! -d "$WORK_DIR/stage2_sft_output_epoch10/final_model" ]; then
    echo "❌ Error: SFT model not found!"
    echo "   Expected: stage2_sft_output_epoch10/final_model"
    echo "   Please complete Stage 2 SFT training first."
    exit 1
fi
echo "  ✓ SFT model exists"

# Check base model
BASE_MODEL_PATH="/data/alice/cjtest/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B"
if [ ! -d "$BASE_MODEL_PATH" ]; then
    echo "❌ Error: Base model not found at $BASE_MODEL_PATH"
    exit 1
fi
echo "  ✓ Base model exists"

echo ""

# ==================== Training Execution ====================

echo "🏋️ Starting GRPO training..."
echo "  Screen session: grpo_train"
echo "  Monitor: tail -f $LOG_FILE"
echo "  TensorBoard: tensorboard --logdir=$WORK_DIR/stage3_grpo_output/tensorboard_logs --port=6008"
echo ""
echo "Press Ctrl+C within 3 seconds to cancel, or wait to start..."
sleep 3

# Run training in background with nohup (使用最后一张显卡 GPU 7)
CUDA_VISIBLE_DEVICES=7 nohup python train_grpo.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo ""
echo "✅ Training started in background."
echo "   PID: $TRAIN_PID"
echo "   Log : $LOG_FILE"
echo ""
echo "Useful commands:"
echo "  tail -f $LOG_FILE                          # 实时查看日志"
echo "  kill $TRAIN_PID                            # 停止训练"
echo "  tensorboard --logdir=$WORK_DIR/stage3_grpo_output/tensorboard_logs --port=6008"
