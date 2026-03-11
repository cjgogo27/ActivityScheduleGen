#!/bin/bash

# Stage 3 GRPO Training Script — LLaMA-3.1-8B-Instruct
# GPU: 倒数第二张卡 (CUDA_VISIBLE_DEVICES=6)
# 注意：请在 Stage 2 SFT 训练完成后再运行此脚本

echo "=========================================="
echo "  Stage 3: GRPO Training"
echo "  Model: LLaMA-3.1-8B-Instruct + SFT LoRA"
echo "=========================================="
echo ""

# Activate conda environment
source /data2/miniconda3/etc/profile.d/conda.sh
conda activate trajlla

# Set working directory
WORK_DIR="/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent"
cd $WORK_DIR

# Log directory
LOG_DIR="$WORK_DIR/stage3_grpo_llama_output/logs"
mkdir -p $LOG_DIR

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "📁 Working directory: $WORK_DIR"
echo "📝 Log file: $LOG_FILE"
echo "📊 TensorBoard: $WORK_DIR/stage3_grpo_llama_output/tensorboard_logs"
echo "🖥️  GPU: CUDA_VISIBLE_DEVICES=6"
echo ""

# Prerequisite checks
if [ ! -f "stage1_training_data_3/metadata.jsonl" ]; then
    echo "❌ Error: GRPO data not found: stage1_training_data_3/metadata.jsonl"
    exit 1
fi

SFT_MODEL="$WORK_DIR/stage2_sft_llama_output_epoch10/final_model"
if [ ! -d "$SFT_MODEL" ]; then
    echo "❌ Error: SFT model not found at: $SFT_MODEL"
    echo "   Please run run_sft_llama_training.sh first and wait for it to complete."
    exit 1
fi

echo "✅ All prerequisites checked"
echo ""
echo "🚀 Starting GRPO training (background)..."
echo "   Monitor: tail -f $LOG_FILE"
echo "   TensorBoard: tensorboard --logdir=$WORK_DIR/stage3_grpo_llama_output/tensorboard_logs --port=6008"
echo ""

# Run training in background with GPU 6
CUDA_VISIBLE_DEVICES=6 nohup python train_grpo_llama.py > "$LOG_FILE" 2>&1 &
GRPO_PID=$!
echo "✅ GRPO training started with PID: $GRPO_PID (GPU 6)"
echo "   To stop: kill $GRPO_PID"
