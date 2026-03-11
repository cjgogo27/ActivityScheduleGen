#!/bin/bash

# Stage 2 SFT Training Script — LLaMA-3.1-8B-Instruct
# GPU: 倒数第二张卡 (CUDA_VISIBLE_DEVICES=6)

echo "=========================================="
echo "  Stage 2: SFT Training - Editor Agent"
echo "  Model: LLaMA-3.1-8B-Instruct + LoRA"
echo "=========================================="
echo ""

# Activate conda environment
source /data2/miniconda3/etc/profile.d/conda.sh
conda activate trajlla

# Set working directory
WORK_DIR="/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent"
cd $WORK_DIR

# Log directory
LOG_DIR="$WORK_DIR/stage2_sft_llama_output_epoch10/logs"
mkdir -p $LOG_DIR

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "📁 Working directory: $WORK_DIR"
echo "📝 Log file: $LOG_FILE"
echo "📊 TensorBoard: $WORK_DIR/stage2_sft_llama_output_epoch10/tensorboard_logs"
echo "🖥️  GPU: CUDA_VISIBLE_DEVICES=6"
echo ""

# Prerequisite checks
if [ ! -f "stage1_training_data_3/sft_training_data.jsonl" ]; then
    echo "❌ Error: Training data not found: stage1_training_data_3/sft_training_data.jsonl"
    exit 1
fi

LLAMA_MODEL="/data/alice/cjtest/FinalTraj_arr/finetune/models/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct"
if [ ! -d "$LLAMA_MODEL" ]; then
    echo "❌ Error: LLaMA model not found at: $LLAMA_MODEL"
    exit 1
fi

echo "✅ All prerequisites checked"
echo ""
echo "🚀 Starting SFT training (background)..."
echo "   Monitor: tail -f $LOG_FILE"
echo "   TensorBoard: tensorboard --logdir=$WORK_DIR/stage2_sft_llama_output_epoch10/tensorboard_logs --port=6007"
echo ""

# Run training in background with GPU 6
CUDA_VISIBLE_DEVICES=6 nohup python train_sft_llama.py > "$LOG_FILE" 2>&1 &
SFT_PID=$!
echo "✅ SFT training started with PID: $SFT_PID (GPU 6)"
echo "   To stop: kill $SFT_PID"
