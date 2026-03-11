#!/bin/bash

# Stage 2 SFT Training Script
# This script runs the SFT training with proper logging and monitoring

echo "=========================================="
echo "  Stage 2: SFT Training - Editor Agent"
echo "  Model: Qwen3-8B + LoRA"
echo "=========================================="
echo ""

# Activate conda environment
source /data2/miniconda3/etc/profile.d/conda.sh
conda activate trajlla

# Set working directory
WORK_DIR="/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent"
cd $WORK_DIR

# Log directory - Updated for epoch 10 training
LOG_DIR="$WORK_DIR/stage2_sft_output_epoch10/logs"
mkdir -p $LOG_DIR

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "📁 Working directory: $WORK_DIR"
echo "📝 Log file: $LOG_FILE"
echo "📊 TensorBoard logs: $WORK_DIR/stage2_sft_output_epoch10/tensorboard_logs"
echo ""

# Check if training data exists
if [ ! -f "stage1_training_data_3/sft_training_data.jsonl" ]; then
    echo "❌ Error: Training data not found!"
    echo "Please run: python convert_separated_to_sft.py"
    exit 1
fi

# Check if model exists
if [ ! -d "/data/alice/cjtest/FinalTraj_KDD/finetune/models/Qwen3-8B/Qwen/Qwen3-8B" ]; then
    echo "❌ Error: Qwen3-8B model not found!"
    echo "Please run: python download_qwen_model.py"
    exit 1
fi

echo "✅ All prerequisites checked"
echo ""
echo "🚀 Starting SFT training..."
echo "   To view logs in real-time: tail -f $LOG_FILE"
echo "   To view TensorBoard: tensorboard --logdir=$WORK_DIR/stage2_sft_output_epoch10/tensorboard_logs --port=6006"
echo ""

# Run training
python train_sft.py 2>&1 | tee -a $LOG_FILE

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  ✅ Training completed successfully!"
    echo "=========================================="
    echo ""
    echo "📊 View training metrics:"
    echo "   tensorboard --logdir=$WORK_DIR/stage2_sft_output/tensorboard_logs --port=6006"
    echo ""
    echo "🧪 Test the model:"
    echo "   python test_sft_model.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "  ❌ Training failed!"
    echo "=========================================="
    echo ""
    echo "Check logs: $LOG_FILE"
    exit 1
fi
