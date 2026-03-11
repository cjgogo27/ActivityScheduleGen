#!/bin/bash
# LLaMA v2 Pipeline: SFT (epoch20, LoRA r=16) → GRPO
# GPU: CUDA_VISIBLE_DEVICES=5
# 以 screen 会话后台运行，SFT 完成后自动接着跑 GRPO

set -e

source /data2/miniconda3/etc/profile.d/conda.sh
conda activate trajlla

WORK_DIR="/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent"
cd "$WORK_DIR"

# ── 日志路径 ──
SFT_LOG_DIR="$WORK_DIR/stage2_sft_llama_v2_epoch20/logs"
GRPO_LOG_DIR="$WORK_DIR/stage3_grpo_llama_v2_output/logs"
mkdir -p "$SFT_LOG_DIR" "$GRPO_LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SFT_LOG="$SFT_LOG_DIR/training_${TIMESTAMP}.log"
GRPO_LOG="$GRPO_LOG_DIR/training_${TIMESTAMP}.log"

echo "=========================================="
echo "  LLaMA v2 Pipeline (GPU 5)"
echo "  SFT  → $SFT_LOG"
echo "  GRPO → $GRPO_LOG"
echo "=========================================="

# ── Step 1: SFT v2 ──
echo "[$(date '+%H:%M:%S')] Starting SFT v2 (epoch=20, LoRA r=16)..." | tee -a "$SFT_LOG"
CUDA_VISIBLE_DEVICES=5 python train_sft_llama_v2.py 2>&1 | tee -a "$SFT_LOG"

# 检查 SFT 是否成功
SFT_MODEL="$WORK_DIR/stage2_sft_llama_v2_epoch20/final_model"
if [ ! -d "$SFT_MODEL" ]; then
    echo "[ERROR] SFT v2 failed — final_model not found. Aborting GRPO." | tee -a "$GRPO_LOG"
    exit 1
fi

echo ""
echo "[$(date '+%H:%M:%S')] SFT v2 完成，开始 GRPO v2..." | tee -a "$GRPO_LOG"

# ── Step 2: GRPO v2 ──
CUDA_VISIBLE_DEVICES=5 python train_grpo_llama_v2.py 2>&1 | tee -a "$GRPO_LOG"

echo ""
echo "[$(date '+%H:%M:%S')] ✅ LLaMA v2 Pipeline 全部完成！"
echo "  SFT  log: $SFT_LOG"
echo "  GRPO log: $GRPO_LOG"
