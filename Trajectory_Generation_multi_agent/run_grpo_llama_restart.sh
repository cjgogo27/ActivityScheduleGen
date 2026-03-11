#!/bin/bash
# 重启 LLaMA GRPO v1 (GPU 6) + GRPO v2 (GPU 5)
# 修复内容:
#   1. time_to_minutes 鲁棒化（防 IndexError / ValueError）
#   2. top_k=0 修正（v1）
#   3. temperature=0.8（降低 entropy，防止输出崩溃）
#   4. beta=0.04（KL 惩罚，防止训练崩溃）

set -e
source /data2/miniconda3/etc/profile.d/conda.sh
conda activate trajlla

WORK_DIR="/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent"
cd "$WORK_DIR"

TS=$(date +"%Y%m%d_%H%M%S")

echo "=== 启动 LLaMA GRPO v1 (GPU 6) ==="
mkdir -p stage3_grpo_llama_output/logs
screen -dmS llama_grpo_v1 bash -c "
  source /data2/miniconda3/etc/profile.d/conda.sh && conda activate trajlla &&
  cd $WORK_DIR &&
  CUDA_VISIBLE_DEVICES=6 python train_grpo_llama.py \
    2>&1 | tee stage3_grpo_llama_output/logs/training_restart_${TS}.log
"
echo "  → screen: llama_grpo_v1 | log: stage3_grpo_llama_output/logs/training_restart_${TS}.log"

echo "=== 启动 LLaMA GRPO v2 (GPU 5) ==="
mkdir -p stage3_grpo_llama_v2_output/logs
screen -dmS llama_grpo_v2 bash -c "
  source /data2/miniconda3/etc/profile.d/conda.sh && conda activate trajlla &&
  cd $WORK_DIR &&
  CUDA_VISIBLE_DEVICES=5 python train_grpo_llama_v2.py \
    2>&1 | tee stage3_grpo_llama_v2_output/logs/training_restart_${TS}.log
"
echo "  → screen: llama_grpo_v2 | log: stage3_grpo_llama_v2_output/logs/training_restart_${TS}.log"

echo ""
echo "监控命令:"
echo "  tail -f $WORK_DIR/stage3_grpo_llama_output/logs/training_restart_${TS}.log"
echo "  tail -f $WORK_DIR/stage3_grpo_llama_v2_output/logs/training_restart_${TS}.log"
echo "  screen -ls"
