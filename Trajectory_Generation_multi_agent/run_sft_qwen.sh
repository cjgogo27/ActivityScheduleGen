#!/usr/bin/env bash
# Qwen SFT inference on GPU 4: California, Arizona, Georgia, Oklahoma
WDIR=/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent
PYTHON=/data2/miniconda3/envs/trajlla/bin/python
LOGS=$WDIR/logs

mkdir -p $LOGS
TS=$(date +%Y%m%d_%H%M%S)

cd $WDIR

echo "[SFT_QWEN] California ..."
CUDA_VISIBLE_DEVICES=4 $PYTHON $WDIR/run_inference.py --model sft --num_samples 100 \
  > $LOGS/inference_sft_california_${TS}.log 2>&1

echo "[SFT_QWEN] Arizona ..."
CUDA_VISIBLE_DEVICES=4 $PYTHON $WDIR/run_inference.py --model sft \
  --data $WDIR/eval_data/arizona_metadata.jsonl \
  > $LOGS/inference_sft_arizona_${TS}.log 2>&1

echo "[SFT_QWEN] Georgia ..."
CUDA_VISIBLE_DEVICES=4 $PYTHON $WDIR/run_inference.py --model sft \
  --data $WDIR/eval_data/georgia_metadata.jsonl \
  > $LOGS/inference_sft_georgia_${TS}.log 2>&1

echo "[SFT_QWEN] Oklahoma ..."
CUDA_VISIBLE_DEVICES=4 $PYTHON $WDIR/run_inference.py --model sft \
  --data $WDIR/eval_data/oklahoma_metadata.jsonl \
  > $LOGS/inference_sft_oklahoma_${TS}.log 2>&1

echo "[SFT_QWEN] All done!"
