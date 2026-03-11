#!/usr/bin/env bash
# LLaMA SFT inference on GPU 5: California, Arizona, Georgia, Oklahoma
WDIR=/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent
PYTHON=/data2/miniconda3/envs/trajlla/bin/python
LOGS=$WDIR/logs

mkdir -p $LOGS
TS=$(date +%Y%m%d_%H%M%S)

cd $WDIR

echo "[SFT_LLAMA] California ..."
CUDA_VISIBLE_DEVICES=5 $PYTHON $WDIR/run_inference.py --model sft_llama --num_samples 100 \
  > $LOGS/inference_sft_llama_california_${TS}.log 2>&1

echo "[SFT_LLAMA] Arizona ..."
CUDA_VISIBLE_DEVICES=5 $PYTHON $WDIR/run_inference.py --model sft_llama \
  --data $WDIR/eval_data/arizona_metadata.jsonl \
  > $LOGS/inference_sft_llama_arizona_${TS}.log 2>&1

echo "[SFT_LLAMA] Georgia ..."
CUDA_VISIBLE_DEVICES=5 $PYTHON $WDIR/run_inference.py --model sft_llama \
  --data $WDIR/eval_data/georgia_metadata.jsonl \
  > $LOGS/inference_sft_llama_georgia_${TS}.log 2>&1

echo "[SFT_LLAMA] Oklahoma ..."
CUDA_VISIBLE_DEVICES=5 $PYTHON $WDIR/run_inference.py --model sft_llama \
  --data $WDIR/eval_data/oklahoma_metadata.jsonl \
  > $LOGS/inference_sft_llama_oklahoma_${TS}.log 2>&1

echo "[SFT_LLAMA] All done!"
