#!/bin/bash
W=/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent
PY=/data2/miniconda3/envs/trajlla/bin/python
export CUDA_VISIBLE_DEVICES=6
mkdir -p $W/logs
echo "[GPU6] Waiting for PID 995792 (grpo_llama AZ)..."
while kill -0 995792 2>/dev/null; do sleep 30; done
echo "[GPU6] grpo_llama AZ done at $(date)"
for state in georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model grpo_llama --data $W/eval_data/${state}_metadata.jsonl >> $W/logs/inference_grpo_llama_${state}_${TS}.log 2>&1
  echo "[GPU6] Table1 grpo_llama $state done $(date)"
done
echo "[GPU6] ALL DONE $(date)"
