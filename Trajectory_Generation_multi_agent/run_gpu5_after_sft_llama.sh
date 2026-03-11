#!/bin/bash
W=/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent
PY=/data2/miniconda3/envs/trajlla/bin/python
export CUDA_VISIBLE_DEVICES=5
mkdir -p $W/logs
echo "[GPU5] Waiting for PID 971846 (sft_llama CA)..."
while kill -0 971846 2>/dev/null; do sleep 30; done
echo "[GPU5] sft_llama CA done at $(date)"
for state in arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model sft_llama --data $W/eval_data/${state}_metadata.jsonl >> $W/logs/inference_sft_llama_${state}_${TS}.log 2>&1
  echo "[GPU5] Table1 sft_llama $state done $(date)"
done
for state in california arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model sft_llama --skip_editor --data $W/eval_data/${state}_50.jsonl >> $W/logs/inference_sft_llama_single_${state}_${TS}.log 2>&1
  echo "[GPU5] Table2 sft_llama_single $state done $(date)"
done
for state in california arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model sft_llama --data $W/eval_data/${state}_50.jsonl >> $W/logs/inference_sft_llama_multi_${state}_${TS}.log 2>&1
  echo "[GPU5] Table2 sft_llama_multi $state done $(date)"
done
echo "[GPU5] ALL DONE $(date)"
