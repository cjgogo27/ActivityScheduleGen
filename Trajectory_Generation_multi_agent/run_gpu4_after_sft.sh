#!/bin/bash
W=/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent
PY=/data2/miniconda3/envs/trajlla/bin/python
export CUDA_VISIBLE_DEVICES=4
mkdir -p $W/logs
echo "[GPU4] Waiting for PID 971845 (sft CA)..."
while kill -0 971845 2>/dev/null; do sleep 30; done
echo "[GPU4] sft CA done at $(date)"
for state in arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model sft --data $W/eval_data/${state}_metadata.jsonl >> $W/logs/inference_sft_${state}_${TS}.log 2>&1
  echo "[GPU4] Table1 sft $state done $(date)"
done
for state in california arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model sft --data $W/eval_data/${state}_50.jsonl >> $W/logs/inference_sft_multi_${state}_${TS}.log 2>&1
  echo "[GPU4] Table2 sft_multi $state done $(date)"
done
echo "[GPU4] ALL DONE $(date)"
