#!/bin/bash
W=/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent
PY=/data2/miniconda3/envs/trajlla/bin/python
mkdir -p $W/logs
echo "[API] Starting GPT experiments at $(date)"
for state in california arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model api_merged --skip_editor --data $W/eval_data/${state}_50.jsonl >> $W/logs/inference_api_single_${state}_${TS}.log 2>&1
  echo "[API] api_single $state done $(date)"
done
for state in california arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model api_merged --data $W/eval_data/${state}_50.jsonl >> $W/logs/inference_api_multi_${state}_${TS}.log 2>&1
  echo "[API] api_multi $state done $(date)"
done
echo "[API] ALL GPT DONE $(date)"
