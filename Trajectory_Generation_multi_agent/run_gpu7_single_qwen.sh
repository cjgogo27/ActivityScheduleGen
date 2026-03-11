#!/bin/bash
W=/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent
PY=/data2/miniconda3/envs/trajlla/bin/python
export CUDA_VISIBLE_DEVICES=7
mkdir -p $W/logs
for state in california arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model sft --skip_editor --data $W/eval_data/${state}_50.jsonl >> $W/logs/inference_sft_single_${state}_${TS}.log 2>&1
  echo "[GPU7] sft_single $state done $(date)"
done
echo "[GPU7] ALL Single_agent_Qwen DONE $(date)"
