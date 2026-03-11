#!/bin/bash
# 运行 RAGHome 基线推理（4个区域×1 = 4个API任务）
W=/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent
PY=/data2/miniconda3/envs/trajlla/bin/python

mkdir -p $W/logs

echo "[RAGHome] 开始 RAGHome 推理 at $(date)"

for state in california arizona georgia oklahoma; do
    TS=$(date +%Y%m%d_%H%M%S)
    echo "[RAGHome] $state ..."
    $PY $W/run_baseline_raghome.py --data $W/eval_data/${state}_50.jsonl --delay 0.5 2>&1 | tee $W/logs/baseline_raghome_${state}_${TS}.log
    echo "[RAGHome] $state done $(date)"
done

echo "[RAGHome] 全部完成 at $(date)"
