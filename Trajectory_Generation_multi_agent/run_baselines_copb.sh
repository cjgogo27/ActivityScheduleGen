#!/bin/bash
# 运行 CoPB 基线推理（4个区域×1 = 4个API任务）
# API密集型，在后台运行
W=/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent
PY=/data2/miniconda3/envs/trajlla/bin/python

mkdir -p $W/logs

echo "[CoPB] 开始 CoPB 推理 at $(date)"

for state in california arizona georgia oklahoma; do
    TS=$(date +%Y%m%d_%H%M%S)
    echo "[CoPB] $state ..."
    $PY $W/run_baseline_copb.py --data $W/eval_data/${state}_50.jsonl --delay 0.5 2>&1 | tee $W/logs/baseline_copb_${state}_${TS}.log
    echo "[CoPB] $state done $(date)"
done

echo "[CoPB] 全部完成 at $(date)"
