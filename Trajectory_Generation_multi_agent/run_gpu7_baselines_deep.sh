#!/bin/bash
# 运行 DeepMove + LSTPM 基线推理（4个区域 × 2个模型 = 8个任务）
# 在GPU 7上顺序运行（这两个模型很轻量，很快）
W=/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent
PY=/data2/miniconda3/envs/trajlla/bin/python
export CUDA_VISIBLE_DEVICES=7

mkdir -p $W/logs

echo "[Baselines] 开始 DeepMove + LSTPM 推理 at $(date)"

for state in california arizona georgia oklahoma; do
    TS=$(date +%Y%m%d_%H%M%S)
    echo "[Baselines] DeepMove $state ..."
    $PY $W/run_baseline_deepmove.py --data $W/eval_data/${state}_50.jsonl --gpu 7 2>&1 | tee $W/logs/baseline_deepmove_${state}_${TS}.log
    echo "[Baselines] DeepMove $state done $(date)"
done

for state in california arizona georgia oklahoma; do
    TS=$(date +%Y%m%d_%H%M%S)
    echo "[Baselines] LSTPM $state ..."
    $PY $W/run_baseline_lstpm.py --data $W/eval_data/${state}_50.jsonl --gpu 7 2>&1 | tee $W/logs/baseline_lstpm_${state}_${TS}.log
    echo "[Baselines] LSTPM $state done $(date)"
done

echo "[Baselines] DeepMove + LSTPM 全部完成 at $(date)"
