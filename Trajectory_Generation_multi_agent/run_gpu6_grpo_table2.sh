#!/bin/bash
# GPU6: grpo_Qwen Table2 (50-sample single+multi) and grpo_LLaMA Table2
W=/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent
PY=/data2/miniconda3/envs/trajlla/bin/python
export CUDA_VISIBLE_DEVICES=6
mkdir -p $W/logs

echo "[GPU6] Starting grpo_Qwen + grpo_LLaMA Table2 at $(date)"

# =============================================
# Table2: grpo_Qwen single-agent (--skip_editor)
# =============================================
echo "[GPU6] --- grpo_Qwen Table2 single-agent ---"
for state in california arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model grpo --skip_editor \
    --data $W/eval_data/${state}_50.jsonl \
    >> $W/logs/inference_grpo_single_${state}_${TS}.log 2>&1
  echo "[GPU6] grpo_single $state done $(date)"
done

# =============================================
# Table2: grpo_Qwen multi-agent (full pipeline)
# =============================================
echo "[GPU6] --- grpo_Qwen Table2 multi-agent ---"
for state in california arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model grpo \
    --data $W/eval_data/${state}_50.jsonl \
    >> $W/logs/inference_grpo_multi_${state}_${TS}.log 2>&1
  echo "[GPU6] grpo_multi $state done $(date)"
done

# =============================================
# Table2: grpo_LLaMA single-agent (--skip_editor)
# =============================================
echo "[GPU6] --- grpo_LLaMA Table2 single-agent ---"
for state in california arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model grpo_llama --skip_editor \
    --data $W/eval_data/${state}_50.jsonl \
    >> $W/logs/inference_grpo_llama_single_${state}_${TS}.log 2>&1
  echo "[GPU6] grpo_llama_single $state done $(date)"
done

# =============================================
# Table2: grpo_LLaMA multi-agent (full pipeline)
# =============================================
echo "[GPU6] --- grpo_LLaMA Table2 multi-agent ---"
for state in california arizona georgia oklahoma; do
  TS=$(date +%Y%m%d_%H%M%S)
  $PY $W/run_inference.py --model grpo_llama \
    --data $W/eval_data/${state}_50.jsonl \
    >> $W/logs/inference_grpo_llama_multi_${state}_${TS}.log 2>&1
  echo "[GPU6] grpo_llama_multi $state done $(date)"
done

echo "[GPU6] ALL DONE $(date)"
