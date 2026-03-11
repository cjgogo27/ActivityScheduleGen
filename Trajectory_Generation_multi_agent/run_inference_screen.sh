#!/bin/bash
# 用法：
#   bash run_inference_screen.sh base 50        # Base 模型跑 50 条
#   bash run_inference_screen.sh sft            # SFT  模型全量跑
#   bash run_inference_screen.sh sft 50 stored  # SFT  模型，跳过 Planner
#   bash run_inference_screen.sh api 50         # API（gpt-5.2/5.1）跑 50 条
#   bash run_inference_screen.sh api 50 stored  # API 跳过 Planner/Realizer，直接 Editor

MODEL=${1:-sft}
NUM=${2:-""}
USE_STORED=${3:-""}

WORK_DIR="/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent"
LOG_DIR="${WORK_DIR}/results/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_NAME="infer_${MODEL}_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/inference_${MODEL}_${TIMESTAMP}.log"

# 拼接命令
CMD="cd ${WORK_DIR}"

# 激活 conda 环境
CMD="${CMD} && source /data2/miniconda3/etc/profile.d/conda.sh && conda activate trajlla"

# 主命令
PYTHON_CMD="python run_inference.py --model ${MODEL}"
[ -n "${NUM}" ] && PYTHON_CMD="${PYTHON_CMD} --num_samples ${NUM}"
[ "${USE_STORED}" = "stored" ] && PYTHON_CMD="${PYTHON_CMD} --use_stored_initial"

CMD="${CMD} && ${PYTHON_CMD}"

# Pipeline 描述
if [ "${MODEL}" = "api" ]; then
    if [ "${USE_STORED}" = "stored" ]; then
        PIPELINE_DESC="Stored Initial → API Editor (gpt-5.1)"
    else
        PIPELINE_DESC="API Planner (gpt-5.2) → Trip Realizer (gpt-5.2) → Editor (gpt-5.1)"
    fi
else
    if [ "${USE_STORED}" = "stored" ]; then
        PIPELINE_DESC="Stored Initial → ${MODEL^^} Editor"
    else
        PIPELINE_DESC="Planner → Editor [${MODEL^^}]"
    fi
fi

echo "=============================="
echo " 启动推理任务（后台 screen）"
echo "=============================="
echo "  Session  : ${SESSION_NAME}"
echo "  模型     : ${MODEL}"
echo "  样本数   : ${NUM:-全量}"
echo "  Pipeline : ${PIPELINE_DESC}"
echo "  日志文件 : ${LOG_FILE}"
echo "=============================="

screen -dmS "${SESSION_NAME}" bash -c "${CMD} > ${LOG_FILE} 2>&1"

sleep 0.5
if screen -ls | grep -q "${SESSION_NAME}"; then
    echo ""
    echo "✓ 任务已在后台启动"
    echo ""
    echo "  实时查看日志："
    echo "    tail -f ${LOG_FILE}"
    echo ""
    echo "  进入 screen 会话："
    echo "    screen -r ${SESSION_NAME}"
    echo ""
    echo "  查看所有 screen 会话："
    echo "    screen -ls"
else
    echo "✗ screen 启动失败，改为前台直接运行："
    eval "${CMD}"
fi
