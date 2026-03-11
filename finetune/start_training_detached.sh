#!/bin/bash
# 完全后台运行训练 - 使用screen/tmux/nohup

set -e

# 配置
SESSION_NAME="trajma"
LOG_DIR="./logs"
TRAINING_LOG="${LOG_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "后台训练启动"
echo "=========================================="
echo "会话名: ${SESSION_NAME}"
echo "日志文件: ${TRAINING_LOG}"
echo ""

# 检查screen是否安装
if command -v screen &> /dev/null; then
    METHOD="screen"
elif command -v tmux &> /dev/null; then
    METHOD="tmux"
else
    METHOD="nohup"
fi

echo "使用方式: ${METHOD}"
echo ""

case ${METHOD} in
    screen)
        # 检查会话是否已存在
        if screen -list | grep -q "${SESSION_NAME}"; then
            echo "⚠️  Screen会话 '${SESSION_NAME}' 已存在"
            echo ""
            echo "选项:"
            echo "  1. 查看现有会话: screen -r ${SESSION_NAME}"
            echo "  2. 杀死现有会话: screen -S ${SESSION_NAME} -X quit"
            echo "  3. 使用不同名称"
            exit 1
        fi
        
        # 创建detached screen会话
        screen -dmS ${SESSION_NAME} bash -c "
            cd $(pwd)
            echo '=========================================='
            echo '训练开始: \$(date)'
            echo '=========================================='
            
            # 启动TensorBoard
            if command -v tensorboard &> /dev/null; then
                nohup tensorboard --logdir=./output/llama3_1_trajectory_lora/runs \
                                  --port=6006 --bind_all > ${LOG_DIR}/tensorboard.log 2>&1 &
                echo \"✓ TensorBoard已启动 (PID: \$!)\"
            fi
            
            # 训练
            echo \"✓ 开始训练...\"
            CUDA_VISIBLE_DEVICES=0 python train_llama_lora.py 2>&1 | tee ${TRAINING_LOG}
            
            echo ''
            echo '=========================================='
            echo '训练完成: \$(date)'
            echo '=========================================='
            
            # 保持会话开启
            echo ''
            echo '按任意键关闭会话...'
            read
        "
        
        echo "✓ 训练已在screen会话中启动"
        echo ""
        echo "常用命令:"
        echo "  查看会话列表:  screen -ls"
        echo "  连接会话:      screen -r ${SESSION_NAME}"
        echo "  断开会话:      Ctrl+A, 然后按 D"
        echo "  杀死会话:      screen -S ${SESSION_NAME} -X quit"
        echo "  查看日志:      tail -f ${TRAINING_LOG}"
        ;;
        
    tmux)
        # 检查会话是否已存在
        if tmux has-session -t ${SESSION_NAME} 2>/dev/null; then
            echo "⚠️  Tmux会话 '${SESSION_NAME}' 已存在"
            echo ""
            echo "选项:"
            echo "  1. 连接会话: tmux attach -t ${SESSION_NAME}"
            echo "  2. 杀死会话: tmux kill-session -t ${SESSION_NAME}"
            exit 1
        fi
        
        # 创建detached tmux会话
        tmux new-session -d -s ${SESSION_NAME} "
            cd $(pwd)
            echo '=========================================='
            echo '训练开始: '\$(date)
            echo '=========================================='
            
            # 启动TensorBoard
            if command -v tensorboard &> /dev/null; then
                nohup tensorboard --logdir=./output/llama3_1_trajectory_lora/runs \
                                  --port=6006 --bind_all > ${LOG_DIR}/tensorboard.log 2>&1 &
                echo 'TensorBoard已启动'
            fi
            
            # 训练
            CUDA_VISIBLE_DEVICES=0 python train_llama_lora.py 2>&1 | tee ${TRAINING_LOG}
            
            echo ''
            echo '=========================================='
            echo '训练完成: '\$(date)
            echo '=========================================='
            bash
        "
        
        echo "✓ 训练已在tmux会话中启动"
        echo ""
        echo "常用命令:"
        echo "  查看会话列表:  tmux ls"
        echo "  连接会话:      tmux attach -t ${SESSION_NAME}"
        echo "  断开会话:      Ctrl+B, 然后按 D"
        echo "  杀死会话:      tmux kill-session -t ${SESSION_NAME}"
        echo "  查看日志:      tail -f ${TRAINING_LOG}"
        ;;
        
    nohup)
        echo "⚠️  screen和tmux都未安装,使用nohup"
        echo "   建议安装: sudo apt install screen 或 sudo apt install tmux"
        echo ""
        
        # 启动TensorBoard
        if command -v tensorboard &> /dev/null; then
            nohup tensorboard --logdir=./output/llama3_1_trajectory_lora/runs \
                              --port=6006 --bind_all > ${LOG_DIR}/tensorboard.log 2>&1 &
            TENSORBOARD_PID=$!
            echo "✓ TensorBoard已启动 (PID: ${TENSORBOARD_PID})"
        fi
        
        # 训练
        nohup bash -c "CUDA_VISIBLE_DEVICES=0 python train_llama_lora.py" > ${TRAINING_LOG} 2>&1 &
        TRAINING_PID=$!
        
        echo "✓ 训练已启动 (PID: ${TRAINING_PID})"
        echo ""
        echo "常用命令:"
        echo "  查看进程:      ps aux | grep train_llama_lora"
        echo "  杀死进程:      kill ${TRAINING_PID}"
        echo "  查看日志:      tail -f ${TRAINING_LOG}"
        ;;
esac

echo ""
echo "TensorBoard: http://localhost:6006"
echo ""
echo "=========================================="
