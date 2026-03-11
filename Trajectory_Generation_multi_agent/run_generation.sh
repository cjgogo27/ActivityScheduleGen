#!/bin/bash

# 直接运行修改后的脚本

echo "════════════════════════════════════════════════════════════════"
echo "   🚀 Oklahoma轨迹生成 - 本地Llama模型"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "配置:"
echo "  - 脚本: generate_trajectories_multiagent_negotiation.py"
echo "  - 数据: Oklahoma (随机10个家庭)"
echo "  - 模型: Llama-3.1-8B + LoRA"
echo "  - 输出: output/, output_trajectories/"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

cd /data/mayue/cjy/Other_method/FinalTraj/Trajectory_Generation_multi_agent

# 运行
python generate_trajectories_multiagent_negotiation.py 2>&1 | tee generation_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "   ✅ 完成!"
echo "════════════════════════════════════════════════════════════════"
