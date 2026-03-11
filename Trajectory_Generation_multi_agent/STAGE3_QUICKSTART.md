# Stage 3 GRPO Training - Quick Start Guide

## 前置条件检查

✅ **已完成**：
- Stage 1: Teacher数据生成（200个样本）
- Stage 2: SFT训练（10 epochs，模型在 `stage2_sft_output_epoch10/final_model/`）
- 环境依赖：trl==0.27.1 已安装

## 一键启动

### 方法1：直接运行（前台）

```bash
cd /data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent
bash run_grpo_training.sh
```

### 方法2：Screen后台运行（推荐）

```bash
cd /data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent
screen -S grpo_train bash run_grpo_training.sh

# 分离: Ctrl+A then D
# 重新连接: screen -r grpo_train
```

### 方法3：分离式启动

```bash
cd /data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent
screen -dmS grpo_train bash run_grpo_training.sh
screen -ls  # 查看状态
```

## 监控训练

### 查看实时日志

```bash
tail -f stage3_grpo_output/logs/grpo_training_*.log
```

### 查看奖励进展

```bash
grep "reward" stage3_grpo_output/logs/grpo_training_*.log
```

### TensorBoard可视化

```bash
tensorboard --logdir=stage3_grpo_output/tensorboard_logs --port=6008
# 然后访问: http://localhost:6008
```

## 训练时间估算

- **硬件**: 1x H200 (144GB)
- **每个样本生成时间**: ~40秒（4个候选 × 10秒/候选）
- **每个epoch**: 200样本 × 40秒 ≈ **2.2小时**
- **总训练时间（3 epochs）**: ≈ **6-7小时**

## 预期输出

训练完成后会生成：

```
stage3_grpo_output/
├── final_model/              # 最终GRPO模型
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── ...
├── checkpoints/              # 中间检查点
│   ├── checkpoint-20/
│   ├── checkpoint-40/
│   └── ...
├── logs/                     # 训练日志
│   └── grpo_training_YYYYMMDD_HHMMSS.log
└── tensorboard_logs/         # TensorBoard事件
```

## 测试模型

### 测试GRPO模型

```bash
python test_grpo_model.py --model grpo --num_samples 5
```

### 测试SFT模型（对比基线）

```bash
python test_grpo_model.py --model sft --num_samples 5
```

### 对比两个模型

```bash
python test_grpo_model.py --compare --num_samples 10
```

## 预期改进

GRPO训练后应看到以下改进：

| 指标 | SFT基线 | GRPO目标 |
|------|---------|----------|
| 平均奖励 | +20~+50 | +60~+90 |
| 硬约束违反率 | 5-10% | <2% |
| 物理约束满足 | ~90% | >98% |
| 逻辑约束满足 | ~95% | >98% |
| 常识得分 | +2~+5 | +5~+10 |

## 故障排查

### 1. CUDA内存不足

**症状**: CUDA out of memory

**解决**:
```python
# 编辑 train_grpo.py
NUM_SAMPLES_PER_PROMPT = 2  # 从4降到2
```

### 2. 奖励始终为负

**症状**: 所有样本奖励都是-100

**原因**: 硬约束惩罚过重

**解决**: 检查生成的日程格式，调整奖励权重

### 3. 训练不稳定

**症状**: 奖励剧烈波动

**解决**:
```python
# 降低学习率
LEARNING_RATE = 1e-6  # 从5e-6降到1e-6

# 降低温度
TEMPERATURE = 0.7  # 从0.9降到0.7
```

## 检查训练状态

```bash
# 查看screen会话
screen -ls | grep grpo

# 查看GPU使用
nvidia-smi

# 查看进程
ps aux | grep train_grpo

# 查看最新日志（最后50行）
tail -50 stage3_grpo_output/logs/grpo_training_*.log
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `train_grpo.py` | GRPO训练主程序（含奖励模型） |
| `run_grpo_training.sh` | 训练启动脚本 |
| `test_grpo_model.py` | 模型测试和对比工具 |
| `STAGE3_GRPO_README.md` | 详细文档 |
| `STAGE3_QUICKSTART.md` | 本文档 |

## 常用命令速查

```bash
# 启动训练
screen -dmS grpo_train bash run_grpo_training.sh

# 查看日志
tail -f stage3_grpo_output/logs/grpo_training_*.log

# 分离screen
# Ctrl+A then D

# 重连screen
screen -r grpo_train

# 停止训练
screen -S grpo_train -X quit

# TensorBoard
tensorboard --logdir=stage3_grpo_output/tensorboard_logs --port=6008

# 测试模型
python test_grpo_model.py --compare --num_samples 10
```

## 下一步

训练完成后：

1. ✅ 运行对比测试：`python test_grpo_model.py --compare`
2. ✅ 分析奖励改进情况
3. ✅ 检查约束满足率
4. ✅ 如果效果好→部署到生产
5. ✅ 如果效果一般→调整奖励权重重新训练
