# Stage 3: GRPO Training - Setup and Documentation

## Overview

第三阶段使用规则引导的强化学习（Rule-Guided GRPO）来优化Editor Agent，使其生成的日程不仅"看起来像"，而且在逻辑上"绝对正确"。

## What is GRPO?

GRPO (Group Relative Policy Optimization) 是一种强化学习算法，通过以下方式优化模型：

1. **批量生成** (Group Generation): 对每个输入生成多个候选输出（如4个）
2. **相对评分** (Relative Scoring): 使用奖励模型对批次内的候选进行评分
3. **策略优化** (Policy Optimization): 增加高分候选的生成概率，降低低分候选的概率

## Key Differences from SFT

| 维度 | SFT (Stage 2) | GRPO (Stage 3) |
|------|---------------|----------------|
| 训练目标 | 模仿ground truth | 最大化奖励信号 |
| 是否需要标注 | 需要teacher数据 | 不需要ground truth |
| 奖励来源 | 交叉熵损失 | 程序化规则检查 |
| 约束满足 | 隐式学习 | 显式优化 |

## Reward Model Architecture

奖励模型基于六类约束构建：

### A. 硬约束 (Hard Constraints) - 决定生死

**违反任一条 → -100分**

1. **物理约束** (Physical)
   - ✓ 无时间重叠
   - ✓ end_time > start_time
   - ✓ 总时长 = 24小时

2. **逻辑约束** (Logical)
   - ✓ 第一个活动从home开始（或00:00开始）
   - ✓ 最后一个活动在home结束（或24:00结束）

### B. 软约束 (Soft Constraints) - 决定优劣

**精细化打分，引导模型生成更真实的日程**

3. **常识约束** (Commonsense): -10 ~ +5分
   - 学生 → 不应有"工作"活动
   - 儿童（<18岁）→ 不应有"驾驶"活动
   - 在职人员 → 应有工作活动

4. **社会经济一致性** (Socioeconomic): -3 ~ +3分
   - 蓝领工人 → 高档餐厅用餐不应过频
   - 专业人士 → 可有休闲娱乐活动

5. **时间节律** (Temporal Rhythm): -3 ~ +5分
   - 工作 → 应在8:00-18:00
   - 夜间（22:00-06:00）→ 应在家/睡觉

6. **内部一致性** (Internal Consistency): -10 ~ +5分
   - work_from_home=Yes → 工作活动不应有通勤
   - work_schedule=Regular daytime → 工作应在白天

### Total Reward Calculation

```
R_total = (w_hard × R_hard) + (w_soft × R_soft)

where:
  w_hard = 1.0  (硬约束权重)
  w_soft = 0.3  (软约束权重)
  
  R_hard = R_physical + R_logical
  R_soft = R_commonsense + R_socioeconomic + R_temporal + R_internal
```

## GRPO Training Loop

```
for each training iteration:
    1. 输入: 用户画像 {user_profile}
    
    2. 生成: 模型生成4个候选方案
       - 候选1: [THOUGHT]...[/THOUGHT][JSON]schedule_1[/JSON]
       - 候选2: [THOUGHT]...[/THOUGHT][JSON]schedule_2[/JSON]
       - 候选3: [THOUGHT]...[/THOUGHT][JSON]schedule_3[/JSON]
       - 候选4: [THOUGHT]...[/THOUGHT][JSON]schedule_4[/JSON]
    
    3. 打分: 奖励模型计算每个候选的分数
       - 候选1得分: +85
       - 候选2得分: -95 (违反硬约束)
       - 候选3得分: +92 (最高分)
       - 候选4得分: +70
    
    4. 更新: GRPO算法更新模型权重
       - 增加生成候选3的概率
       - 降低生成候选2的概率
    
    5. 重复: 模型逐渐学会生成高奖励的日程
```

## Dependencies

### Required Packages

```bash
# Core dependencies (already installed)
torch==2.2.1
transformers==5.0.0
peft==0.18.1

# New for GRPO
trl>=0.13.0          # Transformer Reinforcement Learning library
```

### Installation

```bash
conda activate trajlla
pip install trl
```

## File Structure

```
stage3_grpo/
├── train_grpo.py                    # Main GRPO training script
├── run_grpo_training.sh             # Training launcher
├── test_grpo_model.py               # Model testing & comparison
├── STAGE3_GRPO_README.md            # This file
└── stage3_grpo_output/              # Output directory
    ├── checkpoints/                 # Intermediate checkpoints
    ├── final_model/                 # Final trained model
    ├── logs/                        # Training logs
    └── tensorboard_logs/            # TensorBoard events
```

## Configuration

### Model Paths

```python
BASE_MODEL_PATH = "models/Qwen3-8B/Qwen/Qwen3-8B"
SFT_MODEL_PATH = "stage2_sft_output_epoch10/final_model"  # Start from SFT
OUTPUT_DIR = "stage3_grpo_output"
```

### GRPO Hyperparameters

```python
BATCH_SIZE = 1                      # Small batch for RL
NUM_SAMPLES_PER_PROMPT = 4          # Generate 4 candidates per input
LEARNING_RATE = 5e-6                # Lower LR for fine-tuning
NUM_EPOCHS = 3                      # RL converges faster
TEMPERATURE = 0.9                   # Sampling diversity
TOP_P = 0.95                        # Nucleus sampling
```

### Reward Weights

```python
WEIGHT_HARD = 1.0                   # Hard constraints critical
WEIGHT_SOFT = 0.3                   # Soft constraints for refinement
```

## Usage

### 1. Install Dependencies

```bash
pip install trl
```

### 2. Run Training

```bash
# Option A: Direct execution
bash run_grpo_training.sh

# Option B: Screen session (recommended)
screen -S grpo_train bash run_grpo_training.sh

# Option C: Background
screen -dmS grpo_train bash run_grpo_training.sh
```

### 3. Monitor Training

```bash
# View log
tail -f stage3_grpo_output/logs/grpo_training_*.log

# TensorBoard
tensorboard --logdir=stage3_grpo_output/tensorboard_logs --port=6008
```

### 4. Test Model

```bash
# Test GRPO model only
python test_grpo_model.py --model grpo --num_samples 5

# Test SFT model only
python test_grpo_model.py --model sft --num_samples 5

# Compare both models
python test_grpo_model.py --compare --num_samples 10
```

## Expected Results

### Training Metrics

- **Reward Progression**: 应逐渐上升（从负值→正值→高正值）
- **Hard Constraint Violations**: 应逐渐降至0
- **Soft Constraint Scores**: 应逐渐提升

### Comparison Metrics

| Metric | SFT Model | GRPO Model | Expected Change |
|--------|-----------|------------|-----------------|
| Valid Format | ~95% | >95% | → |
| Avg Reward | +20~+50 | +60~+90 | ↑ 50-100% |
| Hard Violations | 5-10% | <2% | ↓ 60-80% |
| Commonsense | +2~+5 | +5~+10 | ↑ 2x |

## Training Time Estimation

- **Hardware**: 1x H200 (144GB)
- **Batch Size**: 1
- **Samples**: 200
- **Candidates per sample**: 4
- **Generation time**: ~10s per candidate
- **Total time per epoch**: ~200 × 4 × 10s ≈ 2-3 hours
- **Total training (3 epochs)**: ~6-9 hours

## Troubleshooting

### Issue 1: trl not found

```bash
pip install trl
```

### Issue 2: CUDA OOM

```python
# Reduce candidates per prompt
NUM_SAMPLES_PER_PROMPT = 2  # From 4 to 2

# Or use gradient checkpointing
model.gradient_checkpointing_enable()
```

### Issue 3: Reward always negative

- Check reward model logic
- Adjust soft constraint weights
- Lower hard constraint penalties (-100 → -50)

### Issue 4: Training unstable

- Lower learning rate (5e-6 → 1e-6)
- Reduce temperature (0.9 → 0.7)
- Increase batch size

## Next Steps

After GRPO training completes:

1. **Model Evaluation**: Compare SFT vs GRPO performance
2. **Error Analysis**: Identify remaining constraint violations
3. **Hyperparameter Tuning**: Adjust reward weights if needed
4. **Production Deployment**: Use GRPO model for inference
5. **Iterative Improvement**: Collect edge cases, refine reward model

## References

- [TRL Library Documentation](https://huggingface.co/docs/trl)
- [GRPO Algorithm Paper](https://arxiv.org/abs/2402.03300)
- [Qwen3 GRPO Tutorial](https://github.com/datawhalechina/self-llm/blob/master/models/Qwen3/10-Qwen3-8B%20GRPO%E5%BE%AE%E8%B0%83%E5%8F%8A%E9%80%9A%E8%BF%87swanlab%E5%8F%AF%E8%A7%86%E5%8C%96.md)
