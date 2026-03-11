# Stage 2: Supervised Fine-Tuning (SFT) - 使用说明

## 概述

本阶段使用 Stage 1 生成的教师模型 CoT 数据，对学生模型（Qwen3-8B）进行监督微调，使其学会约束检查和编辑操作。

## 训练目标

让学生模型学会:
1. **约束检查**: 检查 5 种约束类型 (Physical, Logical, Common Sense, Temporal, Coherence)
2. **识别违规**: 找出初始日程中的问题
3. **应用编辑**: 使用 DELETE/ADD/SHIFT/REPLACE 操作修复问题
4. **输出格式**: 生成 `[THOUGHT]...[/THOUGHT][JSON]...[/JSON]` 格式的输出

## 数据格式

训练数据格式 (来自 `sft_training_data.jsonl`):

```json
{
  "instruction": "You are a Critic & Editor Agent...",
  "input": "**PERSON PROFILE:**\n{profile}\n\n**INITIAL SCHEDULE:**\n{schedule}",
  "output": "[THOUGHT]...[/THOUGHT][JSON]...[/JSON]"
}
```

## 使用步骤

### 1. 准备训练数据

首先确保已运行 Stage 1 数据生成:

```bash
# 运行 Stage 1 数据生成 (如果还没运行)
python stage1_data_generation_teacher.py

# 转换为 SFT 格式
python convert_to_sft_format.py
```

这会生成 `stage1_training_data/sft_training_data.jsonl`

### 2. (可选) 下载模型

如果模型还未下载，可以先单独下载:

```bash
python download_qwen_model.py
```

或者直接运行训练脚本，它会自动下载。

### 3. 运行 SFT 训练

```bash
python train_sft.py
```

训练过程会显示:
- 数据加载进度
- 模型下载/加载状态
- LoRA 配置信息
- 训练进度和损失

### 4. 训练输出

训练完成后，输出文件位于:

```
stage2_sft_output/
├── checkpoints/
│   ├── checkpoint-50/
│   ├── checkpoint-100/
│   └── checkpoint-150/
├── final_model/          # 最终模型
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
└── logs/                 # 训练日志
```

## 训练配置

### 模型设置
- **基础模型**: Qwen2.5-7B-Instruct
- **微调方法**: LoRA (Low-Rank Adaptation)
- **精度**: bfloat16

### LoRA 参数
- **Rank (r)**: 8
- **Alpha**: 32
- **Dropout**: 0.1
- **Scaling**: 4x (alpha/r = 32/8)
- **目标模块**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### 训练超参数
- **Batch Size**: 2 (per device)
- **Gradient Accumulation**: 8 steps
- **Effective Batch Size**: 16
- **Learning Rate**: 1e-4
- **Epochs**: 3
- **Max Length**: 4096 tokens
- **Warmup Steps**: 50
- **Weight Decay**: 0.01

### 硬件要求
- **GPU**: 至少 24GB 显存 (RTX 3090/4090, A100, etc.)
- **RAM**: 32GB+
- **存储**: ~20GB (模型 + 检查点)

## 预期结果

训练完成后，模型应该能够:
- ✅ 接收人员画像 + 初始日程
- ✅ 输出思考过程 `[THOUGHT]...[/THOUGHT]`
- ✅ 输出精炼后的日程 `[JSON]...[/JSON]`
- ✅ 匹配教师模型的推理风格

## 故障排除

### 1. CUDA Out of Memory
- 减小 `BATCH_SIZE` (改为 1)
- 减小 `MAX_LENGTH` (改为 2048)
- 启用梯度检查点 (已默认开启)

### 2. 训练数据未找到
```bash
# 确保已生成 SFT 格式数据
python convert_to_sft_format.py
```

### 3. 模型下载失败
- 检查网络连接
- 确保 ModelScope 可访问
- 手动下载后放到 `MODEL_CACHE_DIR`

### 4. Tokenization 错误
- 检查 `sft_training_data.jsonl` 格式是否正确
- 确保每行都是有效的 JSON

## 下一步

训练完成后，继续到 Stage 3:
- **Stage 3**: 奖励模型训练 (Reward Model)
- **Stage 4**: GRPO 强化学习优化

## 参考资料

- [Qwen 官方文档](https://github.com/QwenLM/Qwen)
- [LoRA 原理](https://arxiv.org/abs/2106.09685)
- [PEFT 库](https://github.com/huggingface/peft)
- [Transformers 文档](https://huggingface.co/docs/transformers)
