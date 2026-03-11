# 第二阶段文件概览

## 创建的文件

### 1. `train_sft.py` ⭐ 主训练脚本
**功能**: 使用 LoRA 方法微调 Qwen3-8B 模型
**输入**: `stage1_training_data/sft_training_data.jsonl`
**输出**: `stage2_sft_output/final_model/` (LoRA 权重)

**关键特性**:
- 自动下载 Qwen2.5-7B-Instruct 模型 (从 ModelScope)
- 使用 LoRA 高效微调 (只训练 ~0.2% 参数)
- 支持 bfloat16 精度训练
- 梯度检查点节省显存
- 自动保存检查点 (每 50 步)

### 2. `download_qwen_model.py`
**功能**: 单独下载 Qwen 模型 (可选)
**用途**: 如果想提前下载模型,避免训练时等待

### 3. `test_sft_model.py`
**功能**: 测试训练好的模型
**用途**: 验证模型能否正确生成 `[THOUGHT]...[/THOUGHT][JSON]...[/JSON]` 格式输出

### 4. `STAGE2_SFT_README.md`
**功能**: 完整使用文档
**内容**: 
- 训练步骤说明
- 配置参数解释
- 硬件要求
- 故障排除

## 快速开始

### 前提条件
确保已完成 Stage 1 数据生成:
```bash
python stage1_data_generation_teacher.py  # 生成 200 条教师数据
python convert_to_sft_format.py           # 转换为 SFT 格式
```

### 运行训练
```bash
# 方式 1: 直接开始训练 (会自动下载模型)
python train_sft.py

# 方式 2: 先下载模型,再训练
python download_qwen_model.py  # 下载 Qwen2.5-7B-Instruct
python train_sft.py            # 开始训练
```

### 测试模型
```bash
python test_sft_model.py
```

## 训练配置摘要

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 基础模型 | Qwen2.5-7B-Instruct | 7B 参数的指令模型 |
| 微调方法 | LoRA | 高效微调,只训练 ~50M 参数 |
| LoRA Rank | 8 | 低秩分解的秩 |
| LoRA Alpha | 32 | 缩放因子 |
| Batch Size | 2 × 8 = 16 | 实际批次大小 |
| Learning Rate | 1e-4 | 学习率 |
| Epochs | 3 | 训练轮数 |
| Max Length | 4096 tokens | 最大序列长度 |
| 精度 | bfloat16 | 半精度训练 |

## 硬件要求

- **GPU**: 至少 24GB 显存 (推荐 RTX 3090/4090 或 A100)
- **RAM**: 32GB+
- **存储**: ~20GB (模型 + 检查点)

如果显存不足,可以:
- 减小 `BATCH_SIZE` 到 1
- 减小 `MAX_LENGTH` 到 2048

## 预期训练时间

以 200 条训练样本为例:
- **步数**: ~200 / 16 = 13 步/epoch
- **总步数**: 13 × 3 = 39 步
- **时间**: 约 15-20 分钟 (RTX 4090)

## 输出文件结构

```
stage2_sft_output/
├── checkpoints/
│   ├── checkpoint-50/          # 第 50 步检查点
│   │   ├── adapter_config.json
│   │   ├── adapter_model.bin   # LoRA 权重
│   │   └── ...
│   ├── checkpoint-100/
│   └── checkpoint-150/
├── final_model/                # 最终模型 ⭐
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
└── logs/                       # 训练日志
    └── events.out.tfevents.*
```

## 下一步

训练完成后:
1. ✅ 使用 `test_sft_model.py` 测试模型
2. ✅ 检查输出格式是否正确
3. ➡️  继续到 Stage 3: 奖励模型训练
4. ➡️  继续到 Stage 4: GRPO 强化学习

## 参考资料

- **LoRA 原理**: https://arxiv.org/abs/2106.09685
- **Qwen 模型**: https://github.com/QwenLM/Qwen
- **PEFT 库**: https://github.com/huggingface/peft
- **参考教程**: https://github.com/datawhalechina/self-llm
