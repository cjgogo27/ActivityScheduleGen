"""
使用LoRA方法微调Llama 3.1模型用于轨迹生成任务
基于加州真实轨迹数据进行指令微调
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============ 自定义训练回调 ============
class MetricsCallback(TrainerCallback):
    """记录详细的训练指标"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次记录日志时调用"""
        if logs:
            step = state.global_step
            logger.info(f"Step {step}: {logs}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """每个epoch结束时调用"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {state.epoch} 完成!")
        logger.info(f"{'='*70}\n")

# ============ 配置参数 ============
MODEL_PATH = "/data/mayue/cjy/Other_method/FinalTraj/finetune/Llama/LLM-Research/Meta-Llama-3___1-8B-Instruct"
DATASET_PATH = "/data/mayue/cjy/Other_method/FinalTraj/finetune/trajectory_instruction_dataset.json"
OUTPUT_DIR = "/data/mayue/cjy/Other_method/FinalTraj/finetune/output/llama3_1_trajectory_lora"

# 针对小数据集(2679样本)优化的超参数
MAX_LENGTH = 2048  # Llama分词器会将中文切分为多个token,需要较大长度
LEARNING_RATE = 5e-5  # 降低学习率,避免过拟合
NUM_EPOCHS = 5  # 增加epoch数,小数据集需要多轮训练
BATCH_SIZE = 4  # L40显存充足,增加batch size提升稳定性
GRADIENT_ACCUMULATION_STEPS = 4  # 有效batch=16,保持不变
WARMUP_RATIO = 0.15  # 增加warmup,让模型平滑学习

# LoRA配置 - 针对小数据集适度增加容量
LORA_R = 16  # 增加rank,提升表达能力
LORA_ALPHA = 32  # 保持alpha=2*r,标准配置
LORA_DROPOUT = 0.05  # 降低dropout,小数据集不需要太强正则化

# 可视化配置
USE_TENSORBOARD = True  # 启用TensorBoard
USE_WANDB = False  # 关闭Weights & Biases

# Resume配置
RESUME_FROM_CHECKPOINT = None  # 设置为checkpoint路径以恢复训练,如: "./output/llama3_1_trajectory_lora/checkpoint-100"
AUTO_RESUME = True  # 自动从最新checkpoint恢复

# ============ 数据处理函数 ============

def load_dataset_from_json(file_path: str) -> Dataset:
    """加载JSON格式的指令数据集"""
    logger.info(f"正在加载数据集: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"  加载了 {len(data)} 条训练样本")
    
    # 转换为HuggingFace Dataset
    dataset = Dataset.from_list(data)
    return dataset


def process_func(example, tokenizer):
    """
    处理单个样本,将instruction-input-output格式转换为模型训练格式
    
    LLaMA3.1的Prompt Template格式:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    {user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    
    {assistant_response}<|eot_id|>
    """
    
    input_ids, attention_mask, labels = [], [], []
    
    # 构建系统提示
    system_prompt = "You are an AI assistant specialized in generating realistic daily activity schedules and trajectories for household members. You understand human behavior patterns, time constraints, and family dynamics."
    
    # 构建用户消息
    user_message = example['instruction']
    if example.get('input', '').strip():
        user_message += f"\n\n{example['input']}"
    
    # 构建助手回复
    assistant_response = example['output']
    
    # 使用Llama3.1的格式
    instruction_text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    response_text = f"{assistant_response}<|eot_id|>"
    
    # Tokenize
    instruction = tokenizer(instruction_text, add_special_tokens=False)
    response = tokenizer(response_text, add_special_tokens=False)
    
    # 组合input_ids和labels
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    
    # labels: instruction部分用-100屏蔽,只计算response部分的loss
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 截断到最大长度
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# ============ 主训练流程 ============

def main():
    """主训练函数"""
    
    logger.info("=" * 70)
    logger.info("Llama 3.1 轨迹生成模型 LoRA 微调")
    logger.info("=" * 70)
    
    # 1. 加载Tokenizer (按照参考文档的方式)
    logger.info(f"\n[1/6] 加载Tokenizer: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,  # 不使用fast tokenizer
        trust_remote_code=True  # 信任远程代码
    )
    
    # 设置pad_token (Llama3.1使用eos_token作为pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"  ✓ Tokenizer加载完成")
    logger.info(f"  词表大小: {len(tokenizer)}")
    logger.info(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # 2. 加载数据集
    logger.info(f"\n[2/6] 加载训练数据集")
    dataset = load_dataset_from_json(DATASET_PATH)
    
    # 数据预处理
    logger.info(f"\n[3/6] 数据预处理 (tokenization)")
    tokenized_dataset = dataset.map(
        lambda example: process_func(example, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    logger.info(f"  预处理完成: {len(tokenized_dataset)} 条样本")
    
    # 3. 加载基础模型 (按照参考文档的方式)
    logger.info(f"\n[4/6] 加载基础模型: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",  # 自动分配设备
        torch_dtype=torch.bfloat16,  # L40支持bfloat16
        trust_remote_code=True  # 信任远程代码
    )
    
    # 启用梯度检查点以节省显存 (gradient_checkpointing开启时必须执行)
    model.enable_input_require_grads()
    logger.info(f"  ✓ 模型加载完成")
    logger.info(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    logger.info(f"  数据类型: {model.dtype}")
    
    # 4. 配置LoRA (按照参考文档的方式)
    logger.info(f"\n[5/6] 配置LoRA")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言模型
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 需要训练的层
        inference_mode=False,  # 训练模式
        r=LORA_R,  # LoRA秩
        lora_alpha=LORA_ALPHA,  # LoRA alpha (缩放 = alpha/r = 32/16 = 2)
        lora_dropout=LORA_DROPOUT  # Dropout比例
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info(f"  ✓ LoRA配置完成")
    logger.info(f"  LoRA秩: {LORA_R}, Alpha: {LORA_ALPHA}, 缩放: {LORA_ALPHA/LORA_R}")
    logger.info(f"  Dropout: {LORA_DROPOUT}")
    
    # 5. 配置训练参数
    logger.info(f"\n[6/6] 配置训练参数")
    
    # 计算总训练步数
    total_steps = (len(tokenized_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    # 配置可视化工具
    report_to = []
    if USE_TENSORBOARD:
        report_to.append("tensorboard")
        logger.info(f"  ✓ TensorBoard已启用: {OUTPUT_DIR}/runs")
    if USE_WANDB:
        report_to.append("wandb")
        logger.info(f"  ✓ Weights & Biases已启用")
    else:
        logger.info(f"  ✗ Weights & Biases已关闭")
    if not report_to:
        report_to = ["none"]
    
    # 按照参考文档配置TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        
        # 日志和保存策略
        logging_steps=10,  # 每10步记录一次
        logging_first_step=True,
        save_steps=100,  # 每100步保存一次checkpoint
        save_total_limit=5,  # 保留最近5个检查点
        save_on_each_node=True,
        
        # 学习率调度
        lr_scheduler_type="cosine",  # cosine退火
        warmup_steps=warmup_steps,
        
        # 精度和优化
        bf16=True,  # 使用bfloat16训练
        gradient_checkpointing=True,  # 梯度检查点(必须配合model.enable_input_require_grads())
        optim="adamw_torch",  # 优化器
        weight_decay=0.01,  # 权重衰减
        max_grad_norm=1.0,  # 梯度裁剪
        
        # 可视化和日志
        logging_dir=f"{OUTPUT_DIR}/runs",
        report_to=report_to,
        
        # Resume相关
        save_strategy="steps",  # 按步数保存
        resume_from_checkpoint=RESUME_FROM_CHECKPOINT,  # 从指定checkpoint恢复
        
        # 其他
        remove_unused_columns=False,
        load_best_model_at_end=False,
        dataloader_num_workers=0,  # 数据加载线程数
    )
    
    # 记录超参数到TensorBoard
    if USE_TENSORBOARD:
        import tensorboard
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"{OUTPUT_DIR}/runs")
        hparams = {
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'gradient_accumulation': GRADIENT_ACCUMULATION_STEPS,
            'effective_batch_size': BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            'lora_r': LORA_R,
            'lora_alpha': LORA_ALPHA,
            'lora_dropout': LORA_DROPOUT,
            'max_length': MAX_LENGTH,
            'warmup_ratio': WARMUP_RATIO,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'dataset_size': len(tokenized_dataset),
            'total_steps': total_steps,
        }
        writer.add_hparams(hparams, {})
        writer.close()
        logger.info(f"  ✓ 超参数已记录到TensorBoard")
    
    logger.info(f"  输出目录: {OUTPUT_DIR}")
    logger.info(f"  总epoch数: {NUM_EPOCHS}")
    logger.info(f"  批大小: {BATCH_SIZE}")
    logger.info(f"  梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  有效批大小: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  学习率: {LEARNING_RATE}")
    logger.info(f"  总训练步数: {total_steps}")
    logger.info(f"  Warmup步数: {warmup_steps}")
    logger.info(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}, dropout: {LORA_DROPOUT}")
    
    # 6. 创建Trainer并开始训练
    logger.info(f"\n开始训练...")
    logger.info("=" * 70)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[MetricsCallback()],  # 添加自定义回调
    )
    
    # 检查是否自动恢复训练
    resume_checkpoint = None
    if AUTO_RESUME and os.path.exists(OUTPUT_DIR):
        # 查找最新的checkpoint
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')]
        if checkpoints:
            # 按checkpoint编号排序
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            resume_checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            logger.info(f"\n{'='*70}")
            logger.info(f"检测到checkpoint: {resume_checkpoint}")
            logger.info(f"将从此checkpoint恢复训练...")
            logger.info(f"{'='*70}\n")
    elif RESUME_FROM_CHECKPOINT:
        resume_checkpoint = RESUME_FROM_CHECKPOINT
        logger.info(f"\n{'='*70}")
        logger.info(f"从指定checkpoint恢复: {resume_checkpoint}")
        logger.info(f"{'='*70}\n")
    
    # 开始训练 (如果有checkpoint则恢复)
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    
    # 7. 保存最终模型
    logger.info(f"\n保存最终模型到: {OUTPUT_DIR}/final")
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    
    logger.info("\n" + "=" * 70)
    logger.info("训练完成!")
    logger.info("=" * 70)
    logger.info(f"\n模型保存位置:")
    logger.info(f"  最终模型: {OUTPUT_DIR}/final")
    logger.info(f"  检查点: {OUTPUT_DIR}/checkpoint-*")
    logger.info(f"\n使用示例:")
    logger.info(f"  from peft import PeftModel")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{MODEL_PATH}')")
    logger.info(f"  model = PeftModel.from_pretrained(model, '{OUTPUT_DIR}/final')")


if __name__ == "__main__":
    main()
