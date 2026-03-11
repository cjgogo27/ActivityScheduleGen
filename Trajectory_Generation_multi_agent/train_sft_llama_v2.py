"""
Stage 2 v2: Supervised Fine-Tuning (SFT) — LLaMA-3.1-8B-Instruct
调整参数：NUM_EPOCHS 20, LoRA r=16 / alpha=64
"""

import os
import sys
import json
import torch

# Patch for transformers 5.0.0 + torch 2.2.1 compatibility issue
if hasattr(torch, 'is_autocast_enabled'):
    _original_is_autocast_enabled = torch.is_autocast_enabled
    def _patched_is_autocast_enabled(device_type=None):
        if device_type is None:
            return _original_is_autocast_enabled()
        return _original_is_autocast_enabled()
    torch.is_autocast_enabled = _patched_is_autocast_enabled

from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, TaskType, get_peft_model

# ==================== Configuration ====================

MODEL_PATH = "/data/alice/cjtest/FinalTraj_arr/finetune/models/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct"

TRAIN_DATA_FILE = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage1_training_data_3/sft_training_data.jsonl"

# ── v2 输出路径 ──
OUTPUT_DIR          = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage2_sft_llama_v2_epoch20"
CHECKPOINT_DIR      = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR             = os.path.join(OUTPUT_DIR, "logs")
TENSORBOARD_LOG_DIR = os.path.join(OUTPUT_DIR, "tensorboard_logs")

# ── v2 超参调整 ──
MAX_LENGTH                 = 4096
BATCH_SIZE                 = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE              = 1e-4
NUM_EPOCHS                 = 20   # v1: 10  →  v2: 20
LOGGING_STEPS              = 10
SAVE_STEPS                 = 100

# ── v2 LoRA 调整 ──
LORA_R       = 16   # v1: 8  →  v2: 16
LORA_ALPHA   = 64   # 保持 alpha = 4*r
LORA_DROPOUT = 0.05 # 稍微降低 dropout，更大 rank 时过拟合风险本身较低


# ==================== Data Processing ====================

def process_func(example, tokenizer):
    """LLaMA-3 BOS + header chat format."""
    system_message    = example['instruction']
    user_message      = example['input']
    assistant_message = example['output']

    conversation = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_message}<|eot_id|>"
    )

    prompt_prefix = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    input_ids     = tokenizer(conversation,  add_special_tokens=False)["input_ids"]
    prompt_tokens = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]

    labels = [-100] * len(prompt_tokens) + input_ids[len(prompt_tokens):]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels    = labels[:MAX_LENGTH]

    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels":         labels
    }


def load_and_prepare_data(tokenizer):
    print(f"\n📂 Loading training data from: {TRAIN_DATA_FILE}")
    dataset = load_dataset('json', data_files=TRAIN_DATA_FILE, split='train')
    print(f"  ✓ Loaded {len(dataset)} training samples")

    print("\n🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: process_func(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    avg_len = sum(len(x['input_ids']) for x in tokenized_dataset) / len(tokenized_dataset)
    print(f"  ✓ Done. Average token length: {avg_len:.0f}")
    return tokenized_dataset


# ==================== Model Setup ====================

def setup_model_and_tokenizer():
    print(f"\n📖 Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, padding_side='right')
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  ✓ Tokenizer loaded (vocab: {len(tokenizer)}, pad: {tokenizer.pad_token})")

    print(f"\n🔧 Loading model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    print(f"\n⚙️  Configuring LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        inference_mode = False,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT
    )
    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in peft_model.parameters())
    print(f"  ✓ Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    return tokenizer, peft_model


# ==================== Training ====================

def train():
    print("=" * 80)
    print(" STAGE 2 v2: SFT - LLaMA-3.1-8B-Instruct")
    print(f" Epochs: {NUM_EPOCHS}  |  LoRA r={LORA_R}  |  alpha={LORA_ALPHA}")
    print("=" * 80)

    tokenizer, model = setup_model_and_tokenizer()
    train_dataset    = load_and_prepare_data(tokenizer)

    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs            = NUM_EPOCHS,
        learning_rate               = LEARNING_RATE,
        logging_dir                 = TENSORBOARD_LOG_DIR,
        logging_steps               = LOGGING_STEPS,
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 3,
        fp16                        = False,
        bf16                        = True,
        gradient_checkpointing      = True,
        optim                       = "adamw_torch",
        warmup_steps                = 50,
        weight_decay                = 0.01,
        report_to                   = "tensorboard",
        save_on_each_node           = True,
        remove_unused_columns       = False,
    )

    trainer = Trainer(
        model          = model,
        args           = training_args,
        train_dataset  = train_dataset,
        data_collator  = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    print(f"\n🚀 Starting training...")
    trainer.train()

    final_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Final model saved to: {final_path}")


def main():
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"❌ Training data not found: {TRAIN_DATA_FILE}")
        sys.exit(1)
    os.makedirs(OUTPUT_DIR,          exist_ok=True)
    os.makedirs(CHECKPOINT_DIR,      exist_ok=True)
    os.makedirs(LOG_DIR,             exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    train()


if __name__ == "__main__":
    main()
