"""
Stage 2: Supervised Fine-Tuning (SFT) - Domain-Adaptive SFT with LoRA
=====================================================================

Purpose: Train the Editor agent to perform constraint-based schedule refinement

Model: LLaMA-3.1-8B-Instruct
Method: LoRA fine-tuning
Training Data: Stage 1 teacher-generated CoT data
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
from typing import Dict, List

# ==================== Configuration ====================

# Model path (local)
MODEL_PATH = "/data/alice/cjtest/FinalTraj_arr/finetune/models/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct"

# Data paths
TRAIN_DATA_FILE = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage1_training_data_3/sft_training_data.jsonl"

# Output paths
OUTPUT_DIR = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage2_sft_llama_output_epoch10"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
TENSORBOARD_LOG_DIR = os.path.join(OUTPUT_DIR, "tensorboard_logs")

# Training hyperparameters
MAX_LENGTH = 4096
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
LOGGING_STEPS = 10
SAVE_STEPS = 50

# LoRA parameters
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


# ==================== Data Processing ====================

def process_func(example, tokenizer):
    """
    Process one training sample for LLaMA-3.1-8B-Instruct model.

    LLaMA 3 uses BOS + header format:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>
    """
    system_message   = example['instruction']
    user_message     = example['input']
    assistant_message = example['output']

    # Full conversation
    conversation = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant_message}<|eot_id|>"
    )

    # Prompt prefix (for label masking — mask everything up to and including the assistant header)
    prompt_prefix = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    # Tokenize (no special tokens added manually; BOS is included in template)
    input_ids     = tokenizer(conversation,   add_special_tokens=False)["input_ids"]
    prompt_tokens = tokenizer(prompt_prefix,  add_special_tokens=False)["input_ids"]

    # Labels: -100 for prompt, actual token IDs for assistant response
    labels = [-100] * len(prompt_tokens) + input_ids[len(prompt_tokens):]

    # Truncate if too long
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels    = labels[:MAX_LENGTH]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels
    }


def load_and_prepare_data(tokenizer):
    """Load SFT training data and prepare for training."""
    print(f"\n📂 Loading training data from: {TRAIN_DATA_FILE}")
    dataset = load_dataset('json', data_files=TRAIN_DATA_FILE, split='train')
    print(f"  ✓ Loaded {len(dataset)} training samples")

    print("\n📝 Example training sample:")
    print(f"  Instruction: {dataset[0]['instruction'][:100]}...")
    print(f"  Input length: {len(dataset[0]['input'])} chars")
    print(f"  Output length: {len(dataset[0]['output'])} chars")

    print("\n🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: process_func(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing training data"
    )

    print(f"  ✓ Tokenization complete")
    print(f"  Average input length: {sum(len(x['input_ids']) for x in tokenized_dataset) / len(tokenized_dataset):.0f} tokens")

    return tokenized_dataset


# ==================== Model Setup ====================

def setup_model_and_tokenizer():
    """Load tokenizer and model, apply LoRA configuration."""
    print(f"\n📖 Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        padding_side='right'
    )

    # LLaMA-3.1 doesn't have a dedicated pad token; use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  ✓ Tokenizer loaded")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Pad token: {tokenizer.pad_token}")
    print(f"  EOS token: {tokenizer.eos_token}")

    print(f"\n🔧 Loading model in bfloat16 precision...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"  ✓ Model loaded")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    print(f"  ✓ Gradient checkpointing enabled")

    print(f"\n⚙️  Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )

    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in peft_model.parameters())

    print(f"  ✓ LoRA applied")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}, scaling: {LORA_ALPHA/LORA_R}x")

    return tokenizer, peft_model


# ==================== Training ====================

def train():
    print("=" * 80)
    print(" STAGE 2: SUPERVISED FINE-TUNING (SFT) - EDITOR AGENT")
    print(" Model: LLaMA-3.1-8B-Instruct")
    print(" Method: LoRA fine-tuning")
    print(" Task: Constraint-based schedule refinement with CoT")
    print("=" * 80)

    tokenizer, model = setup_model_and_tokenizer()
    train_dataset = load_and_prepare_data(tokenizer)

    print(f"\n🏋️  Configuring training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_dir=TENSORBOARD_LOG_DIR,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_steps=50,
        weight_decay=0.01,
        report_to="tensorboard",
        save_on_each_node=True,
        remove_unused_columns=False,
    )

    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  TensorBoard logs: {TENSORBOARD_LOG_DIR}")
    print(f"  Batch size: {BATCH_SIZE} x grad_accum {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )

    print(f"\n{'=' * 80}")
    print(f" STARTING TRAINING")
    print(f"{'=' * 80}\n")

    try:
        trainer.train()

        print(f"\n{'=' * 80}")
        print(f" TRAINING COMPLETE!")
        print(f"{'=' * 80}")

        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)

        print(f"\n✅ Final model saved to: {final_model_path}")

    except Exception as e:
        print(f"\n❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ==================== Main ====================

def main():
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"❌ Training data not found: {TRAIN_DATA_FILE}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

    train()


if __name__ == "__main__":
    main()
