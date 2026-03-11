"""
Stage 2: Supervised Fine-Tuning (SFT) - Domain-Adaptive SFT with LoRA
=====================================================================

Purpose: Train the Editor agent to perform constraint-based schedule refinement

Training Objective:
- Learn to check 5 constraint types (Physical, Logical, Common Sense, Temporal, Coherence)
- Learn to identify violations
- Learn to apply edit operations (DELETE/ADD/SHIFT/REPLACE)
- Output format: [THOUGHT]...[/THOUGHT][JSON]...[/JSON]

Model: Qwen3-8B (from ModelScope)
Method: LoRA fine-tuning
Training Data: Stage 1 teacher-generated CoT data

References:
- https://github.com/datawhalechina/self-llm (Llama3.1 Lora tutorial)
- Uses transformers, peft, datasets
"""

import os
import sys
import json
import torch

# Patch for transformers 5.0.0 + torch 2.2.1 compatibility issue
# Fix torch.is_autocast_enabled() signature mismatch
if hasattr(torch, 'is_autocast_enabled'):
    _original_is_autocast_enabled = torch.is_autocast_enabled
    def _patched_is_autocast_enabled(device_type=None):
        if device_type is None:
            return _original_is_autocast_enabled()
        # torch 2.2.1 doesn't support device_type parameter
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

# Model paths
MODEL_NAME = "Qwen/Qwen3-8B"  # Will download from ModelScope
MODEL_CACHE_DIR = "/data/alice/cjtest/FinalTraj_KDD/finetune/models/Qwen3-8B/Qwen/Qwen3-8B"

# Data paths
TRAIN_DATA_FILE = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage1_training_data_3/sft_training_data.jsonl"

# Output paths - Epoch 10 training run
OUTPUT_DIR = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage2_sft_output_epoch10"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
TENSORBOARD_LOG_DIR = os.path.join(OUTPUT_DIR, "tensorboard_logs")

# Training hyperparameters
MAX_LENGTH = 4096  # Qwen can handle longer context; our CoT + JSON is ~2000-3000 tokens
BATCH_SIZE = 2  # Small batch due to long sequences
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = 2 * 8 = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10  # Increased from 3 to 10 for better convergence
LOGGING_STEPS = 10
SAVE_STEPS = 50

# LoRA parameters
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


# ==================== Data Processing ====================

def process_func(example, tokenizer):
    """
    Process one training sample for Qwen model.
    
    Qwen3 uses ChatML format:
    <|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_message}<|im_end|>
    
    Input format (from convert_to_sft_format.py):
    {
        "instruction": "You are a Critic & Editor Agent...",
        "input": "**PERSON PROFILE:**\n{profile}\n\n**INITIAL SCHEDULE:**\n{schedule}",
        "output": "[THOUGHT]...[/THOUGHT][JSON]...[/JSON]"
    }
    """
    
    # Construct prompt in Qwen format
    # System message: instruction
    # User message: input (profile + initial schedule)
    # Assistant message: output (CoT + refined schedule)
    
    system_message = example['instruction']
    user_message = example['input']
    assistant_message = example['output']
    
    # Build full conversation using Qwen's ChatML format
    conversation = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{assistant_message}<|im_end|>"
    
    # Tokenize
    input_ids = tokenizer(conversation, add_special_tokens=False)["input_ids"]
    
    # Create labels: mask everything except assistant's response
    # We want to train only on the assistant's output
    system_part = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    system_tokens = tokenizer(system_part, add_special_tokens=False)["input_ids"]
    
    # Labels: -100 for prompt, actual tokens for assistant response
    labels = [-100] * len(system_tokens) + input_ids[len(system_tokens):]
    
    # Truncate if too long
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    # Attention mask (all 1s for valid tokens)
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def load_and_prepare_data(tokenizer):
    """
    Load SFT training data and prepare for training.
    
    Returns:
        Tokenized dataset ready for training
    """
    print(f"\n📂 Loading training data from: {TRAIN_DATA_FILE}")
    
    # Load JSONL file
    dataset = load_dataset('json', data_files=TRAIN_DATA_FILE, split='train')
    print(f"  ✓ Loaded {len(dataset)} training samples")
    
    # Show example
    print("\n📝 Example training sample:")
    print(f"  Instruction: {dataset[0]['instruction'][:100]}...")
    print(f"  Input length: {len(dataset[0]['input'])} chars")
    print(f"  Output length: {len(dataset[0]['output'])} chars")
    
    # Tokenize dataset
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

def download_model():
    """
    Download Qwen3-8B model from ModelScope if not already cached.
    """
    print("\n🤖 Checking model availability...")
    
    if os.path.exists(MODEL_CACHE_DIR) and os.listdir(MODEL_CACHE_DIR):
        print(f"  ✓ Model found in cache: {MODEL_CACHE_DIR}")
        return MODEL_CACHE_DIR
    
    print(f"  ⬇️  Downloading model from ModelScope: {MODEL_NAME}")
    print(f"  📁 Cache directory: {MODEL_CACHE_DIR}")
    
    try:
        from modelscope import snapshot_download
        model_dir = snapshot_download(
            MODEL_NAME,
            cache_dir=MODEL_CACHE_DIR,
            revision='master'
        )
        print(f"  ✓ Model downloaded successfully")
        return model_dir
    except Exception as e:
        print(f"  ❌ Error downloading model: {str(e)}")
        sys.exit(1)


def setup_model_and_tokenizer():
    """
    Load tokenizer and model, apply LoRA configuration.
    
    Returns:
        tokenizer, peft_model
    """
    # Download model if needed
    model_path = download_model()
    
    # Load tokenizer
    print(f"\n📖 Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side='right'  # Important for training
    )
    
    # Qwen uses <|endoftext|> as pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  ✓ Tokenizer loaded")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Pad token: {tokenizer.pad_token}")
    print(f"  EOS token: {tokenizer.eos_token}")
    
    # Load model in half precision
    print(f"\n🔧 Loading model in bfloat16 precision...")
    # Qwen3 requires trust_remote_code for both model and config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print(f"  ✓ Model loaded")
    print(f"  Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    # Enable gradient checkpointing (saves memory)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    print(f"  ✓ Gradient checkpointing enabled")
    
    # Configure LoRA
    print(f"\n⚙️  Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )
    
    # Apply LoRA
    peft_model = get_peft_model(model, lora_config)
    
    # Show trainable parameters
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    
    print(f"  ✓ LoRA applied")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  LoRA rank: {LORA_R}")
    print(f"  LoRA alpha: {LORA_ALPHA}")
    print(f"  LoRA scaling: {LORA_ALPHA/LORA_R}x")
    
    return tokenizer, peft_model


# ==================== Training ====================

def train():
    """
    Main training function.
    """
    print("=" * 80)
    print(" STAGE 2: SUPERVISED FINE-TUNING (SFT) - EDITOR AGENT")
    print(" Model: Qwen3-8B")
    print(" Method: LoRA fine-tuning")
    print(" Task: Constraint-based schedule refinement with CoT")
    print("=" * 80)
    
    # Setup model and tokenizer
    tokenizer, model = setup_model_and_tokenizer()
    
    # Load and prepare data
    train_dataset = load_and_prepare_data(tokenizer)
    
    # Configure training arguments
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
        save_total_limit=3,  # Keep only 3 latest checkpoints
        fp16=False,
        bf16=True,  # Use bfloat16 for better stability
        gradient_checkpointing=True,
        optim="adamw_torch",
        warmup_steps=50,
        weight_decay=0.01,
        report_to="tensorboard",  # Enable tensorboard logging
        save_on_each_node=True,
        remove_unused_columns=False,
    )
    
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  TensorBoard logs: {TENSORBOARD_LOG_DIR}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Total steps: ~{len(train_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)}")
    
    # Initialize trainer
    print(f"\n🎯 Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True
        )
    )
    
    print(f"  ✓ Trainer ready")
    
    # Start training
    print(f"\n{'=' * 80}")
    print(f" STARTING TRAINING")
    print(f"{'=' * 80}\n")
    
    try:
        trainer.train()
        
        print(f"\n{'=' * 80}")
        print(f" TRAINING COMPLETE!")
        print(f"{'=' * 80}")
        
        # Save final model
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        print(f"\n✅ Final model saved to: {final_model_path}")
        print(f"\n📊 Training summary:")
        print(f"  Total samples: {len(train_dataset)}")
        print(f"  Epochs completed: {NUM_EPOCHS}")
        print(f"  Model: Qwen3-8B + LoRA")
        print(f"  LoRA parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
    except Exception as e:
        print(f"\n❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ==================== Main ====================

def main():
    """
    Entry point for SFT training.
    """
    # Check if training data exists
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"❌ Training data not found: {TRAIN_DATA_FILE}")
        print(f"\n💡 Please run convert_to_sft_format.py first to generate training data:")
        print(f"   python convert_to_sft_format.py")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    
    # Run training
    train()


if __name__ == "__main__":
    main()
