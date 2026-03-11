
import os
import sys
import json
import torch

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


MODEL_NAME = "Qwen/Qwen3-8B"  
MODEL_CACHE_DIR = "/data/alice/cjtest/FinalTraj_KDD/finetune/models/Qwen3-8B/Qwen/Qwen3-8B"

TRAIN_DATA_FILE = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage1_training_data_3/sft_training_data.jsonl"

OUTPUT_DIR = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage2_sft_output_epoch10"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
TENSORBOARD_LOG_DIR = os.path.join(OUTPUT_DIR, "tensorboard_logs")

MAX_LENGTH = 4096  
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8 
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10  
LOGGING_STEPS = 10
SAVE_STEPS = 50

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1



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
    
    system_message = example['instruction']
    user_message = example['input']
    assistant_message = example['output']
    
    conversation = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n{assistant_message}<|im_end|>"

    input_ids = tokenizer(conversation, add_special_tokens=False)["input_ids"]
    
    system_part = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    system_tokens = tokenizer(system_part, add_special_tokens=False)["input_ids"]

    labels = [-100] * len(system_tokens) + input_ids[len(system_tokens):]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    

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



def download_model():
    """
    Download Qwen3-8B model from ModelScope if not already cached.
    """
    print("\n Checking model availability...")
    
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
    model_path = download_model()
    
    print(f"\n📖 Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side='right' 
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"  ✓ Tokenizer loaded")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Pad token: {tokenizer.pad_token}")
    print(f"  EOS token: {tokenizer.eos_token}")

    print(f"\n🔧 Loading model in bfloat16 precision...")
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

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    print(f"  ✓ Gradient checkpointing enabled")

    print(f"\n⚙️  Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    
    print(f"  ✓ LoRA applied")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  LoRA rank: {LORA_R}")
    print(f"  LoRA alpha: {LORA_ALPHA}")
    print(f"  LoRA scaling: {LORA_ALPHA/LORA_R}x")
    
    return tokenizer, peft_model


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
    
    tokenizer, model = setup_model_and_tokenizer()
    
    train_dataset = load_and_prepare_data(tokenizer)

    print(f"\n  Configuring training...")
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
    
    try:
        trainer.train()
        
        print(f"\n{'=' * 80}")
        print(f" TRAINING COMPLETE!")
        print(f"{'=' * 80}")
        
        # Save final model
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
    except Exception as e:
        print(f"\n❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """
    Entry point for SFT training.
    """
    if not os.path.exists(TRAIN_DATA_FILE):
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
