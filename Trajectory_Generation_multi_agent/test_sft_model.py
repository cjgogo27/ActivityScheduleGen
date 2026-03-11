"""
Test the trained SFT model (Stage 2)

This script loads the LoRA fine-tuned model and tests it on a sample input.
"""

import os
import sys
import json
import torch

# Patch for transformers 5.0.0 + torch 2.2.1 compatibility
if hasattr(torch, 'is_autocast_enabled'):
    _original_is_autocast_enabled = torch.is_autocast_enabled
    def _patched_is_autocast_enabled(device_type=None):
        if device_type is None:
            return _original_is_autocast_enabled()
        # torch 2.2.1 doesn't support device_type parameter
        return _original_is_autocast_enabled()
    torch.is_autocast_enabled = _patched_is_autocast_enabled

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==================== Configuration ====================

# Paths
BASE_MODEL_PATH = "/data/alice/cjtest/FinalTraj_KDD/finetune/models/Qwen3-8B/Qwen/Qwen3-8B"
LORA_MODEL_PATH = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage2_sft_output/final_model"

# Or specify a checkpoint
# LORA_MODEL_PATH = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage2_sft_output/checkpoint-100"

# Test sample (you can modify this)
TEST_PROFILE = {
    "user_id": "test_001",
    "age_range": "35-44",
    "gender": "Male",
    "employment_status": "Full-time employed",
    "occupation": "Software Engineer",
    "work_schedule": "Regular daytime",
    "work_from_home": "No"
}

TEST_INITIAL_SCHEDULE = [
    {"activity": "home", "start_time": "00:00", "end_time": "08:00", "duration": 8.0},
    {"activity": "work", "start_time": "08:00", "end_time": "17:00", "duration": 9.0},
    {"activity": "shopping", "start_time": "17:00", "end_time": "18:00", "duration": 1.0},
    {"activity": "home", "start_time": "18:00", "end_time": "24:00", "duration": 6.0}
]

# ==================== Model Loading ====================

def load_model():
    """Load base model + LoRA weights"""
    
    print("=" * 80)
    print(" LOADING TRAINED SFT MODEL")
    print("=" * 80)
    
    # Check if paths exist
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"❌ Base model not found: {BASE_MODEL_PATH}")
        print("Please download the model first using download_qwen_model.py")
        sys.exit(1)
    
    if not os.path.exists(LORA_MODEL_PATH):
        print(f"❌ LoRA model not found: {LORA_MODEL_PATH}")
        print("Please train the model first using train_sft.py")
        sys.exit(1)
    
    print(f"\n📖 Loading tokenizer from: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )
    print(f"  ✓ Tokenizer loaded")
    
    print(f"\n🤖 Loading base model from: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print(f"  ✓ Base model loaded")
    
    print(f"\n🔧 Loading LoRA weights from: {LORA_MODEL_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    print(f"  ✓ LoRA weights loaded")
    
    model.eval()
    print(f"  ✓ Model set to evaluation mode")
    
    return tokenizer, model


# ==================== Inference ====================

def format_test_input(profile: dict, initial_schedule: list) -> str:
    """Format test input in the same format as training data"""
    
    instruction = """You are a Critic & Editor Agent for daily schedule refinement.

Your task:
1. Check the INITIAL SCHEDULE against 5 constraint types (Physical, Logical, Common Sense, Temporal, Coherence)
2. Identify any violations
3. Apply edit operations (DELETE/ADD/SHIFT/REPLACE) to fix issues
4. Output the refined schedule

Output format:
[THOUGHT]
**Constraint Checking (5 types):**
1. Physical Constraints (Hard): overlaps, 24h coverage, ends at 24:00
2. Logical Constraints (Hard): starts/ends at home, starts at 00:00
3. Common Sense Constraints (Soft): age/employment appropriate activities
4. Temporal Constraints (Soft): realistic durations
5. Coherence Constraints (Soft): logical transitions, not over-fragmented

**Applying Edit Operations:**
- DELETE: activity 'X' at index N (reason)
- ADD: activity 'Y' at time (reason)
- SHIFT: activity time adjustment (reason)
- REPLACE: activity type change (reason)

**Final Result:** ✓ Yes / ✗ No
[/THOUGHT]

[JSON]
[refined schedule as JSON array]
[/JSON]
"""
    
    input_text = "**PERSON PROFILE:**\n"
    input_text += json.dumps(profile, indent=2, ensure_ascii=False)
    input_text += "\n\n**INITIAL SCHEDULE:**\n"
    input_text += json.dumps(initial_schedule, indent=2, ensure_ascii=False)
    
    # Build ChatML format
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text}
    ]
    
    return messages


def generate_output(tokenizer, model, messages, max_new_tokens=2048):
    """Generate model output"""
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    print("\n🚀 Generating output...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part (skip input)
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response


# ==================== Main ====================

def main():
    """Test the trained model"""
    
    # Load model
    tokenizer, model = load_model()
    
    # Prepare input
    print("\n" + "=" * 80)
    print(" TEST INPUT")
    print("=" * 80)
    print(f"\n📋 Person Profile:")
    print(json.dumps(TEST_PROFILE, indent=2, ensure_ascii=False))
    print(f"\n📅 Initial Schedule:")
    print(json.dumps(TEST_INITIAL_SCHEDULE, indent=2, ensure_ascii=False))
    
    messages = format_test_input(TEST_PROFILE, TEST_INITIAL_SCHEDULE)
    
    # Generate output
    response = generate_output(tokenizer, model, messages)
    
    # Display result
    print("\n" + "=" * 80)
    print(" MODEL OUTPUT")
    print("=" * 80)
    print(response)
    
    # Validate format
    print("\n" + "=" * 80)
    print(" VALIDATION")
    print("=" * 80)
    
    required_sections = ["[THOUGHT]", "[/THOUGHT]", "[JSON]", "[/JSON]"]
    validation_results = []
    
    for section in required_sections:
        present = section in response
        status = "✅" if present else "❌"
        validation_results.append(present)
        print(f"{status} {section}")
    
    if all(validation_results):
        print("\n✅ Output format is valid!")
        
        # Try to extract and parse JSON
        try:
            json_start = response.find("[JSON]") + 6
            json_end = response.find("[/JSON]")
            json_str = response[json_start:json_end].strip()
            
            # Try to parse
            schedule = json.loads(json_str)
            print(f"\n✅ JSON is valid! Contains {len(schedule)} activities")
            
        except Exception as e:
            print(f"\n⚠️  JSON parsing failed: {str(e)}")
    else:
        print("\n❌ Output format is incomplete!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
