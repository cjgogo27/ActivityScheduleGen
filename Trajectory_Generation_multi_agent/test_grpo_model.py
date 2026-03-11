"""
Test the GRPO-trained model.

This script loads the GRPO model and tests it on sample user profiles,
evaluating both generation quality and reward scores.
"""

import json
import re
import torch
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import reward model from training script
import sys
sys.path.append('/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent')
from train_grpo import ScheduleRewardModel, extract_schedule_from_output

# ==================== Configuration ====================

BASE_MODEL_PATH = "/data/alice/cjtest/FinalTraj_KDD/finetune/models/Qwen3-8B/Qwen/Qwen3-8B"
GRPO_MODEL_PATH = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage3_grpo_output/final_model"

# For comparison, can also test SFT model
SFT_MODEL_PATH = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage2_sft_output_epoch10/final_model"

TEST_DATA_FILE = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage1_training_data_3/metadata.jsonl"

# Generation parameters
MAX_LENGTH = 4096
TEMPERATURE = 0.7
TOP_P = 0.9
NUM_TEST_SAMPLES = 5


# ==================== Model Loading ====================

def load_model(model_path: str, model_name: str = "Model"):
    """Load model and tokenizer."""
    print(f"\n{'='*80}")
    print(f" Loading {model_name}")
    print(f"{'='*80}")
    print(f"📍 Model path: {model_path}")
    
    # Load tokenizer
    print("\n📖 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("  ✓ Tokenizer loaded")
    
    # Load base model
    print("\n🤖 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("  ✓ Base model loaded")
    
    # Load LoRA weights
    print(f"\n🔧 Loading LoRA weights from {model_name}...")
    model = PeftModel.from_pretrained(base_model, model_path)
    print(f"  ✓ {model_name} LoRA weights loaded")
    
    return tokenizer, model


# ==================== Testing ====================

def create_prompt(user_profile: Dict) -> str:
    """Create prompt from user profile."""
    instruction = """You are a Critic & Editor Agent for daily schedule generation and refinement.

Your task:
1. Based on the user profile below, generate a realistic and constraint-compliant daily schedule.
2. Output format: [THOUGHT]reasoning here[/THOUGHT][JSON]schedule array[/JSON]

Constraints to satisfy:
- Physical: No time overlaps, total 24 hours
- Logical: Start/end appropriately
- Commonsense: Activities match user attributes
- Socioeconomic: Reasonable for user's occupation/status
- Temporal: Activities at appropriate times
- Internal: Consistent with user profile details"""
    
    profile_str = json.dumps(user_profile, indent=2, ensure_ascii=False)
    prompt_text = f"{instruction}\n\nUser Profile:\n{profile_str}\n\nGenerate schedule:"
    
    return prompt_text


def generate_schedule(model, tokenizer, user_profile: Dict) -> str:
    """Generate schedule for a user profile."""
    prompt = create_prompt(user_profile)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response (remove prompt)
    response = generated_text[len(prompt):].strip()
    
    return response


def evaluate_output(user_profile: Dict, output: str, reward_model: ScheduleRewardModel) -> Dict:
    """Evaluate generated output."""
    # Extract schedule
    schedule = extract_schedule_from_output(output)
    
    if not schedule:
        return {
            'valid_format': False,
            'reward': -100,
            'breakdown': {'error': 'Failed to extract schedule'},
            'schedule': None
        }
    
    # Compute reward
    reward, breakdown = reward_model.compute_reward(user_profile, schedule)
    
    return {
        'valid_format': True,
        'reward': reward,
        'breakdown': breakdown,
        'schedule': schedule
    }


def test_model(model_path: str, model_name: str, test_samples: List[Dict], num_samples: int = 5):
    """Test a model on sample data."""
    print(f"\n{'='*80}")
    print(f" Testing {model_name}")
    print(f"{'='*80}\n")
    
    # Load model
    tokenizer, model = load_model(model_path, model_name)
    
    # Initialize reward model
    reward_model = ScheduleRewardModel()
    
    # Test samples
    results = []
    total_reward = 0
    
    for i, sample in enumerate(test_samples[:num_samples]):
        print(f"\n{'─'*80}")
        print(f" Test Sample {i+1}/{num_samples}")
        print(f"{'─'*80}")
        
        user_profile = sample['user_profile']
        user_id = user_profile.get('user_id', f'test_{i}')
        
        print(f"\n👤 User ID: {user_id}")
        print(f"   Occupation: {user_profile.get('occupation', 'N/A')}")
        print(f"   Age: {user_profile.get('age_range', 'N/A')}")
        print(f"   Employment: {user_profile.get('employment_status', 'N/A')}")
        
        # Generate
        print(f"\n🤖 Generating schedule...")
        output = generate_schedule(model, tokenizer, user_profile)
        
        # Evaluate
        print(f"📊 Evaluating...")
        evaluation = evaluate_output(user_profile, output, reward_model)
        
        # Print results
        print(f"\n{'─'*40}")
        print(f" Results")
        print(f"{'─'*40}")
        print(f"✓ Valid format: {evaluation['valid_format']}")
        print(f"🎯 Reward: {evaluation['reward']:.2f}")
        
        if evaluation['valid_format']:
            breakdown = evaluation['breakdown']
            print(f"\n📋 Reward Breakdown:")
            print(f"  Physical: {breakdown.get('physical', {}).get('score', 'N/A')}")
            print(f"  Logical: {breakdown.get('logical', {}).get('score', 'N/A')}")
            print(f"  Commonsense: {breakdown.get('commonsense', 'N/A')}")
            print(f"  Socioeconomic: {breakdown.get('socioeconomic', 'N/A')}")
            print(f"  Temporal: {breakdown.get('temporal', 'N/A')}")
            print(f"  Internal: {breakdown.get('internal_consistency', 'N/A')}")
            print(f"  Hard constraint violated: {breakdown.get('violated_hard_constraint', 'N/A')}")
            
            print(f"\n📅 Generated Schedule ({len(evaluation['schedule'])} activities):")
            for j, act in enumerate(evaluation['schedule'][:5]):  # Show first 5
                print(f"  {j+1}. {act.get('start_time', '?')} - {act.get('end_time', '?')}: "
                      f"{act.get('activity', '?')}")
            if len(evaluation['schedule']) > 5:
                print(f"  ... and {len(evaluation['schedule']) - 5} more activities")
        
        results.append({
            'user_id': user_id,
            'output': output,
            'evaluation': evaluation
        })
        
        total_reward += evaluation['reward']
    
    # Summary
    avg_reward = total_reward / num_samples
    valid_count = sum(1 for r in results if r['evaluation']['valid_format'])
    
    print(f"\n{'='*80}")
    print(f" {model_name} Summary")
    print(f"{'='*80}")
    print(f"  Samples tested: {num_samples}")
    print(f"  Valid outputs: {valid_count}/{num_samples} ({100*valid_count/num_samples:.1f}%)")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"{'='*80}\n")
    
    return results, avg_reward


def compare_models():
    """Compare SFT and GRPO models."""
    print(f"\n{'#'*80}")
    print(f"# MODEL COMPARISON: SFT vs GRPO")
    print(f"{'#'*80}\n")
    
    # Load test data
    print("📂 Loading test data...")
    test_samples = []
    with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            test_samples.append(json.loads(line.strip()))
    print(f"  ✓ Loaded {len(test_samples)} samples\n")
    
    # Test SFT model
    print("\n" + "🔵" * 40)
    print("Testing SFT Model (Baseline)")
    print("🔵" * 40)
    sft_results, sft_avg_reward = test_model(
        SFT_MODEL_PATH, 
        "SFT Model", 
        test_samples, 
        NUM_TEST_SAMPLES
    )
    
    # Test GRPO model
    print("\n" + "🟢" * 40)
    print("Testing GRPO Model (Rule-Guided)")
    print("🟢" * 40)
    grpo_results, grpo_avg_reward = test_model(
        GRPO_MODEL_PATH, 
        "GRPO Model", 
        test_samples, 
        NUM_TEST_SAMPLES
    )
    
    # Final comparison
    print(f"\n{'='*80}")
    print(f" FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"  SFT Average Reward:  {sft_avg_reward:+.2f}")
    print(f"  GRPO Average Reward: {grpo_avg_reward:+.2f}")
    improvement = grpo_avg_reward - sft_avg_reward
    print(f"  Improvement:         {improvement:+.2f} ({100*improvement/abs(sft_avg_reward):.1f}%)")
    
    if grpo_avg_reward > sft_avg_reward:
        print(f"\n  ✅ GRPO model shows improvement!")
    else:
        print(f"\n  ⚠️ GRPO model needs more training or tuning")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GRPO-trained model')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare SFT and GRPO models')
    parser.add_argument('--model', type=str, default='grpo',
                       choices=['sft', 'grpo'],
                       help='Which model to test (if not comparing)')
    parser.add_argument('--num_samples', type=int, default=NUM_TEST_SAMPLES,
                       help='Number of test samples')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        # Load test data
        print("📂 Loading test data...")
        test_samples = []
        with open(TEST_DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                test_samples.append(json.loads(line.strip()))
        print(f"  ✓ Loaded {len(test_samples)} samples\n")
        
        # Test selected model
        model_path = GRPO_MODEL_PATH if args.model == 'grpo' else SFT_MODEL_PATH
        model_name = "GRPO Model" if args.model == 'grpo' else "SFT Model"
        
        test_model(model_path, model_name, test_samples, args.num_samples)
