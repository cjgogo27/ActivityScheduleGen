"""
Convert Stage 1 training data to SFT format.

Input: teacher_generated_training_data.jsonl
  - user_profile
  - initial_schedule (from Planner+Realizer)
  - ground_truth_schedule (not used in input, only for reference)
  - teacher_output (becomes output target)

Output: sft_training_data.jsonl
  - instruction: Fixed prompt describing Editor's task
  - input: User profile + initial schedule JSON
  - output: [THOUGHT]...[/THOUGHT][JSON]...[/JSON]
"""

import json
import os
from typing import Dict, List

# ==================== Configuration ====================
INPUT_FILE = "stage1_training_data/teacher_generated_training_data.jsonl"
OUTPUT_FILE = "stage1_training_data/sft_training_data.jsonl"

# ==================== Instruction Template ====================
INSTRUCTION = """You are a Critic & Editor Agent for daily schedule refinement.

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


def format_input(user_profile: Dict, initial_schedule: List[Dict]) -> str:
    """
    Format user profile and initial schedule as input text.
    
    Returns:
        Formatted string with profile and schedule
    """
    input_text = "**PERSON PROFILE:**\n"
    input_text += json.dumps(user_profile, indent=2, ensure_ascii=False)
    input_text += "\n\n**INITIAL SCHEDULE:**\n"
    input_text += json.dumps(initial_schedule, indent=2, ensure_ascii=False)
    
    return input_text


def convert_sample(raw_sample: Dict) -> Dict:
    """
    Convert one Stage 1 sample to SFT format.
    
    Args:
        raw_sample: {
            "user_profile": {...},
            "initial_schedule": [...],
            "ground_truth_schedule": [...],
            "teacher_output": "[THOUGHT]...[/THOUGHT][JSON]...[/JSON]",
            "success": True
        }
    
    Returns:
        {
            "instruction": "...",
            "input": "...",
            "output": "[THOUGHT]...[/THOUGHT][JSON]...[/JSON]"
        }
    """
    # Only process successful samples
    if not raw_sample.get('success', False):
        return None
    
    user_profile = raw_sample['user_profile']
    initial_schedule = raw_sample['initial_schedule']
    teacher_output = raw_sample['teacher_output']
    
    # Format input
    input_text = format_input(user_profile, initial_schedule)
    
    # Create SFT sample
    sft_sample = {
        "instruction": INSTRUCTION.strip(),
        "input": input_text,
        "output": teacher_output
    }
    
    return sft_sample


def main():
    print("=" * 80)
    print(" CONVERTING STAGE 1 DATA TO SFT FORMAT")
    print("=" * 80)
    
    # Check input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ Error: Input file not found: {INPUT_FILE}")
        return
    
    # Read and convert
    print(f"\n📂 Reading from: {INPUT_FILE}")
    
    total_samples = 0
    successful_samples = 0
    failed_samples = 0
    
    # Clear output file if exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
        for line_num, line in enumerate(f_in, 1):
            total_samples += 1
            
            try:
                raw_sample = json.loads(line)
                sft_sample = convert_sample(raw_sample)
                
                if sft_sample:
                    # Save to output file
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                        f_out.write(json.dumps(sft_sample, ensure_ascii=False) + '\n')
                    successful_samples += 1
                else:
                    failed_samples += 1
                    
            except Exception as e:
                print(f"  ⚠️ Error processing line {line_num}: {str(e)[:100]}")
                failed_samples += 1
    
    print(f"\n✅ Conversion complete!")
    print(f"  📊 Total samples: {total_samples}")
    print(f"  ✓ Converted: {successful_samples}")
    print(f"  ✗ Failed: {failed_samples}")
    print(f"\n  💾 Output saved to: {OUTPUT_FILE}")
    
    # Show example
    if successful_samples > 0:
        print(f"\n" + "=" * 80)
        print(" EXAMPLE SFT SAMPLE")
        print("=" * 80)
        
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            example = json.loads(f.readline())
            
            print(f"\n📝 INSTRUCTION ({len(example['instruction'])} chars):")
            print("-" * 80)
            print(example['instruction'][:300] + "..." if len(example['instruction']) > 300 else example['instruction'])
            
            print(f"\n📥 INPUT ({len(example['input'])} chars):")
            print("-" * 80)
            print(example['input'][:400] + "..." if len(example['input']) > 400 else example['input'])
            
            print(f"\n📤 OUTPUT ({len(example['output'])} chars):")
            print("-" * 80)
            print(example['output'][:500] + "..." if len(example['output']) > 500 else example['output'])
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
