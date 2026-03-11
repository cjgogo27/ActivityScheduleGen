"""
Convert separated metadata and CoT files to SFT format.

Input sources:
  - metadata.jsonl: Contains user_id, user_profile, initial_schedule, ground_truth_schedule
  - cot_only.jsonl: Contains user_id, teacher_output

Output: sft_training_data.jsonl
  - instruction: Fixed prompt describing Editor's task
  - input: User profile ONLY (no initial_schedule)
  - output: teacher_output from cot_only.jsonl
"""

import json
import os
from typing import Dict

# ==================== Configuration ====================
METADATA_FILE = "stage1_training_data_3/metadata.jsonl"
COT_FILE = "stage1_training_data_3/cot_only.jsonl"
OUTPUT_FILE = "stage1_training_data_3/sft_training_data.jsonl"

# ==================== Instruction Template ====================
INSTRUCTION = """You are a Critic & Editor Agent for daily schedule generation and refinement.

Your task:
1. Based on the person's profile, generate a reasonable daily schedule
2. Check against 5 constraint types (Physical, Logical, Common Sense, Temporal, Coherence)
3. Ensure all constraints are satisfied
4. Output the final schedule

Output format:
[THOUGHT]
**Constraint Checking (5 types):**
1. Physical Constraints (Hard): overlaps, 24h coverage, ends at 24:00
2. Logical Constraints (Hard): starts/ends at home, starts at 00:00
3. Common Sense Constraints (Soft): age/employment appropriate activities
4. Temporal Constraints (Soft): realistic durations
5. Coherence Constraints (Soft): logical transitions, not over-fragmented

**Schedule Planning:**
[Your reasoning about what activities this person should do based on their profile]

**Final Result:** ✓ Yes / ✗ No
[/THOUGHT]

[JSON]
[final schedule as JSON array]
[/JSON]
"""


def load_metadata(filepath: str) -> Dict[str, Dict]:
    """
    Load metadata file and create user_id -> metadata mapping.
    
    Returns:
        Dict mapping user_id to metadata dict
    """
    metadata_dict = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_id = data.get('user_id')
            if user_id:
                metadata_dict[user_id] = data
    
    return metadata_dict


def load_cot(filepath: str) -> Dict[str, str]:
    """
    Load CoT file and create user_id -> teacher_output mapping.
    
    Returns:
        Dict mapping user_id to teacher_output string
    """
    cot_dict = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            user_id = data.get('user_id')
            teacher_output = data.get('teacher_output')
            if user_id and teacher_output:
                cot_dict[user_id] = teacher_output
    
    return cot_dict


def format_input(user_profile: Dict) -> str:
    """
    Format user profile as input text (without initial_schedule).
    
    Returns:
        Formatted string with profile only
    """
    input_text = "**PERSON PROFILE:**\n"
    input_text += json.dumps(user_profile, indent=2, ensure_ascii=False)
    
    return input_text


def create_sft_sample(user_id: str, metadata: Dict, teacher_output: str) -> Dict:
    """
    Create one SFT training sample.
    
    Args:
        user_id: User identifier
        metadata: Metadata dict containing user_profile
        teacher_output: Teacher model's CoT output
    
    Returns:
        {
            "instruction": "...",
            "input": "**PERSON PROFILE:**\n{...}",
            "output": "[THOUGHT]...[/THOUGHT][JSON]...[/JSON]"
        }
    """
    user_profile = metadata['user_profile']
    
    # Format input (profile only, no initial_schedule)
    input_text = format_input(user_profile)
    
    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": teacher_output
    }


def main():
    print("=" * 80)
    print(" CONVERTING SEPARATED FILES TO SFT FORMAT")
    print("=" * 80)
    
    # Check input files
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Metadata file not found: {METADATA_FILE}")
        return
    
    if not os.path.exists(COT_FILE):
        print(f"❌ CoT file not found: {COT_FILE}")
        return
    
    # Load data
    print(f"\n📂 Loading metadata from: {METADATA_FILE}")
    metadata_dict = load_metadata(METADATA_FILE)
    print(f"  ✓ Loaded {len(metadata_dict)} metadata records")
    
    print(f"\n📂 Loading CoT data from: {COT_FILE}")
    cot_dict = load_cot(COT_FILE)
    print(f"  ✓ Loaded {len(cot_dict)} CoT records")
    
    # Match and convert
    print(f"\n🔄 Converting to SFT format...")
    matched = 0
    converted = 0
    
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for user_id in metadata_dict:
            matched += 1
            
            # Check if we have CoT for this user
            if user_id not in cot_dict:
                print(f"  ⚠️  No CoT found for user_id: {user_id}")
                continue
            
            # Create SFT sample
            sft_sample = create_sft_sample(
                user_id,
                metadata_dict[user_id],
                cot_dict[user_id]
            )
            
            # Write to output
            f_out.write(json.dumps(sft_sample, ensure_ascii=False) + '\n')
            converted += 1
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f" CONVERSION COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Total metadata records: {len(metadata_dict)}")
    print(f"  Total CoT records: {len(cot_dict)}")
    print(f"  Matched samples: {matched}")
    print(f"  Converted samples: {converted}")
    print(f"\n  📁 Output saved to: {OUTPUT_FILE}")
    print(f"{'=' * 80}\n")
    
    # Show example
    if converted > 0:
        print("📝 Example SFT sample:")
        print("─" * 80)
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            example = json.loads(first_line)
            print(f"Instruction: {example['instruction'][:150]}...")
            print(f"\nInput (length): {len(example['input'])} chars")
            print(f"Input preview:\n{example['input'][:300]}...")
            print(f"\nOutput (length): {len(example['output'])} chars")
            print(f"Output preview:\n{example['output'][:300]}...")
            print("─" * 80)


if __name__ == "__main__":
    main()
