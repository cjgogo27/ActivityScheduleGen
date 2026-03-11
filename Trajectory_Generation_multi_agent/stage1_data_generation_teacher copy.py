"""
Stage 1: Chain-of-Thought Distillation for Critic & Editor Agent
==================================================================

Method: Distill Editor's constraint-checking and editing reasoning

Target Agent: Critic & Editor ONLY (NOT Planner or Realizer)

Approach:
- Teacher Model: GPT-5.1
- Input: Person Profile + Initial Schedule (Ground Truth as proxy)
- Task: Show Editor's CoT process
  1. Check 5 constraint types (Physical, Logical, Common Sense, Temporal, Coherence)
  2. Identify violations
  3. Apply edit operations: ADD/DELETE/SHIFT/REPLACE activities
  4. Output refined schedule
  
Output Format:
  [THOUGHT] 
    - Constraint checking (5 types)
    - Edit operations (if violations found)
    - Final verification
  [/THOUGHT]
  
  [JSON]
    - Final schedule
  [/JSON]

This creates training data for Editor agent's constraint-based refinement process.
"""


import json
import time
import os
import sys
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Import Planner and Realizer from three_agent_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_trajectories_three_agent_pipeline import PlannerAgent, TripRealizerAgent

# ==================== Configuration ====================
API_KEY = "sk-qyl51vYITpOoElayZ5gmNuIlsU2p3iNQnawX9G0RyMzOICym"
BASE_URL = "https://api.nuwaflux.com/v1"
TIMEOUT = 60
TEACHER_MODEL = "gpt-5.1"  # Must match the model used in pipeline

# Input files
# IMPORTANT: Person and Schedule files MUST be from the SAME state for user_id matching!
# Using California data (has both person profiles and schedules)
PERSON_FILE = r"/data/alice/cjtest/FinalTraj_KDD/California/processed_data/california_person_static.json"
GROUND_TRUTH_FILE = r"/data/alice/cjtest/FinalTraj_KDD/California/processed_data/all_user_schedules.json"

# Output directory
OUTPUT_DIR = r"/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage1_training_data copy 3"
OUTPUT_FILE = "teacher_generated_training_data.jsonl"

# ==================== Utility Functions ====================
def create_openai_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)

def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_person_profile(person_data: Dict) -> Dict[str, Any]:
    """Extract relevant user profile information"""
    return {
        'user_id': person_data.get('user_id', 'Unknown'),  # Key field for matching
        'age_range': person_data.get('age_range', 'Unknown'),
        'gender': person_data.get('gender', 'Unknown'),
        'race': person_data.get('race', 'Unknown'),
        'education': person_data.get('education', 'Unknown'),
        'employment_status': person_data.get('employment_status', 'Unknown'),
        'work_schedule': person_data.get('work_schedule', 'Unknown'),
        'occupation': person_data.get('occupation', 'Unknown'),
        'primary_activity': person_data.get('primary_activity', 'Unknown'),
        'work_from_home': person_data.get('work_from_home', 'Unknown'),
        'driver_on_travel_day': person_data.get('driver_on_travel_day', 'Unknown'),
        'distance_to_work_miles': person_data.get('distance_to_work_miles', 0),
        'work_state': person_data.get('work_state', 'Unknown')
    }


def construct_teacher_prompt(user_profile: Dict, initial_schedule: List[Dict], ground_truth_schedule: List[Dict]) -> str:
    """
    Chain-of-Thought Distillation for Critic & Editor Agent ONLY.
    
    Task: Given an initial schedule (from Planner+Realizer), show the Editor's reasoning:
      1. Check 5 constraint types
      2. Identify violations
      3. Apply edit operations (ADD/DELETE/SHIFT/REPLACE)
      4. Output final schedule (matching ground truth)
    
    This trains the Editor to perform constraint-based iterative refinement.
    
    NOTE: We DO run Planner+Realizer to generate initial schedule,
          but we DON'T save their CoT - only Editor's CoT is saved.
    """
    
    # Format user profile
    profile_str = json.dumps(user_profile, indent=2, ensure_ascii=False)
    
    # Format initial schedule (from Planner+Realizer)
    initial_schedule_str = json.dumps(initial_schedule, indent=2, ensure_ascii=False)
    
    # Format ground truth (target for Editor to match)
    gt_schedule_str = json.dumps(ground_truth_schedule, indent=2, ensure_ascii=False)
    
    prompt = f"""You are a Critic & Editor Agent. Your job is to validate and refine a daily schedule.

**PERSON PROFILE:**
{profile_str}

**INITIAL SCHEDULE (from Planner + Realizer):**
{initial_schedule_str}

**GROUND TRUTH REFERENCE (for comparison):**
{gt_schedule_str}

**YOUR TASK:** 
Act as the Editor. Check constraints on the INITIAL SCHEDULE, identify violations, and apply edits to match the GROUND TRUTH.

Output in 2 sections:

[THOUGHT]
**Constraint Checking:**

1. Physical (Hard): overlaps? 24h coverage? ends 24:00? → ✓/✗
2. Logical (Hard): starts/ends home? starts 00:00? → ✓/✗
3. Common Sense (Soft): activities match profile? → ✓/✗
4. Temporal (Soft): realistic durations? → ✓/✗
5. Coherence (Soft): logical flow? not fragmented? → ✓/✗

**Edits to Match GT:**
[If already matches: "No edits needed"]
[Otherwise list:]
- DELETE: 'X' at idx N (reason)
- ADD: 'Y' at time (reason)
- SHIFT: 'Z' time adjustment (reason)
- REPLACE: 'A'→'B' (reason)

**Final Result:**
All constraints satisfied after edits? ✓ Yes / ✗ No
[/THOUGHT]

[JSON]
{gt_schedule_str}
[/JSON]

Be thorough in checking each constraint on the INITIAL SCHEDULE. 
Show edit operations explicitly if violations are found.
The final JSON should match the GROUND TRUTH.

Begin:"""

    return prompt


def generate_training_sample(
    client: OpenAI,
    user_profile: Dict,
    ground_truth_schedule: List[Dict],
    max_retries: int = 3
) -> Optional[Dict]:
    """
    Generate one training sample with Editor's CoT reasoning.
    
    Steps:
      1. Use Planner to generate activity skeleton (don't save CoT)
      2. Use Realizer to assign times (don't save CoT)
      3. Use Teacher model to generate Editor's CoT (CHECK + EDIT to reach GT)
    
    Returns:
        {
            "user_profile": {...},
            "initial_schedule": [...],  # From Planner+Realizer
            "ground_truth_schedule": [...],
            "teacher_output": "[THOUGHT]...[/THOUGHT][JSON]...[/JSON]",  # Editor's CoT only
            "generation_time": "...",
            "success": True/False
        }
    """
    
    # Step 1: Generate initial schedule using Planner + Realizer
    try:
        # Create agents for this specific person
        planner_agent = PlannerAgent(client, user_profile)
        realizer_agent = TripRealizerAgent(client, user_profile)
        
        # Planner generates activity skeleton (returns True/False)
        planner_success = planner_agent.plan_activities()
        if not planner_success:
            print(f"    ❌ Planner failed to generate activities")
            return None
        
        activities = planner_agent.activity_skeleton
        if not activities:
            print(f"    ❌ Planner returned empty activity list")
            return None
        
        # Realizer assigns times (returns True/False)
        realizer_success = realizer_agent.realize_schedule(activities)
        if not realizer_success:
            print(f"    ❌ Realizer failed to assign times")
            return None
        
        initial_schedule = realizer_agent.schedule
        if not initial_schedule:
            print(f"    ❌ Realizer returned empty schedule")
            return None
            
        print(f"    ✓ Generated initial schedule: {len(initial_schedule)} activities")
        
    except Exception as e:
        print(f"    ❌ Planner/Realizer error: {str(e)[:100]}")
        return None
    
    # Step 2: Generate Editor's CoT using teacher model
    prompt = construct_teacher_prompt(user_profile, initial_schedule, ground_truth_schedule)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in human behavior analysis. Your task is to REVERSE-ENGINEER decision-making processes from observed real-world schedules. You analyze why people made specific activity and time choices based on their demographics and circumstances."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=3072  # Increased: reasoning (640) + detailed output (2400+)
            )
            
            # Handle different response types
            teacher_output = None
            if isinstance(response, str):
                teacher_output = response
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    teacher_output = content.strip()
                else:
                    teacher_output = ""
            else:
                try:
                    teacher_output = response['choices'][0]['message']['content'].strip()
                except (KeyError, IndexError, TypeError) as e:
                    teacher_output = ""
            
            if teacher_output is None:
                teacher_output = ""
            
            # Validate output format - check for Editor sections only
            required_sections = ["[THOUGHT]", "[/THOUGHT]", "[JSON]", "[/JSON]"]
            all_present = all(section in teacher_output for section in required_sections)
            
            if all_present:
                return {
                    "user_profile": user_profile,
                    "initial_schedule": initial_schedule,  # From Planner+Realizer
                    "ground_truth_schedule": ground_truth_schedule,
                    "teacher_output": teacher_output,  # Editor's CoT only
                    "generation_time": datetime.now().isoformat(),
                    "success": True
                }
            else:
                missing = [s for s in required_sections if s not in teacher_output]
                print(f"    ⚠️ Attempt {attempt+1}: Missing sections: {missing}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"    ❌ Failed after {max_retries} attempts")
                    return {
                        "user_profile": user_profile,
                        "initial_schedule": initial_schedule,
                        "ground_truth_schedule": ground_truth_schedule,
                        "teacher_output": teacher_output,
                        "generation_time": datetime.now().isoformat(),
                        "success": False,
                        "error": f"Missing sections: {missing}"
                    }
                    
        except Exception as e:
            print(f"    ⚠️ Attempt {attempt+1} error: {str(e)[:150]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return {
                    "user_profile": user_profile,
                    "initial_schedule": initial_schedule,
                    "ground_truth_schedule": ground_truth_schedule,
                    "teacher_output": "",
                    "generation_time": datetime.now().isoformat(),
                    "success": False,
                    "error": str(e)[:200]
                }
    
    return None


def save_training_sample(sample: Dict, output_file: str):
    """Append one training sample to JSONL file"""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def save_metadata_and_cot_separately(sample: Dict, metadata_file: str, cot_file: str):
    """Save metadata and CoT separately for efficient storage"""
    if not sample.get('success', False):
        return
    
    # Extract metadata (static information)
    metadata = {
        'user_id': sample['user_profile']['user_id'],
        'user_profile': sample['user_profile'],
        'initial_schedule': sample['initial_schedule'],
        'ground_truth_schedule': sample['ground_truth_schedule'],
        'generation_time': sample['generation_time']
    }
    
    # Extract CoT only
    cot_sample = {
        'user_id': sample['user_profile']['user_id'],
        'teacher_output': sample['teacher_output']
    }
    
    # Save to separate files
    with open(metadata_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
    
    with open(cot_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(cot_sample, ensure_ascii=False) + '\n')


def main():
    print("=" * 80)
    print(" STAGE 1: CRITIC & EDITOR AGENT - CoT DISTILLATION")
    print(" Teacher Model: GPT-5.1")
    print(" Target: Editor's constraint-checking + editing operations")
    print(" Pipeline: Planner → Realizer → Editor (only save Editor's CoT)")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    metadata_filepath = os.path.join(OUTPUT_DIR, "metadata.jsonl")
    cot_filepath = os.path.join(OUTPUT_DIR, "cot_only.jsonl")
    
    # Note: Do NOT clear existing files - we're appending new samples
    if os.path.exists(output_filepath):
        print(f"\n✓ Will append to existing output files (no duplication)")
    else:
        print(f"\n✓ Creating new output files")
    
    # Initialize OpenAI client for all agents
    print(f"\n🤖 Initializing OpenAI client...")
    client = create_openai_client()
    print(f"  ✓ OpenAI client ready")
    print(f"  ✓ Will create Planner & Realizer for each person")
    print(f"  ✓ Teacher Model ({TEACHER_MODEL}) ready for Editor CoT")
    
    # Load data
    print(f"\n📂 Loading data...")
    persons_list = load_json(PERSON_FILE)
    ground_truth_data = load_json(GROUND_TRUTH_FILE)
    
    print(f"  ✓ Loaded {len(persons_list)} person records")
    print(f"  ✓ Loaded {len(ground_truth_data)} ground truth schedules")
    
    # Create mapping: user_id -> ground_truth_schedule
    gt_dict = {}
    for entry in ground_truth_data:
        user_id = entry.get('user_id')
        schedule = entry.get('schedule', [])
        if user_id and schedule:
            gt_dict[user_id] = schedule
    
    print(f"  ✓ Mapped {len(gt_dict)} user schedules")
    
    # Match persons with ground truth schedules
    matched_samples = []
    for person in persons_list:
        user_id = person.get('user_id')
        if user_id and user_id in gt_dict:
            profile = extract_person_profile(person)
            schedule = gt_dict[user_id]
            matched_samples.append((profile, schedule))
    
    print(f"\n🎯 Found {len(matched_samples)} matched samples (person + ground truth)")
    
    # Limit to reasonable number
    # Target: 200 samples for high-quality SFT training
    max_samples = 200  # Optimized: simplified prompt + separated storage
    if len(matched_samples) > max_samples:
        import random
        random.seed(42)
        matched_samples = random.sample(matched_samples, max_samples)
        print(f"  → Sampling {max_samples} for generation")
        print(f"  💰 Estimated cost: ~${max_samples * 0.026:.2f} (¥{max_samples * 0.026 * 7.2:.2f})")
        print(f"  ⏱️  Estimated time: ~{int(max_samples * 4 / 60)} minutes")
    else:
        print(f"  💰 Estimated cost: ~${len(matched_samples) * 0.026:.2f}")
    
    # Generate training data
    successful = 0
    failed = 0
    
    print(f"\n🚀 Starting generation with 3-Agent Pipeline...")
    print(f"   Step 1: Planner generates activities (CoT not saved)")
    print(f"   Step 2: Realizer assigns times (CoT not saved)")
    print(f"   Step 3: Teacher Model generates Editor's CoT ✅ SAVED")
    print(f"{'─' * 80}\n")
    
    for idx, (profile, schedule) in enumerate(matched_samples, 1):
        user_id = profile['user_id']
        print(f"[{idx}/{len(matched_samples)}] Processing {user_id}")
        print(f"  Profile: {profile['age_range']}, {profile['employment_status']}, {profile['primary_activity']}")
        print(f"  Ground Truth: {len(schedule)} activities")
        
        # Generate training sample (runs Planner→Realizer→Editor)
        sample = generate_training_sample(client, profile, schedule)
        
        if sample and sample['success']:
            save_training_sample(sample, output_filepath)
            save_metadata_and_cot_separately(sample, metadata_filepath, cot_filepath)
            successful += 1
            print(f"  ✓ Generated and saved")
            
            # Show snippet of output
            output = sample['teacher_output']
            
            # Extract [THOUGHT] snippet for display
            thought_start = output.find('[THOUGHT]')
            thought_end = output.find('[/THOUGHT]')
            if thought_start >= 0 and thought_end > thought_start:
                thought_snippet = output[thought_start+9:thought_end].strip()[:100]
                print(f"  💭 Editor Check: {thought_snippet}...")
        else:
            failed += 1
            if sample:
                save_training_sample(sample, output_filepath)  # Save even failed attempts for analysis
            print(f"  ❌ Failed")
        
        print()
        
        # Rate limiting
        if idx < len(matched_samples):
            time.sleep(1)  # Adjust based on API rate limits
    
    # Summary
    print(f"\n{'=' * 80}")
    print(f" GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"  Total samples: {len(matched_samples)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    if len(matched_samples) > 0:
        print(f"  Success rate: {successful/len(matched_samples)*100:.1f}%")
    else:
        print(f"  Success rate: N/A (no matched samples found)")
    print(f"\n  📁 Output saved to: {output_filepath}")
    print(f"{'=' * 80}\n")
    
    # Show example
    if successful > 0:
        print("📝 Example training sample:")
        print("─" * 80)
        with open(output_filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            example = json.loads(first_line)
            if example['success']:
                print(f"User ID: {example['user_profile']['user_id']}")
                print(f"\nTeacher Output:\n{example['teacher_output'][:800]}...")
                print("─" * 80)


if __name__ == "__main__":
    main()
