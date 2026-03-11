

import json
import time
import os
import sys
from openai import OpenAI
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict


API_KEY = "sk-qyl51vYITpONuIlsU2p3iNQnawX9GMzOICym"
BASE_URL = "https://api.nuwaflux.com/v1"
TIMEOUT = 60
TEACHER_MODEL = "gpt-5.2"  

PERSON_FILE = r"/data/alice/cjtest/FinalTraj_KDD/California/processed_data/california_person_static.json"
GROUND_TRUTH_FILE = r"/data/alice/cjtest/FinalTraj_KDD/California/processed_data/all_user_schedules.json"

OUTPUT_DIR = r"/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage1_training_data copy 3"
OUTPUT_FILE = "teacher_generated_training_data.jsonl"

def create_openai_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)

def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_person_profile(person_data: Dict) -> Dict[str, Any]:
    return {
        'user_id': person_data.get('user_id', 'Unknown'), 
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


def generate_initial_schedule_unified(client: OpenAI, user_profile: Dict, max_retries: int = 3) -> Optional[List[Dict]]:
    profile_str = json.dumps(user_profile, indent=2, ensure_ascii=False)

    prompt = f"""You are a daily schedule planner. Generate a realistic 24-hour schedule for this person.

**PERSON PROFILE:**
{profile_str}

**YOUR TASK:** Generate schedule in TWO STAGES:

**STAGE 1: Activity Sequence**
Based on the person's profile (occupation, employment status, work schedule, etc.), determine:
- What activities should occur during the day
- Logical order of activities
- Activity types: work, home, shopping, dining, recreation, commute, personal, education, etc.

**STAGE 2: Time Assignment** 
For each activity, assign:
- start_time (HH:MM format, 24-hour)
- end_time (HH:MM format, 24-hour)
- duration (hours, decimal)

**CONSTRAINTS:**
- Must start at 00:00 (first activity)
- Must end at 24:00 (last activity) 
- No time overlaps
- Total duration = 24 hours exactly
- Activities must be realistic for the person's profile

**OUTPUT FORMAT:**
First show your reasoning:
[STAGE1_ACTIVITIES]
Activity 1: [activity type and brief reason]
Activity 2: [activity type and brief reason]
...
[/STAGE1_ACTIVITIES]

Then provide the complete schedule:
[SCHEDULE]
[
  {{
    "activity": "home/sleep",
    "start_time": "00:00",
    "end_time": "07:30",
    "duration": 7.5
  }},
  {{
    "activity": "commute to work",
    "start_time": "07:30",
    "end_time": "08:00",
    "duration": 0.5
  }},
  ...
]
[/SCHEDULE]

Generate the schedule now:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a schedule planning expert. You understand human behavior patterns and generate realistic daily schedules."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            output = None
            if isinstance(response, str):
                output = response
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                output = response.choices[0].message.content
            else:
                try:
                    output = response['choices'][0]['message']['content']
                except:
                    output = None
            
            if not output:
                continue
            
            import re
            schedule_match = re.search(r'\[SCHEDULE\](.*?)\[/SCHEDULE\]', output, re.DOTALL)
            if schedule_match:
                schedule_str = schedule_match.group(1).strip()
                schedule_str = re.sub(r'^```json\s*', '', schedule_str)
                schedule_str = re.sub(r'\s*```$', '', schedule_str)
                
                try:
                    schedule = json.loads(schedule_str)
                    if isinstance(schedule, list) and len(schedule) > 0:
                        return schedule
                except json.JSONDecodeError:
                    print(f"    ⚠️ Attempt {attempt+1}: Failed to parse JSON")
                    
        except Exception as e:
            print(f"    ⚠️ Attempt {attempt+1}: {str(e)[:100]}")
        
        if attempt < max_retries - 1:
            time.sleep(2)
    
    return None


def construct_teacher_prompt(user_profile: Dict, initial_schedule: List[Dict], ground_truth_schedule: List[Dict]) -> str:
    
    profile_str = json.dumps(user_profile, indent=2, ensure_ascii=False)
    
    initial_schedule_str = json.dumps(initial_schedule, indent=2, ensure_ascii=False)
    
    gt_schedule_str = json.dumps(ground_truth_schedule, indent=2, ensure_ascii=False)
    
    prompt = f"""You are a Critic & Editor Agent. Your job is to validate and refine a daily schedule.

**PERSON PROFILE:**
{profile_str}

**INITIAL SCHEDULE (from unified 2-stage generator):**
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
    initial_schedule = generate_initial_schedule_unified(client, user_profile)
    
    if not initial_schedule:
        return None
    
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
                max_tokens=3072 
            )

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
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def save_metadata_and_cot_separately(sample: Dict, metadata_file: str, cot_file: str):
    if not sample.get('success', False):
        return

    metadata = {
        'user_id': sample['user_profile']['user_id'],
        'user_profile': sample['user_profile'],
        'initial_schedule': sample['initial_schedule'],
        'ground_truth_schedule': sample['ground_truth_schedule'],
        'generation_time': sample['generation_time']
    }
    
    cot_sample = {
        'user_id': sample['user_profile']['user_id'],
        'teacher_output': sample['teacher_output']
    }
    
    with open(metadata_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
    
    with open(cot_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(cot_sample, ensure_ascii=False) + '\n')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filepath = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    metadata_filepath = os.path.join(OUTPUT_DIR, "metadata.jsonl")
    cot_filepath = os.path.join(OUTPUT_DIR, "cot_only.jsonl")
    client = create_openai_client()

    persons_list = load_json(PERSON_FILE)
    ground_truth_data = load_json(GROUND_TRUTH_FILE)
    
    gt_dict = {}
    for entry in ground_truth_data:
        user_id = entry.get('user_id')
        schedule = entry.get('schedule', [])
        if user_id and schedule:
            gt_dict[user_id] = schedule
    
    matched_samples = []
    for person in persons_list:
        user_id = person.get('user_id')
        if user_id and user_id in gt_dict:
            profile = extract_person_profile(person)
            schedule = gt_dict[user_id]
            matched_samples.append((profile, schedule))
    
    existing_user_ids = set()
    if os.path.exists(metadata_filepath):
        try:
            with open(metadata_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    user_id = data.get('user_id') or data.get('user_profile', {}).get('user_id')
                    if user_id:
                        existing_user_ids.add(user_id)
            print(f"  ℹ️  Found {len(existing_user_ids)} existing samples, will skip duplicates")
        except Exception as e:
            print(f"  ⚠️  Error reading existing samples: {str(e)[:100]}")
            pass

    if existing_user_ids:
        matched_samples = [(p, s) for p, s in matched_samples if p['user_id'] not in existing_user_ids]
        print(f"  → {len(matched_samples)} new samples available after filtering")

    max_samples = 5  
    if len(matched_samples) > max_samples:
        import random
        random.seed(42)
        matched_samples = random.sample(matched_samples, max_samples)
    
    successful = 0
    failed = 0

    
    for idx, (profile, schedule) in enumerate(matched_samples, 1):
        user_id = profile['user_id']
        sample = generate_training_sample(client, profile, schedule)
        
        if sample and sample['success']:
            save_training_sample(sample, output_filepath)
            save_metadata_and_cot_separately(sample, metadata_filepath, cot_filepath)
            successful += 1
            print(f"  ✓ Generated and saved")
            
            output = sample['teacher_output']
            
            thought_start = output.find('[THOUGHT]')
            thought_end = output.find('[/THOUGHT]')
            if thought_start >= 0 and thought_end > thought_start:
                thought_snippet = output[thought_start+9:thought_end].strip()[:100]
                print(f"  💭 Editor Check: {thought_snippet}...")
        else:
            failed += 1
            if sample:
                save_training_sample(sample, output_filepath)  
        print()
        
        if idx < len(matched_samples):
            time.sleep(1) 
    
    if successful > 0:
        with open(output_filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            example = json.loads(first_line)
            if example['success']:
                pass


if __name__ == "__main__":
    main()
