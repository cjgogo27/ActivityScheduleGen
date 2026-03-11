
import json
import time
import os
import random
import sys
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Any
import re

# 添加finetune目录到路径
sys.path.insert(0, '../finetune')
# from local_llama_client import create_local_llama_client # 暂时注释，假设使用OpenAI或已配置好的Client

# 模型配置
USE_LOCAL_MODEL = False  # 设置为True使用本地Llama,False使用OpenAI
BASE_MODEL_PATH = "../finetune/Llama/LLM-Research/Meta-Llama-3___1-8B-Instruct"
LORA_PATH = "../finetune/output/llama3_1_trajectory_lora_v2_20251210_225420/final"
MODEL = "gpt-4o" # Groq API 模型名称

# API配置
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1"
TIMEOUT = 60

# 数据路径 - California州
PERSON_FILE = r"E:\mayue\FinalTraj\California\processed_data\california_person_static.json"
HOUSEHOLD_FILE = r"E:\mayue\FinalTraj\California\processed_data\california_household_static.json"
OUTPUT_DIR = "./output"
OUTPUT_TRAJECTORIES_DIR = "./output_trajectories"
HOUSEHOLD_ID_DIR = "./household_id"
USER_ID_DIR = "./user_id"

# 生成模式配置
GENERATION_MODE = "household_file" # Options: "household_file", "random"
HOUSEHOLD_ID_FILE = r"E:\mayue\FinalTraj\Trajectory_Generation_multi_agent\output_trajectories\all_trajectories_20251214_180155_California.json"

ALLOWED_ACTIVITIES = {
    "home", "work", "education", "shopping", "service", 
    "medical", "dine_out", "socialize", "exercise", "dropoff_pickup"
}

TASK_TO_ACTIVITY_MAPPING = {
    "shopping": "shopping",
    "grocery_shopping": "shopping",
    "vehicle_maintenance": "service",
    "car_maintenance": "service",
    "house_cleaning": "home",
    "cleaning": "home",
    "laundry": "home",
    "cooking": "home",
    "medical_appointment": "medical",
    "doctor_visit": "medical",
    "dropoff_pickup": "dropoff_pickup",
    "school_pickup": "dropoff_pickup",
    "childcare": "dropoff_pickup",
}

def map_task_to_activity(task_id: str) -> str:
    task_id_lower = task_id.lower().strip()
    if task_id_lower in TASK_TO_ACTIVITY_MAPPING:
        return TASK_TO_ACTIVITY_MAPPING[task_id_lower]
    for key, value in TASK_TO_ACTIVITY_MAPPING.items():
        if key in task_id_lower or task_id_lower in key:
            return value
    return "service"

# ==================== RAG Module ====================
class RetrievalAugmentedLLM:
    """RAG module for household coordination - stores and retrieves household activities"""
    
    def __init__(self):
        self.household_activities = {}  # {household_id: {user_id: trajectory_text}}
        self.household_parsed_schedules = {} # {household_id: {user_id: parsed_schedule_list}}
    
    def store_generated_activity(self, household_id, user_id, trajectory_text, parsed_schedule):
        """Store generated activity in database"""
        if household_id not in self.household_activities:
            self.household_activities[household_id] = {}
            self.household_parsed_schedules[household_id] = {}
            
        self.household_activities[household_id][user_id] = trajectory_text
        self.household_parsed_schedules[household_id][user_id] = parsed_schedule
    
    def retrieve_household_activities(self, household_id, exclude_user_id=None):
        """Retrieve other household members' activities (Text format for Prompt)"""
        if household_id not in self.household_activities:
            return {}
        
        activities = {}
        for user_id, trajectory in self.household_activities[household_id].items():
            if exclude_user_id and user_id == exclude_user_id:
                continue
            activities[user_id] = trajectory
        
        return activities

    def retrieve_parsed_schedules(self, household_id, exclude_user_id=None):
        """Retrieve other household members' parsed schedules (List format for Logic Check)"""
        if household_id not in self.household_parsed_schedules:
            return {}
        
        schedules = {}
        for user_id, schedule in self.household_parsed_schedules[household_id].items():
            if exclude_user_id and user_id == exclude_user_id:
                continue
            schedules[user_id] = schedule
        return schedules

def clean_json_response(result_text: str) -> str:
    """清理模型输出的JSON文本"""
    result_text = result_text.strip()
    if "```json" in result_text:
        json_start = result_text.find("```json") + 7
        json_end = result_text.find("```", json_start)
        if json_end == -1: json_end = len(result_text)
        result_text = result_text[json_start:json_end].strip()
    elif "```" in result_text:
        json_start = result_text.find("```") + 3
        json_end = result_text.find("```", json_start)
        if json_end == -1: json_end = len(result_text)
        result_text = result_text[json_start:json_end].strip()
    return result_text

class Agent:
    def __init__(
        self, 
        agent_id: str, 
        person_info: Dict[str, Any],
        household_info: Dict[str, Any],
        rag_module: RetrievalAugmentedLLM,
        openai_client
    ):
        self.agent_id = agent_id
        self.person_info = person_info
        self.household_info = household_info
        self.rag_module = rag_module
        self.client = openai_client
        
        # State variables
        self.mandatory_activities = []
        self.allocated_tasks = []
        self.conversation_history = []
        self.final_schedule = []
        self.raw_trajectory_text = ""

    def get_profile_summary(self) -> str:
        p = self.person_info
        return f"""- User ID: {p['user_id']}
- Relationship: {p['relationship']}
- Age: {p['age_range']}, Gender: {p['gender']}
- Education: {p['education']}
- Employment: {p['employment_status']}, Schedule: {p['work_schedule']}
- Occupation: {p['occupation']}
- Driver License: {p['driver_on_travel_day']}
- Work from home: {p['work_from_home']}"""

    def get_mandatory_summary(self) -> str:
        if not self.mandatory_activities:
            return "Free all day"
        return ", ".join([
            a['activity'] if isinstance(a, dict) else str(a)
            for a in self.mandatory_activities
        ])

    def update_conversation_history(self, conversation: List[Dict[str, str]]):
        self.conversation_history = conversation
    
    def set_allocated_tasks(self, tasks: List[Dict[str, Any]]):
        self.allocated_tasks = tasks

    # --- Phase 1: Mandatory Activities ---
    def propose_mandatory_activities(self, max_retries: int = 3) -> bool:
        has_young_children = self.household_info.get('young_children_count', 0) > 0
        work_schedule = self.person_info.get('work_schedule', 'Unknown')
        employment = self.person_info.get('employment_status', 'Unknown')
        education = self.person_info.get('education', 'Unknown')
        relationship = str(self.person_info.get('relationship', 'Unknown'))
        
        prompt = f"""You are Agent {self.agent_id} proposing MANDATORY activity TYPES (no times).

PROFILE:
- Employment: {employment}
- Work schedule: {work_schedule}
- Education: {education}
- Primary activity: {self.person_info['primary_activity']}
- Can drive: {self.person_info['driver_on_travel_day']}

HOUSEHOLD:
- Young children: {self.household_info.get('young_children_count', 0)}
- Size: {self.household_info['household_size']}

TASK: Propose MANDATORY activity TYPES for today.

CRITICAL RULES:
1. Return ONLY activity TYPES - NO times, NO duration
2. Maximum 1-2 mandatory activities ONLY
3. Use ONLY: work, education{', dropoff_pickup' if has_young_children else ''}
4. {'Include dropoff_pickup ONLY if responsible for childcare' if has_young_children else 'NO dropoff_pickup (no children)'}
5. Do NOT include both 'work' and 'education' at the same time.
6. If Relationship is 'Child' and Education is 'Less than a high school graduate' or 'High school graduate or GED', include 'education' and exclude 'work'.

Output JSON format:
```json
{{
  "mandatory_activities": [{{"activity": "work"}}, {{"activity": "education"}}]
}}
```

If none, return: {{"mandatory_activities": []}}

Generate mandatory activity TYPES:"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a household member proposing mandatory activities."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                result_text = clean_json_response(response.choices[0].message.content.strip())
                result = json.loads(result_text)
                
                self.mandatory_activities = []
                for a in result.get("mandatory_activities", []):
                    if isinstance(a, dict): activity = a.get("activity")
                    elif isinstance(a, str): activity = a
                    else: continue
                    
                    if activity in ALLOWED_ACTIVITIES:
                        self.mandatory_activities.append({"activity": activity})

                # --- Enforce domain rule: no simultaneous work & education ---
                # Determine child + target education levels
                rel_norm = relationship.strip().lower()
                edu_norm = str(education).strip().lower()
                is_child = rel_norm.startswith('child')
                child_edu_levels = {
                    'less than a high school graduate',
                    'high school graduate or ged'
                }
                should_have_education = is_child and (edu_norm in child_edu_levels)

                # Current activity set
                acts = [d.get('activity') for d in self.mandatory_activities]
                acts_set = set(acts)

                if should_have_education:
                    # Ensure education present and remove work
                    if 'education' not in acts_set:
                        self.mandatory_activities.append({"activity": "education"})
                    # Remove any 'work'
                    self.mandatory_activities = [
                        a for a in self.mandatory_activities if a.get('activity') != 'work'
                    ]
                else:
                    # If both present, prefer 'work' only for employed non-child-student
                    both_present = ('work' in acts_set) and ('education' in acts_set)
                    emp_norm = str(employment).strip().lower()
                    is_employed = emp_norm == 'employed'
                    if both_present:
                        if is_employed:
                            # Keep work, drop education
                            self.mandatory_activities = [
                                a for a in self.mandatory_activities if a.get('activity') != 'education'
                            ]
                        else:
                            # If not employed, prefer education by default
                            self.mandatory_activities = [
                                a for a in self.mandatory_activities if a.get('activity') != 'work'
                            ]
                
                print(f"  → Agent {self.agent_id}: {len(self.mandatory_activities)} mandatory activity types")
                return True
            except Exception as e:
                print(f"   ⚠️ Agent {self.agent_id} mandatory error: {str(e)}")
                time.sleep(0.5)
        return False

    # --- Phase 2: Negotiation Methods ---
    def propose_initial(self, tasks_list: List[Dict[str, Any]], all_agents: Dict[str, 'Agent'], max_retries: int = 3) -> str:
        tasks_desc = ", ".join([t.get("activity", "task") for t in tasks_list])
        prompt = f"""You are household member {self.agent_id} in a family meeting allocating today's tasks.

PROFILE: {self.person_info['relationship']}, {self.person_info['age_range']}, {self.person_info['employment_status']}
SCHEDULE: {self.get_mandatory_summary()}

TASKS TODAY: {tasks_desc}

Naturally propose which task(s) you can handle based on your schedule. Be brief and reference family members by relationship.
Example: "I can do shopping after work. Maybe mom can handle the medical appointment since she's home earlier?"

Your proposal:"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=150
                )
                return response.choices[0].message.content.strip().strip('"')
            except Exception:
                time.sleep(1)
        return "I can help with some tasks if needed."

    def respond(self, tasks_list: List[Dict[str, Any]], all_agents: Dict[str, 'Agent'], max_retries: int = 3) -> Optional[str]:
        recent_conv = self.conversation_history[-6:] if len(self.conversation_history) > 6 else self.conversation_history
        conv_text = "\n".join([f"{h['speaker']}: {h['statement']}" for h in recent_conv])
        
        prompt = f"""You are {self.agent_id} continuing the family discussion about tasks.

YOUR PROFILE:
- Relationship: {self.person_info['relationship']}
- Employment: {self.person_info['employment_status']}
- Scheduled activities: {self.get_mandatory_summary()}

RECENT CONVERSATION:
{conv_text}

Based on what others said, respond naturally and briefly (1-2 sentences).
Response rules:
1. Focus on which tasks you can or cannot handle
2. Reference family members by relationship if relevant
3. Do NOT repeat what's already been discussed
Your response:"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    max_tokens=100
                )
                statement = response.choices[0].message.content.strip().strip('"')
                return statement if len(statement) >= 10 else None
            except Exception:
                time.sleep(1)
        return None

    # --- Phase 3: RAG Generation ---
    def generate_activity_chain(self, max_retries: int = 3) -> bool:
        """
        Generate full daily activity chain using RAG and Feedback Loop.
        Incorporates allocated tasks from Phase 2.
        """
        household_id = self.household_info['household_id']
        
        # 1. Retrieve other members' activities
        other_activities_text = self.rag_module.retrieve_household_activities(household_id, self.agent_id)
        other_schedules_parsed = self.rag_module.retrieve_parsed_schedules(household_id, self.agent_id)
        
        # 2. Build Prompt
        system_prompt = """You are an expert in human mobility and household coordination.
Generate a realistic 24-hour activity schedule for a household member.
Your goal is to create a schedule that is:
1. Logically consistent (no time gaps, no overlaps).
2. Statistically realistic (based on NHTS data).
3. Coordinated with other household members (shared meals, rides, etc.).

OUTPUT FORMAT:
Provide the schedule as a JSON object with a "schedule" key containing a list of activities.
Each activity must have: "activity", "start_time", "end_time", "participants".
"participants": List of household members joining (e.g., ["Spouse", "Child 1"]). If alone, use [].

Example:
```json
{
  "schedule": [
    {"activity": "home", "start_time": "00:00", "end_time": "07:30", "participants": ["Spouse"]},
    {"activity": "work", "start_time": "07:30", "end_time": "17:00", "participants": []},
    {"activity": "dine_out", "start_time": "18:00", "end_time": "19:30", "participants": ["Spouse", "Child 1"]},
    {"activity": "home", "start_time": "19:30", "end_time": "24:00", "participants": ["Spouse", "Child 1"]}
  ]
}
```
"""
        
        # Prepare allocated tasks string
        allocated_tasks_str = "None"
        if self.allocated_tasks:
            allocated_tasks_str = ", ".join([t.get('activity', 'task') for t in self.allocated_tasks])

        # Construct User Prompt
        user_prompt = f"""# Profile
{self.get_profile_summary()}

# Mandatory Commitments
{self.get_mandatory_summary()}

# Household Context
- Size: {self.household_info['household_size']}
- Children: {self.household_info['young_children_count']}
- Vehicles: {self.household_info['vehicle_count']}

# Unfulfilled Coordination Tasks (Allocated via Negotiation)
You MUST include these activities in your schedule:
- {allocated_tasks_str}

# Other Household Members' Schedules (ALREADY GENERATED)
"""
        if other_activities_text:
            for uid, text in other_activities_text.items():
                user_prompt += f"--- Member {uid} ---\n{text}\n"
            
            user_prompt += """
# Coordination Opportunities
Based on the schedules above, identify opportunities or requirements for coordination:
1. **Joint Meals**: If others are eating at home or dining out, consider joining them.
2. **Shared Rides**: If a child needs dropoff/pickup and you are a driver, consider this.
3. **Family Time**: Evenings and weekends often involve shared leisure/socializing.
4. **Consistency**: If you say you are with someone, their schedule MUST show they are available (not at work/school).
"""
        else:
            user_prompt += "(No other members generated yet. You are the first one.)\n"

        user_prompt += """
# Instructions
Generate the full day schedule (00:00 - 24:00).
- Ensure work/school hours match the profile.
- If 'Employment' is 'Employed', include 'work'.
- If 'Age' indicates student, include 'education'.
- **CRITICAL**: You MUST schedule the 'Unfulfilled Coordination Tasks' listed above.
- Explicitly list "participants" for each activity to ensure coordination.
     - Do NOT include both 'work' and 'education' on the same day.
     - If Relationship is 'Child' AND Education is 'Less than a high school graduate' OR 'High school graduate or GED', you MUST include 'education' and MUST NOT include 'work'.
"""

        # 3. Generation Loop with Feedback
        for attempt in range(max_retries):
            try:
                # Call LLM
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                
                result_text = response.choices[0].message.content.strip()
                cleaned_json = clean_json_response(result_text)
                result_data = json.loads(cleaned_json)
                schedule = result_data.get("schedule", [])
                
                # 4. Consistency Check (Feedback Loop)
                is_valid, feedback = self.check_consistency(schedule, other_schedules_parsed)
                
                if is_valid:
                    self.final_schedule = schedule
                    # Store raw text for RAG (simplified representation)
                    self.raw_trajectory_text = self.format_schedule_to_text(schedule)
                    self.rag_module.store_generated_activity(household_id, self.agent_id, self.raw_trajectory_text, schedule)
                    print(f"  ✓ Agent {self.agent_id} generated valid schedule.")
                    return True
                else:
                    print(f"  ⚠️ Agent {self.agent_id} consistency check failed: {feedback}")
                    # Add feedback to prompt and retry
                    user_prompt += f"\n\n# FEEDBACK - PLEASE FIX:\n{feedback}\nEnsure your schedule is consistent with others."
            
            except Exception as e:
                print(f"  ❌ Agent {self.agent_id} generation error (attempt {attempt+1}): {e}")
                time.sleep(1)
        
        return False

    def check_consistency(self, my_schedule: List[Dict], other_schedules: Dict[str, List[Dict]]) -> tuple:
        """
        Check logical consistency of the generated schedule against others.
        Returns (bool, feedback_string)
        """
        feedback = []
        
        # 0. Global constraint: no simultaneous 'work' and 'education'
        activities_all = [act.get('activity') for act in my_schedule]
        if 'work' in activities_all and 'education' in activities_all:
            rel_norm = str(self.person_info.get('relationship', '')).strip().lower()
            edu_norm = str(self.person_info.get('education', '')).strip().lower()
            emp_norm = str(self.person_info.get('employment_status', '')).strip().lower()
            child_edu_levels = {
                'less than a high school graduate',
                'high school graduate or ged'
            }
            is_child_student = rel_norm.startswith('child') and (edu_norm in child_edu_levels)
            preferred = 'education' if is_child_student else ('work' if emp_norm == 'employed' else 'education')
            feedback.append(
                "You included both 'work' and 'education' which is not allowed. "
                f"Keep '{preferred}' only according to profile rules."
            )
        
        # 1. Check if allocated tasks are included
        if self.allocated_tasks:
            my_activities = [act.get('activity') for act in my_schedule]
            for task in self.allocated_tasks:
                task_act = task.get('activity')
                if task_act and task_act not in my_activities:
                    feedback.append(f"You were allocated task '{task_act}' but did not include it in your schedule.")

        # 2. Check Coordination Consistency
        for my_act in my_schedule:
            my_start = my_act.get("start_time")
            my_end = my_act.get("end_time")
            participants = my_act.get("participants", [])
            
            if len(participants) > 0 and other_schedules:
                match_found = False
                potential_conflicts = []
                
                for other_id, other_sched in other_schedules.items():
                    for other_act in other_sched:
                        if self.check_overlap(my_start, my_end, other_act.get("start_time"), other_act.get("end_time")):
                            if other_act.get("activity") == my_act.get("activity"):
                                match_found = True
                            else:
                                potential_conflicts.append(f"Member {other_id} is doing {other_act.get('activity')}")
                
                if not match_found and my_act.get("activity") not in ["home"]: 
                     if potential_conflicts:
                         feedback.append(f"At {my_start}-{my_end}, you planned '{my_act['activity']}' with participants, but other members are busy: {'; '.join(potential_conflicts[:2])}.")

        if feedback:
            return False, "\n".join(feedback)
        return True, ""

    def check_overlap(self, start1, end1, start2, end2):
        """Check if two time ranges overlap"""
        try:
            s1 = int(start1.split(':')[0]) * 60 + int(start1.split(':')[1])
            e1 = int(end1.split(':')[0]) * 60 + int(end1.split(':')[1])
            s2 = int(start2.split(':')[0]) * 60 + int(start2.split(':')[1])
            e2 = int(end2.split(':')[0]) * 60 + int(end2.split(':')[1])
            return max(s1, s2) < min(e1, e2)
        except:
            return False

    def format_schedule_to_text(self, schedule: List[Dict]) -> str:
        lines = []
        for act in schedule:
            participants = ", ".join(act.get("participants", []))
            part_str = f" (with {participants})" if participants else ""
            lines.append(f"[{act['activity']}, {act['start_time']}-{act['end_time']}]{part_str}")
        return "\n".join(lines)

# ===== Helper Functions for Negotiation =====
def propose_household_tasks(client, household_info: Dict, member_infos: Dict, max_retries: int = 3) -> tuple:
    members_summary = "\n".join([
        f"- {uid}: {info['age_range']}, {info['employment_status']}, Driver: {info['driver_on_travel_day']}"
        for uid, info in member_infos.items()
    ])
    
    has_young_children = household_info.get('young_children_count', 0) > 0
    household_size = household_info.get('household_size', 2)
    max_tasks = max(1, min(2, household_size))
    
    if has_young_children: allowed_activities_str = "shopping, medical, dropoff_pickup"
    else: allowed_activities_str = "shopping, medical"
    
    prompt = f"""Generate AT MOST {max_tasks} household task types for this family TODAY.

Household:
- Size: {household_info['household_size']} people
- Young children: {household_info['young_children_count']}

Members:
{members_summary}

Available task types: {allowed_activities_str}

DECISION LOGIC:
1. shopping: MOST COMMON 
2. medical: LEAST COMMON 
3. {'dropoff_pickup: Include if young children need transportation' if has_young_children else ''}

RULES:
- Generate ONLY {max_tasks} task(s) maximum
- Return activity TYPES only (no duration, no time)
- shopping is the PRIMARY household maintenance task
- medical is RARE 
Output format (NO other text):
```json
{{
  "household_tasks": [
    {{
      "task_id": "shopping",
      "activity": "shopping"
    }}
  ]
}}
```"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Generate tasks based on actual household needs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=400
            )
            result_text = clean_json_response(response.choices[0].message.content.strip())
            result = json.loads(result_text)
            
            tasks = result.get("household_tasks", [])
            valid_tasks = []
            for task in tasks:
                activity = task.get("activity", "")
                task_id = task.get("task_id", activity)
                mapped_activity = map_task_to_activity(task_id)
                if mapped_activity == "dropoff_pickup" and not has_young_children: continue
                if mapped_activity in ALLOWED_ACTIVITIES:
                    valid_tasks.append({"activity": mapped_activity})
                if len(valid_tasks) >= max_tasks: break
            
            result["household_tasks"] = valid_tasks
            return True, result
        except Exception:
            time.sleep(0.5)
    return False, {"household_tasks": []}

def negotiate_task_allocation_multiagent(agents: Dict[str, Agent], household_tasks: Dict, all_agents: Dict[str, Agent], max_rounds: int = 2) -> tuple:
    tasks_list = household_tasks.get("household_tasks", [])
    if not tasks_list:
        return True, {"family_discussion": [], "task_allocations": []}
    
    member_ids = list(agents.keys())
    conversation = []
    print(f"      Multi-agent negotiation using Agent instances...")
    
    # Round 1: Initial proposals 
    for uid in member_ids:
        agent = agents[uid]
        statement = agent.propose_initial(tasks_list, all_agents)
        conversation.append({"speaker": uid, "statement": statement})
        print(f"         {uid}: {statement[:60]}...")
        for a in agents.values(): a.update_conversation_history(conversation)
        time.sleep(0.1)
    
    # Rounds 2-N: Discussion
    for round_num in range(2, max_rounds):
        for uid in member_ids:
            agent = agents[uid]
            statement = agent.respond(tasks_list, all_agents)
            if statement:
                conversation.append({"speaker": uid, "statement": statement})
                print(f"         {uid}: {statement[:60]}...")
                for a in agents.values(): a.update_conversation_history(conversation)
                time.sleep(0.1)
    
    allocations = extract_allocations_from_conversation(agents[member_ids[0]].client, member_ids, tasks_list, conversation)
    
    allocations_by_member = defaultdict(list)
    for alloc in allocations:
        assigned_to = alloc.get("assigned_to")
        if isinstance(assigned_to, list): assigned_to = assigned_to[0] if assigned_to else None
        if assigned_to and assigned_to in agents:
            allocations_by_member[assigned_to].append(alloc)
            agents[assigned_to].set_allocated_tasks(allocations_by_member[assigned_to])
            print(f"      → Agent {assigned_to}: Allocated task {alloc.get('activity', 'unknown')}")
    
    return True, {"family_discussion": conversation, "task_allocations": allocations}

def extract_allocations_from_conversation(client, member_ids: List[str], tasks_list: List[Dict], conversation: List[Dict]) -> List[Dict]:
    conv_text = "\n".join([f"{h['speaker']}: {h['statement']}" for h in conversation])
    tasks_desc = "\n".join([f"- {t.get('activity', 'unknown')}" for t in tasks_list])
    
    prompt = f"""Analyze this family conversation and extract who will do which tasks:

CONVERSATION:
{conv_text}

TASKS (activity types):
{tasks_desc}

MEMBERS: {', '.join(member_ids)}

Output JSON:
```json
{{
  "allocations": [
    {{
      "activity": "shopping",
      "assigned_to": "30007884_1",
      "reasoning": "Volunteered to handle"
    }}
  ]
}}
```"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        result_text = clean_json_response(response.choices[0].message.content.strip())
        result = json.loads(result_text)
        return result.get("allocations", [])
    except Exception:
        return [{
            "activity": task.get("activity", "unknown"),
            "assigned_to": member_ids[i % len(member_ids)],
            "reasoning": "Default allocation"
        } for i, task in enumerate(tasks_list)]

# ==================== Main Processing ====================
def process_household_rag(
    client,
    household_id: str,
    member_user_ids: List[str],
    persons_dict: Dict,
    households_dict: Dict
) -> Dict:
    print(f"\n{'='*70}")
    print(f" Household: {household_id} (Hybrid: Negotiation + RAG)")
    print(f" Members: {len(member_user_ids)}")
    print(f"{'='*70}")
    
    household_info = households_dict.get(household_id, {})
    if 'household_size' not in household_info:
        household_info['household_size'] = len(member_user_ids)
        
    rag_module = RetrievalAugmentedLLM()
    member_infos = {uid: persons_dict[uid] for uid in member_user_ids}
    
    # Create Agents
    agents = {}
    for uid in member_user_ids:
        agent = Agent(
            agent_id=uid,
            person_info=persons_dict[uid],
            household_info=household_info,
            rag_module=rag_module,
            openai_client=client
        )
        agents[uid] = agent

    # --- Phase 1: Mandatory Activities ---
    print(f"\n PHASE 1: Mandatory Activities")
    for uid in member_user_ids:
        agents[uid].propose_mandatory_activities()

    # --- Phase 2: Negotiation & Task Allocation ---
    print(f"\n PHASE 2: Negotiation & Task Allocation")
    success, household_tasks = propose_household_tasks(client, household_info, member_infos)
    if success and household_tasks.get("household_tasks"):
        print(f"  Generated tasks: {[t['activity'] for t in household_tasks['household_tasks']]}")
        negotiate_task_allocation_multiagent(agents, household_tasks, agents)
    else:
        print("  No household tasks generated.")

    # --- Phase 3: RAG Sequential Generation ---
    print(f"\n PHASE 3: RAG Sequential Generation")
    # Sort members: Head/Spouse first, then children
    sorted_members = sorted(member_user_ids, key=lambda uid: persons_dict[uid].get('age_range', '0'), reverse=True)
    
    schedules = {}
    for idx, uid in enumerate(sorted_members, 1):
        print(f"  [{idx}/{len(sorted_members)}] Generating for Agent {uid}...")
        success = agents[uid].generate_activity_chain()
        if success:
            schedules[uid] = {"full_schedule": agents[uid].final_schedule}
        else:
            print(f"    ❌ Failed to generate schedule for {uid}")
            schedules[uid] = {"full_schedule": []}
        time.sleep(0.5)

    return {
        "household_id": household_id,
        "phase4_schedules": schedules,
        "generation_time": datetime.now().isoformat(),
        "method": "Hybrid_Negotiation_RAG"
    }

# ==================== Utils ====================
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_household_id(user_id):
    return user_id.split('_')[0] if '_' in user_id else user_id

def create_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)

def save_individual_trajectories(results: List[Dict], timestamp: str):
    os.makedirs(OUTPUT_TRAJECTORIES_DIR, exist_ok=True)
    all_trajectories = []
    for household_result in results:
        schedules = household_result.get("phase4_schedules", {})
        for user_id, schedule_data in schedules.items():
            full_schedule = schedule_data.get("full_schedule", [])
            if full_schedule:
                trajectory = {
                    "user_id": user_id,
                    "schedule": full_schedule
                }
                all_trajectories.append(trajectory)
    
    if all_trajectories:
        trajectory_filename = f"all_trajectories_{timestamp}.json"
        trajectory_filepath = os.path.join(OUTPUT_TRAJECTORIES_DIR, trajectory_filename)
        with open(trajectory_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_trajectories, f, indent=2, ensure_ascii=False)
        print(f"\n    ✓ Saved all trajectories → {trajectory_filename}")
    return len(all_trajectories)

def main():
    print("="*70)
    print(" 🚀 Hybrid Multi-Agent Negotiation + RAG Generation")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    persons_list = load_json(PERSON_FILE)
    households_list = load_json(HOUSEHOLD_FILE)
    
    persons_dict = {p["user_id"]: p for p in persons_list if p.get("user_id")}
    households_dict = {h["household_id"]: h for h in households_list if h.get("household_id")}
    
    # Select households
    selected_households = []
    
    if GENERATION_MODE == "household_file":
        print(f"📂 Reading household IDs from: {HOUSEHOLD_ID_FILE}")
        if os.path.exists(HOUSEHOLD_ID_FILE):
            with open(HOUSEHOLD_ID_FILE, 'r') as f:
                target_ids = json.load(f)
                if target_ids and isinstance(target_ids[0], dict):
                     hids = set()
                     for t in target_ids:
                         uid = t.get('user_id')
                         if uid: hids.add(extract_household_id(uid))
                     target_ids = list(hids)
            
            for hid in target_ids:
                members = [uid for uid, p in persons_dict.items() if extract_household_id(uid) == str(hid)]
                if members and len(members) >= 2:
                    selected_households.append((hid, members))
        else:
            print(f"⚠️ File {HOUSEHOLD_ID_FILE} not found.")

    elif GENERATION_MODE == "random":
        print("🎲 Randomly selecting 20 households...")
        all_household_ids = list(households_dict.keys())
        random.shuffle(all_household_ids)
        
        count = 0
        for hid in all_household_ids:
            members = [uid for uid, p in persons_dict.items() if extract_household_id(uid) == str(hid)]
            if members and len(members) >= 2:
                selected_households.append((hid, members))
                count += 1
                if count >= 20:
                    break 
    
    print(f"🎯 Selected {len(selected_households)} households for processing.")
    
    client = create_openai_client()
    results = []
    
    for idx, (hid, members) in enumerate(selected_households, 1):
        try:
            result = process_household_rag(client, hid, members, persons_dict, households_dict)
            results.append(result)
        except Exception as e:
            print(f"Error processing {hid}: {e}")
            
    if results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"hybrid_generation_{timestamp}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        save_individual_trajectories(results, timestamp)
        print(f"Done. Saved to {output_file}")

if __name__ == "__main__":
    main()
