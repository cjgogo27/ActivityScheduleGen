"""
Three-Agent Pipeline for Individual Trajectory Generation
===========================================================

Pipeline Architecture:
1. Planner Agent: Plans activity sequence skeleton (no times)
2. Trip Realizer Agent: Fills in start_time, end_time, and location
3. Critic & Editor Agent: Validates constraints and refines the schedule

Constraints:
- Physical (Hard): No time overlaps, must end at 24:00, full 24-hour coverage
- Logical (Hard): Must start and end at home
- Common Sense (Soft): Age-appropriate, employment-appropriate activities
- Temporal (Soft): Realistic activity durations
- Coherence (Soft): Logical activity transitions
"""

import json
import time
import os
import random
from openai import OpenAI
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

# ==================== Configuration ====================
API_KEY = "sk-d8F4cfrU3WymR4j0eaqDFfjka6Dj9W2rsTp5uK18qSJN1IaG"
BASE_URL = "https://api.nuwaflux.com/v1"
TIMEOUT = 60

PERSON_FILE = r"/data/alice/cjtest/FinalTraj/Oklahoma/processed_data/oklahoma_person_static.json"
OUTPUT_DIR = r"/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/output_three_agent"
OUTPUT_TRAJECTORIES_DIR = r"/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/output_trajectories_three_agent"

ALLOWED_ACTIVITIES = {
    "home", "work", "education", "shopping", "service", 
    "medical", "dine_out", "socialize", "exercise", "dropoff_pickup"
}

# ==================== Utility Functions ====================
def create_openai_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)

def load_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def time_to_minutes(time_str: str) -> int:
    """Convert HH:MM to minutes since midnight"""
    try:
        h, m = map(int, time_str.split(':'))
        return h * 60 + m
    except:
        return 0

def minutes_to_time(minutes: int) -> str:
    """Convert minutes since midnight to HH:MM"""
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"

def extract_person_info(person_data: Dict) -> Dict[str, Any]:
    """Extract relevant person information from raw data"""
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


# ==================== Agent A: Planner ====================
class PlannerAgent:
    """
    Plans the activity sequence skeleton based on user profile.
    Output: List of activity types (no times yet)
    """
    
    def __init__(self, client: OpenAI, person_info: Dict[str, Any]):
        self.client = client
        self.person_info = person_info
        self.activity_skeleton = []
    
    def plan_activities(self, max_retries: int = 3) -> bool:
        """Generate activity sequence skeleton using Chain-of-Thought reasoning"""
        
        employment = self.person_info.get('employment_status', 'Unknown')
        primary_activity = self.person_info.get('primary_activity', 'Unknown')
        age_range = self.person_info.get('age_range', 'Unknown')
        work_schedule = self.person_info.get('work_schedule', 'Unknown')
        
        prompt = f"""You are a Planner Agent using Chain-of-Thought reasoning to plan a person's daily activities.

USER PROFILE:
- Age: {age_range}
- Employment: {employment}
- Primary Activity: {primary_activity}
- Work Schedule: {work_schedule}
- Can Drive: {self.person_info.get('driver_on_travel_day', 'Unknown')}

TASK: Plan a realistic activity sequence for ONE DAY.

REASONING PROCESS (Chain-of-Thought):
1. What is this person's primary obligation? (work/school/retired/child)
2. What household/personal maintenance do they need? (shopping, meals)
3. What optional social/leisure activities might they do?

RULES:
- Output ONLY activity TYPES (no times, no locations yet)
- Keep it SIMPLE and REALISTIC (3-5 activities total)
- Available activities: {', '.join(sorted(ALLOWED_ACTIVITIES))}
- Must include "home" as both first and last activity
- Consider age and employment appropriateness

EXAMPLES:

Retired person → Reasoning: "Retired, no work obligation. Likely social activities or errands."
Activities: [home, shopping, socialize, home]

Full-time worker → Reasoning: "Full-time employed, work is primary activity today."
Activities: [home, work, home]

Student → Reasoning: "Student, education is primary, may have social time after."
Activities: [home, education, socialize, home]

NOW PLAN FOR THIS USER:

Step 1 - Reasoning (your thinking):
[Explain why this person would do certain activities]

Step 2 - Activity Skeleton:
```json
{{
  "reasoning": "Brief explanation of your planning logic",
  "activities": ["home", "...", "home"]
}}
```

Generate plan:"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-5.2",
                    messages=[
                        {"role": "system", "content": "You are a thoughtful planner who uses reasoning to plan realistic daily activities."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=600
                )
                
                # Handle different response types
                if isinstance(response, str):
                    result_text = response
                elif hasattr(response, 'choices'):
                    result_text = response.choices[0].message.content.strip()
                else:
                    # Try to access as dict
                    result_text = response['choices'][0]['message']['content'].strip()
                
                # Debug: print raw response
                if attempt == 0:
                    print(f"  [DEBUG] Raw response length: {len(result_text)}")
                    print(f"  [DEBUG] First 200 chars: {result_text[:200]}")
                
                # Extract JSON
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                
                # Try to parse JSON
                if not result_text:
                    raise ValueError("Empty response after extraction")
                
                result = json.loads(result_text)
                activities = result.get("activities", [])
                
                # Validate and clean
                valid_activities = [act for act in activities if act in ALLOWED_ACTIVITIES]
                
                # Ensure home is first and last
                if valid_activities and valid_activities[0] != "home":
                    valid_activities.insert(0, "home")
                if valid_activities and valid_activities[-1] != "home":
                    valid_activities.append("home")
                
                # Default if empty
                if not valid_activities:
                    valid_activities = ["home"]
                
                self.activity_skeleton = valid_activities
                print(f"  ✓ Planner: {len(self.activity_skeleton)} activities planned")
                print(f"    Skeleton: {' → '.join(self.activity_skeleton)}")
                return True
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️ Planner attempt {attempt+1} JSON error: {str(e)}")
                print(f"      Response text: {result_text[:300] if 'result_text' in locals() else 'N/A'}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    # Fallback
                    self.activity_skeleton = ["home"]
                    print(f"  ⚠️ Planner using fallback: {self.activity_skeleton}")
                    return False
            except Exception as e:
                print(f"  ⚠️ Planner attempt {attempt+1} failed: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    # Fallback
                    self.activity_skeleton = ["home"]
                    print(f"  ⚠️ Planner using fallback: {self.activity_skeleton}")
                    return False
        
        return False


# ==================== Agent B: Trip Realizer ====================
class TripRealizerAgent:
    """
    Fills in temporal details (start_time, end_time) for the activity skeleton.
    """
    
    def __init__(self, client: OpenAI, person_info: Dict[str, Any]):
        self.client = client
        self.person_info = person_info
        self.schedule = []
    
    def realize_schedule(self, activity_skeleton: List[str], max_retries: int = 3) -> bool:
        """Convert activity skeleton to full schedule with times"""
        
        if not activity_skeleton:
            return False
        
        employment = self.person_info.get('employment_status', 'Unknown')
        work_schedule = self.person_info.get('work_schedule', 'Unknown')
        
        activities_str = ' → '.join(activity_skeleton)
        
        prompt = f"""You are a Trip Realizer Agent. Fill in start/end times for this activity sequence.

USER PROFILE:
- Employment: {employment}
- Work Schedule: {work_schedule}

ACTIVITY SKELETON: {activities_str}

TASK: Assign realistic start_time and end_time to each activity.

CRITICAL RULES:
1. Cover FULL 24 hours: 00:00 to 24:00 with NO gaps
2. Use realistic times with minutes (e.g., 07:30, not 07:00)
3. Consecutive activities must connect seamlessly (end_time = next start_time)
4. Activity durations should be realistic:
   - home (night): 6-9 hours
   - work (full-time): 8-9 hours
   - work (part-time): 3-4 hours
   - shopping/medical/service: 1-2 hours
   - dine_out/socialize: 1-3 hours
   - education: 4-7 hours

WORK TIME GUIDANCE:
- Full-time: typically 08:00-17:00 (but vary start time!)
- Part-time: often evening/night shifts (15:00-19:00)
- If work_from_home = Yes: work happens AT home (not separate work activity)

OUTPUT FORMAT:
```json
{{
  "schedule": [
    {{"activity": "home", "start_time": "00:00", "end_time": "08:00"}},
    {{"activity": "work", "start_time": "08:00", "end_time": "17:00"}},
    {{"activity": "home", "start_time": "17:00", "end_time": "24:00"}}
  ]
}}
```

Generate schedule with times:"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-5.2",
                    messages=[
                        {"role": "system", "content": "You assign realistic times to activities, ensuring full 24-hour coverage."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=800
                )
                
                # Handle different response types
                if isinstance(response, str):
                    result_text = response
                elif hasattr(response, 'choices'):
                    result_text = response.choices[0].message.content.strip()
                else:
                    result_text = response['choices'][0]['message']['content'].strip()
                
                # Extract JSON
                if "```json" in result_text:
                    json_start = result_text.find("```json") + 7
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                elif "```" in result_text:
                    json_start = result_text.find("```") + 3
                    json_end = result_text.find("```", json_start)
                    result_text = result_text[json_start:json_end].strip()
                
                result = json.loads(result_text)
                schedule = result.get("schedule", [])
                
                # Validate schedule
                if self._validate_basic_schedule(schedule):
                    self.schedule = schedule
                    print(f"  ✓ Trip Realizer: Generated schedule with {len(self.schedule)} segments")
                    return True
                else:
                    print(f"  ⚠️ Trip Realizer attempt {attempt+1}: Invalid schedule structure")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    else:
                        # Create fallback schedule
                        self.schedule = self._create_fallback_schedule(activity_skeleton)
                        print(f"  ⚠️ Using fallback schedule")
                        return False
                        
            except Exception as e:
                print(f"  ⚠️ Trip Realizer attempt {attempt+1} failed: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    self.schedule = self._create_fallback_schedule(activity_skeleton)
                    return False
        
        return False
    
    def _validate_basic_schedule(self, schedule: List[Dict]) -> bool:
        """Basic validation of schedule structure"""
        if not schedule:
            return False
        
        # Check all have required fields
        for seg in schedule:
            if not all(key in seg for key in ['activity', 'start_time', 'end_time']):
                return False
            if seg['activity'] not in ALLOWED_ACTIVITIES:
                return False
        
        return True
    
    def _create_fallback_schedule(self, activity_skeleton: List[str]) -> List[Dict]:
        """Create simple fallback schedule"""
        if not activity_skeleton or activity_skeleton == ["home"]:
            return [{"activity": "home", "start_time": "00:00", "end_time": "24:00"}]
        
        # Simple equal division
        num_activities = len(activity_skeleton)
        minutes_per_activity = 1440 // num_activities
        
        schedule = []
        current_time = 0
        
        for i, activity in enumerate(activity_skeleton):
            start = current_time
            if i == num_activities - 1:
                end = 1440  # Last activity ends at 24:00
            else:
                end = start + minutes_per_activity
            
            schedule.append({
                "activity": activity,
                "start_time": minutes_to_time(start),
                "end_time": minutes_to_time(end)
            })
            current_time = end
        
        return schedule


# ==================== Agent C: Critic & Editor ====================
class CriticEditorAgent:
    """
    Validates schedule against constraints and applies edits to fix violations.
    Uses Chain-of-Thought for constraint checking and editing process.
    """
    
    def __init__(self, client: OpenAI, person_info: Dict[str, Any]):
        self.client = client
        self.person_info = person_info
        self.final_schedule = []
        self.constraint_violations = []
    
    def check_and_edit(self, initial_schedule: List[Dict], max_iterations: int = 3) -> bool:
        """
        Iterative refinement process:
        1. Check constraints
        2. If violations found, apply edits
        3. Repeat until valid or max iterations
        """
        
        current_schedule = initial_schedule.copy()
        
        for iteration in range(max_iterations):
            print(f"  📝 Critic iteration {iteration + 1}:")
            
            # Step 1: Check all constraints
            violations = self._check_all_constraints(current_schedule)
            
            if not violations:
                print(f"    ✓ All constraints satisfied!")
                self.final_schedule = current_schedule
                return True
            
            print(f"    ⚠️ Found {len(violations)} violation(s)")
            for v in violations:
                print(f"      - {v}")
            
            # Step 2: Apply edits to fix violations
            edited_schedule = self._apply_edits(current_schedule, violations)
            
            if edited_schedule == current_schedule:
                # No changes made, can't improve further
                print(f"    ⚠️ Cannot fix violations, using best effort")
                self.final_schedule = current_schedule
                self.constraint_violations = violations
                return False
            
            current_schedule = edited_schedule
        
        # Max iterations reached
        print(f"  ⚠️ Max iterations reached, some violations may remain")
        self.final_schedule = current_schedule
        self.constraint_violations = self._check_all_constraints(current_schedule)
        return len(self.constraint_violations) == 0
    
    def _check_all_constraints(self, schedule: List[Dict]) -> List[str]:
        """Check all constraint types and return list of violations"""
        violations = []
        
        # 1. Physical Constraints (Hard)
        violations.extend(self._check_physical_constraints(schedule))
        
        # 2. Logical Constraints (Hard)
        violations.extend(self._check_logical_constraints(schedule))
        
        # 3. Common Sense Constraints (Soft)
        violations.extend(self._check_common_sense_constraints(schedule))
        
        # 4. Temporal Constraints (Soft)
        violations.extend(self._check_temporal_constraints(schedule))
        
        # 5. Coherence Constraints (Soft)
        violations.extend(self._check_coherence_constraints(schedule))
        
        return violations
    
    def _check_physical_constraints(self, schedule: List[Dict]) -> List[str]:
        """Physical: No overlaps, must end at 24:00, full 24-hour coverage"""
        violations = []
        
        if not schedule:
            return ["Physical: Schedule is empty"]
        
        # Check last activity ends at 24:00 or 00:00 next day
        last_end = schedule[-1].get('end_time', '')
        if last_end not in ['24:00', '00:00']:
            violations.append(f"Physical: Schedule must end at 24:00, but ends at {last_end}")
        
        # Check no overlaps and no gaps
        for i in range(len(schedule) - 1):
            curr_end = time_to_minutes(schedule[i]['end_time'])
            next_start = time_to_minutes(schedule[i+1]['start_time'])
            
            if curr_end > next_start:
                violations.append(f"Physical: Time overlap between {schedule[i]['activity']} and {schedule[i+1]['activity']}")
            elif curr_end < next_start:
                gap_minutes = next_start - curr_end
                violations.append(f"Physical: {gap_minutes}-minute gap between {schedule[i]['activity']} and {schedule[i+1]['activity']}")
        
        # Check total duration covers 24 hours
        total_minutes = 0
        for seg in schedule:
            start_min = time_to_minutes(seg.get('start_time', '00:00'))
            end_min = time_to_minutes(seg.get('end_time', '00:00'))
            total_minutes += (end_min - start_min)
        
        if total_minutes != 1440:
            violations.append(f"Physical: Schedule covers {total_minutes} minutes, not 1440 (24 hours)")
        
        return violations
    
    def _check_logical_constraints(self, schedule: List[Dict]) -> List[str]:
        """Logical: Must start and end at home"""
        violations = []
        
        if not schedule:
            return violations
        
        if schedule[0].get('activity') != 'home':
            violations.append(f"Logical: Must start at home, but starts at {schedule[0].get('activity')}")
        
        if schedule[-1].get('activity') != 'home':
            violations.append(f"Logical: Must end at home, but ends at {schedule[-1].get('activity')}")
        
        # Check first activity starts at 00:00
        if schedule[0].get('start_time') != '00:00':
            violations.append(f"Logical: Must start at 00:00, but starts at {schedule[0].get('start_time')}")
        
        return violations
    
    def _check_common_sense_constraints(self, schedule: List[Dict]) -> List[str]:
        """Common sense: Age and employment appropriate activities"""
        violations = []
        
        age_range = self.person_info.get('age_range', '')
        employment = self.person_info.get('employment_status', '')
        primary_activity = self.person_info.get('primary_activity', '')
        
        # Check for work activity in retired/unemployed/student
        has_work = any(seg['activity'] == 'work' for seg in schedule)
        
        if has_work:
            if 'retired' in primary_activity.lower() or 'retired' in employment.lower():
                violations.append("Common Sense: Retired person should not have work activity")
            elif 'unemployed' in employment.lower():
                violations.append("Common Sense: Unemployed person should not have work activity")
            elif 'not in labor force' in employment.lower():
                violations.append("Common Sense: Person not in labor force should not have work activity")
        
        # Check for driving activities if cannot drive
        can_drive = self.person_info.get('driver_on_travel_day', '')
        if can_drive == 'No':
            # In real system, would check if activities require driving
            pass
        
        return violations
    
    def _check_temporal_constraints(self, schedule: List[Dict]) -> List[str]:
        """Temporal: Realistic activity durations"""
        violations = []
        
        for seg in schedule:
            activity = seg.get('activity')
            start_min = time_to_minutes(seg.get('start_time', '00:00'))
            end_min = time_to_minutes(seg.get('end_time', '00:00'))
            duration_min = end_min - start_min
            
            # Define reasonable duration ranges (in minutes)
            duration_ranges = {
                'work': (180, 600),  # 3-10 hours
                'education': (180, 480),  # 3-8 hours
                'shopping': (15, 180),  # 15min - 3 hours
                'medical': (15, 240),  # 15min - 4 hours
                'service': (15, 180),
                'dine_out': (30, 180),
                'socialize': (30, 360),
                'exercise': (15, 180),
                'home': (60, 1440),  # Can be all day
                'dropoff_pickup': (10, 60)
            }
            
            if activity in duration_ranges:
                min_dur, max_dur = duration_ranges[activity]
                if duration_min < min_dur:
                    violations.append(f"Temporal: {activity} too short ({duration_min}min < {min_dur}min)")
                elif duration_min > max_dur:
                    violations.append(f"Temporal: {activity} too long ({duration_min}min > {max_dur}min)")
        
        return violations
    
    def _check_coherence_constraints(self, schedule: List[Dict]) -> List[str]:
        """Coherence: Logical activity transitions"""
        violations = []
        
        # Check for too many fragments (indicates over-segmentation)
        if len(schedule) > 10:
            violations.append(f"Coherence: Too many activity segments ({len(schedule)}), consider merging similar activities")
        
        return violations
    
    def _apply_edits(self, schedule: List[Dict], violations: List[str]) -> List[Dict]:
        """Apply rule-based edits to fix violations"""
        edited = schedule.copy()
        
        # Fix: Must start at 00:00
        if edited and edited[0].get('start_time') != '00:00':
            edited[0]['start_time'] = '00:00'
        
        # Fix: Must end at 24:00
        if edited and edited[-1].get('end_time') not in ['24:00', '00:00']:
            edited[-1]['end_time'] = '24:00'
        
        # Fix: Must start/end with home
        if edited and edited[0].get('activity') != 'home':
            # Check if we can extend first home segment
            first_home_idx = next((i for i, seg in enumerate(edited) if seg['activity'] == 'home'), None)
            if first_home_idx is not None and first_home_idx > 0:
                # Move first home to start
                edited[first_home_idx]['start_time'] = '00:00'
                edited = edited[first_home_idx:] + edited[:first_home_idx]
            else:
                # Insert home at start
                edited.insert(0, {'activity': 'home', 'start_time': '00:00', 'end_time': edited[0]['start_time']})
        
        if edited and edited[-1].get('activity') != 'home':
            # Extend or add home at end
            last_home_idx = next((i for i in range(len(edited)-1, -1, -1) if edited[i]['activity'] == 'home'), None)
            if last_home_idx is not None:
                edited[last_home_idx]['end_time'] = '24:00'
                edited = edited[:last_home_idx+1]
            else:
                edited.append({'activity': 'home', 'start_time': edited[-1]['end_time'], 'end_time': '24:00'})
        
        # Fix gaps: Connect consecutive activities
        for i in range(len(edited) - 1):
            edited[i+1]['start_time'] = edited[i]['end_time']
        
        # Remove work for retired/unemployed
        employment = self.person_info.get('employment_status', '').lower()
        primary = self.person_info.get('primary_activity', '').lower()
        if 'retired' in employment or 'retired' in primary or 'unemployed' in employment:
            edited = [seg for seg in edited if seg['activity'] != 'work']
            # Redistribute time
            if edited:
                edited = self._redistribute_times(edited)
        
        return edited
    
    def _redistribute_times(self, schedule: List[Dict]) -> List[Dict]:
        """Redistribute times evenly after removing activities"""
        if not schedule:
            return schedule
        
        # Keep first start and last end
        first_start = schedule[0]['start_time']
        last_end = schedule[-1]['end_time']
        
        start_min = time_to_minutes(first_start)
        end_min = time_to_minutes(last_end)
        total_min = end_min - start_min
        
        num_activities = len(schedule)
        minutes_per = total_min // num_activities
        
        current = start_min
        for i, seg in enumerate(schedule):
            seg['start_time'] = minutes_to_time(current)
            if i == num_activities - 1:
                seg['end_time'] = last_end
            else:
                current += minutes_per
                seg['end_time'] = minutes_to_time(current)
        
        return schedule


# ==================== Main Pipeline ====================
def process_individual_three_agent_pipeline(
    client: OpenAI,
    user_id: str,
    person_info: Dict[str, Any]
) -> Dict:
    """
    Three-Agent Pipeline for single individual:
    1. Planner: Generate activity skeleton
    2. Trip Realizer: Add temporal details
    3. Critic & Editor: Validate and refine
    """
    
    print(f"\n{'='*70}")
    print(f" Processing: {user_id}")
    print(f" Employment: {person_info.get('employment_status')}")
    print(f" Primary Activity: {person_info.get('primary_activity')}")
    print(f"{'='*70}")
    
    # ===== Agent A: Planner =====
    print(f"\n🔹 AGENT A: PLANNER")
    planner = PlannerAgent(client, person_info)
    success_plan = planner.plan_activities()
    
    if not success_plan or not planner.activity_skeleton:
        print(f"  ❌ Planning failed for {user_id}")
        return {
            "user_id": user_id,
            "person_info": person_info,
            "success": False,
            "reason": "Planning failed"
        }
    
    # ===== Agent B: Trip Realizer =====
    print(f"\n🔹 AGENT B: TRIP REALIZER")
    realizer = TripRealizerAgent(client, person_info)
    success_realize = realizer.realize_schedule(planner.activity_skeleton)
    
    if not success_realize or not realizer.schedule:
        print(f"  ❌ Trip realization failed for {user_id}")
        return {
            "user_id": user_id,
            "person_info": person_info,
            "activity_skeleton": planner.activity_skeleton,
            "success": False,
            "reason": "Trip realization failed"
        }
    
    # ===== Agent C: Critic & Editor =====
    print(f"\n🔹 AGENT C: CRITIC & EDITOR")
    critic = CriticEditorAgent(client, person_info)
    success_edit = critic.check_and_edit(realizer.schedule)
    
    final_schedule = critic.final_schedule if critic.final_schedule else realizer.schedule
    
    print(f"\n✅ Pipeline complete for {user_id}")
    print(f"   Final schedule: {len(final_schedule)} segments")
    print(f"   Constraints satisfied: {success_edit}")
    
    return {
        "user_id": user_id,
        "person_info": person_info,
        "activity_skeleton": planner.activity_skeleton,
        "initial_schedule": realizer.schedule,
        "final_schedule": final_schedule,
        "constraint_violations": critic.constraint_violations,
        "success": True,
        "constraints_satisfied": success_edit,
        "generation_time": datetime.now().isoformat()
    }


def save_results(results: List[Dict], timestamp: str):
    """Save both full results and individual trajectories"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TRAJECTORIES_DIR, exist_ok=True)
    
    # Save full results
    full_output = os.path.join(OUTPUT_DIR, f"three_agent_pipeline_{timestamp}.json")
    with open(full_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved full results → {full_output}")
    
    # Extract and save individual trajectories
    trajectories = []
    for result in results:
        if result.get('success') and result.get('final_schedule'):
            trajectories.append({
                "user_id": result['user_id'],
                "schedule": result['final_schedule']
            })
    
    if trajectories:
        traj_output = os.path.join(OUTPUT_TRAJECTORIES_DIR, f"trajectories_{timestamp}.json")
        with open(traj_output, 'w', encoding='utf-8') as f:
            json.dump(trajectories, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(trajectories)} trajectories → {traj_output}")
    
    return len(trajectories)


def main():
    print("="*70)
    print(" THREE-AGENT PIPELINE FOR INDIVIDUAL TRAJECTORY GENERATION")
    print(" Pipeline: Planner → Trip Realizer → Critic & Editor")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TRAJECTORIES_DIR, exist_ok=True)
    
    # Load person data
    persons_list = load_json(PERSON_FILE)
    print(f"\n✓ Loaded {len(persons_list)} person records")
    
    # Sample individuals (not households)
    sample_n = 5
    random.seed(42)
    
    if len(persons_list) > sample_n:
        selected_persons = random.sample(persons_list, sample_n)
    else:
        selected_persons = persons_list
    
    print(f"\n🎯 Selected {len(selected_persons)} individuals for processing")
    
    # Initialize client
    client = create_openai_client()
    
    # Process each individual
    results = []
    for idx, person_data in enumerate(selected_persons, 1):
        user_id = person_data.get('user_id')
        if not user_id:
            continue
        
        print(f"\n{'─'*70}")
        print(f"Processing {idx}/{len(selected_persons)}: {user_id}")
        
        try:
            person_info = extract_person_info(person_data)
            result = process_individual_three_agent_pipeline(client, user_id, person_info)
            results.append(result)
            
            # Small delay between requests
            if idx < len(selected_persons):
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Error processing {user_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save results
    if results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        num_trajectories = save_results(results, timestamp)
        
        # Summary
        successful = sum(1 for r in results if r.get('success'))
        constraints_ok = sum(1 for r in results if r.get('constraints_satisfied'))
        
        print(f"\n{'='*70}")
        print(f" PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"  Total processed: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Constraints satisfied: {constraints_ok}")
        print(f"  Trajectories saved: {num_trajectories}")
        print(f"{'='*70}")
    else:
        print("\n❌ No results to save.")


if __name__ == "__main__":
    main()
