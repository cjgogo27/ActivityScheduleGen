
import json
import os
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import openai
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 🎯 Model Selection Configuration
USE_LOCAL_LLAMA = False  # True: 使用本地Llama模型, False: 使用OpenAI API
LOCAL_LLAMA_MODEL_PATH = "/data/mayue/cjy/Other_method/FinalTraj/Trajectory_Generation_multi_agent/llama3_1_trajectory_lora_rank32_20251209_184649/final"
LOCAL_LLAMA_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = "https://api.openai.com/v1"
MODEL = "gpt-4o"
TIMEOUT = 120

COPB_ACTIVITIES = [
    "go to work",      # work
    "go home",         # home  
    "eat",             # dine_out
    "do shopping",     # shopping
    "do sports",       # exercise
    "excursion",       # socialize
    "leisure or entertainment",  # socialize
    "go to sleep",     # home (睡觉在家)
    "medical treatment",  # medical
    "handle the trivialities of life",  # service
]

ACTIVITY_MAPPING = {
    "go to work": "work",
    "go home": "home",
    "eat": "dine_out",
    "do shopping": "shopping",
    "do sports": "exercise",
    "excursion": "socialize",
    "leisure or entertainment": "socialize",
    "go to sleep": "home",
    "medical treatment": "medical",
    "handle the trivialities of life": "service",
}

def create_openai_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)


def create_local_llama_client():
    """Create local Llama model client with LoRA adapter"""
    print("\n🚀 Loading local Llama model with LoRA adapter...")
    print(f"  Base model: {LOCAL_LLAMA_BASE_MODEL}")
    print(f"  LoRA adapter: {LOCAL_LLAMA_MODEL_PATH}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_LLAMA_BASE_MODEL,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLAMA_BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            LOCAL_LLAMA_MODEL_PATH,
            torch_dtype=torch.bfloat16
        )
        model.eval()
        
        print("✓ Model loaded successfully\n")
        return {"model": model, "tokenizer": tokenizer, "type": "local_llama"}
        
    except Exception as e:
        print(f"✗ Failed to load local model: {e}")
        raise


def create_client():
    """Create client based on configuration"""
    if USE_LOCAL_LLAMA:
        return create_local_llama_client()
    else:
        return create_openai_client()


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_person_description(person_info: Dict) -> str:
    user_id = person_info.get('user_id', 'Unknown')
    gender = person_info.get('gender', 'Unknown')
    age_range = person_info.get('age_range', 'Unknown')
    race = person_info.get('race', 'Unknown')
    hispanic = person_info.get('hispanic', 'Unknown')
    relationship = person_info.get('relationship', 'Unknown')
    
    education = person_info.get('education', 'Unknown')
    employment = person_info.get('employment_status', 'Unknown')
    occupation = person_info.get('occupation', 'Unknown')
    work_schedule = person_info.get('work_schedule', 'Unknown')
    primary_activity = person_info.get('primary_activity', 'Unknown')

    work_from_home = person_info.get('work_from_home', 'Unknown')
    distance_to_work = person_info.get('distance_to_work_miles', 0)
    work_state = person_info.get('work_state', 'Unknown')

    driver = person_info.get('driver_on_travel_day', 'Unknown')
    traveled_abroad = person_info.get('traveled_abroad', 'Unknown')

    description = f"""You are a person and your complete profile is as follows:

User Information (California NHTS 2017 Data - Complete Profile):
- User ID: {user_id}
- Age Range: {age_range}
- Hispanic: {hispanic}
- Relationship: {relationship}
- Gender: {gender}
- Race: {race}
- Education: {education}
- Employment Status: {employment}
- Traveled Abroad: {traveled_abroad}
- Distance to Work: {distance_to_work:.2f} miles
- Work State: {work_state}
- Driver on Travel Day: {driver}
- Work From Home: {work_from_home}
- Work Schedule: {work_schedule}
- Occupation: {occupation}
- Primary Activity: {primary_activity}"""
    
    return description.strip()


def generate_anchors(client, person_description: str, day: str = "Monday") -> str:

    system_prompt = """Most people in life have some fixed routines or habits that are generally non-negotiable and must be adhered to. For example, civil servants usually have fixed working hours (such as from 9 am to 5 pm), and engineers at technology companies usually go to work close to noon, and may not get off work until 10 p.m.. Some people insist on going to bed before 23:00, while some people are used to staying up late and getting up very late too.
Now I give you a description of a person, and I hope you can generate 3 habits or tendencies that this person may have.

Hint: I hope you can take into consideration the habits that people of this kind might realistically have in their daily life. For example, a significant number of people may not exercise every day; for most people, their lives may have little aside from work and rest with few fixed activities; some jobs may require frequent overtime until 22:00 or even later, while others may only require half-day work. I don't need you to tell me how this person should plan their life; I want you, based on this person's attributes, to tell me what kind of life and habits they might have in real life."""

    user_prompt = f"""The person's basic information is as follows: 
{person_description}

Please generate 3 anchor points for him. No explanation is required. Try to keep it concise, emphasizing time and key terms (Example1: You are accustomed to waking up before 8 AM. Example2: Your working hours are from 9 AM to 7 PM.).
Please answer in the second person using an affirmative tone and organize your answers in 1.xxx 2.xxx 3.xxx format."""

    try:
        # Call LLM based on client type
        if isinstance(client, dict) and client.get("type") == "local_llama":
            # Local Llama model
            model = client["model"]
            tokenizer = client["tokenizer"]
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.85,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            return result
        else:
            # OpenAI API
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,  # CoPB使用temperature=1
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating anchors: {e}")
        return "1. You wake up at 7 AM.\n2. You work from 9 AM to 5 PM.\n3. You go to bed at 11 PM."


def generate_day_intentions(client, person_description: str, anchors: str, day: str, N: int = 4) -> List[Dict]:
    day_note = ""
    if day in ["Saturday", "Sunday"]:
        day_note = ". It is important to note that people generally do not work on weekends and prefer getting up later, entertainment, sports and leisure activities. There will also be more freedom in the allocation of time."

    system_prompt = f"""{person_description}

You have some habits or tendencies as follows:
{anchors}

Now I want you to generate your own schedule for today.(today is {day}{day_note})
The specific requirements of the task are as follows:
1. You need to consider how your character attributes, routines or habits relate to your behavior decisions.
2. I want to limit your total number of events in a day to {N}. I hope you can make every decision based on this limit.
3. I want you to answer the basis and reason behind each intention decision.

Note that: 
1. All times are in 24-hour format.
2. The generated schedule must start at 0:00 and end at 24:00. Don't let your schedule spill over into the next day.
3. Must remember that events can only be choosed from [go to work, go home, eat, do shopping, do sports, excursion, leisure or entertainment, go to sleep, medical treatment, handle the trivialities of life, banking and financial services, cultural institutions and events].
4. I'll ask you step by step what to do, and you just have to decide what to do next each time."""

    user_prompt = """Please generate the complete schedule for the day in JSON format.

Output format:
```json
[
  {
    "intention": "go to sleep",
    "start_time": "00:00",
    "end_time": "07:00",
    "reasoning": "Need rest before work"
  },
  {
    "intention": "go to work",
    "start_time": "08:00",
    "end_time": "17:00",
    "reasoning": "Regular full-time work hours"
  },
  {
    "intention": "go home",
    "start_time": "17:00",
    "end_time": "24:00",
    "reasoning": "Return home and rest"
  }
]
```

Generate the full day schedule:"""

    for attempt in range(3):
        try:
            # Call LLM based on client type
            if isinstance(client, dict) and client.get("type") == "local_llama":
                # Local Llama model
                model = client["model"]
                tokenizer = client["tokenizer"]
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1500,
                        temperature=0.85,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                result_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            else:
                # OpenAI API
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=1.0, 
                    max_tokens=1500
                )
                result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if "```json" in result_text:
                json_start = result_text.find("```json") + 7
                json_end = result_text.find("```", json_start)
                result_text = result_text[json_start:json_end].strip()
            elif "```" in result_text:
                json_start = result_text.find("```") + 3
                json_end = result_text.find("```", json_start)
                result_text = result_text[json_start:json_end].strip()
            
            intentions = json.loads(result_text)

            if isinstance(intentions, list) and len(intentions) > 0:
                return intentions
            else:
                raise ValueError("Invalid format")
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2)
            else:

                return [
                    {"intention": "go to sleep", "start_time": "00:00", "end_time": "07:00", "reasoning": "Sleep"},
                    {"intention": "go to work", "start_time": "08:00", "end_time": "17:00", "reasoning": "Work"},
                    {"intention": "go home", "start_time": "17:00", "end_time": "24:00", "reasoning": "Home"}
                ]
    
    return []


def ensure_continuous_time(intentions: List[Dict]) -> List[Dict]:
    if not intentions:
        return [{"intention": "go to sleep", "start_time": "00:00", "end_time": "24:00", "reasoning": "Default"}]
    
    def time_to_minutes(time_str: str) -> int:
        h, m = map(int, time_str.split(':'))
        return h * 60 + m
    
    def minutes_to_time(minutes: int) -> str:
        h = minutes // 60
        m = minutes % 60
        return f"{h:02d}:{m:02d}"
    
    sorted_intentions = sorted(intentions, key=lambda x: time_to_minutes(x.get("start_time", "00:00")))
    
    continuous_intentions = []
    
    first_start = time_to_minutes(sorted_intentions[0].get("start_time", "00:00"))
    if first_start > 0:
        continuous_intentions.append({
            "intention": "go to sleep",
            "start_time": "00:00",
            "end_time": minutes_to_time(first_start),
            "reasoning": "Sleep before daily activities"
        })
    
    for i, intent in enumerate(sorted_intentions):
        current_start = time_to_minutes(intent.get("start_time", "00:00"))
        current_end = time_to_minutes(intent.get("end_time", "24:00"))
        
        if continuous_intentions:
            last_end = time_to_minutes(continuous_intentions[-1]["end_time"])
            if current_start > last_end:
                continuous_intentions[-1]["end_time"] = minutes_to_time(current_start)
        
        continuous_intentions.append(intent)
    
    last_end = time_to_minutes(continuous_intentions[-1]["end_time"])
    if last_end < 24 * 60:
        last_intention = continuous_intentions[-1]["intention"]
        if last_intention in ["go to sleep", "go home"]:
            continuous_intentions[-1]["end_time"] = "24:00"
        else:
            continuous_intentions.append({
                "intention": "go to sleep",
                "start_time": minutes_to_time(last_end),
                "end_time": "24:00",
                "reasoning": "Sleep at night"
            })
    
    return continuous_intentions


def intentions_to_trajectory(intentions: List[Dict]) -> List[Dict]:
    """
    
    CoPB格式:
    [{"intention": "go to work", "start_time": "08:00", "end_time": "17:00", "reasoning": "..."}]
    
    California格式:
    [{"activity": "work", "start_time": "08:00", "end_time": "17:00"}]
    """
    continuous_intentions = ensure_continuous_time(intentions)
    
    trajectory = []
    
    for intent in continuous_intentions:
        intention_name = intent.get("intention", "go home")
        
        activity = ACTIVITY_MAPPING.get(intention_name, "home")
        
        trajectory.append({
            "activity": activity,
            "start_time": intent.get("start_time", "00:00"),
            "end_time": intent.get("end_time", "24:00")
        })
    
    return trajectory


def generate_trajectory_for_person(client: OpenAI, person_info: Dict, day: str = "Monday") -> Dict:
    user_id = person_info['user_id']
    print(f"\n生成轨迹: {user_id}")
    
    # Step 1: 生成个人描述
    person_description = get_person_description(person_info)
    
    # Step 2: 生成锚点
    anchors = generate_anchors(client, person_description, day)
    
    # Step 3: 生成意图序列
    N = random.choice([3, 4, 5]) 
    intentions = generate_day_intentions(client, person_description, anchors, day, N)
    
    # Step 4: 转换为轨迹
    trajectory = intentions_to_trajectory(intentions)
    
    return {
        "user_id": user_id,
        "schedule": trajectory,
        "metadata": {
            "person_description": person_description,
            "anchors": anchors,
            "intentions": intentions,
            "day": day
        }
    }


def main():
    PERSON_FILE = "E:/FrankYcj/FinalTraj/California/processed_data/california_person_static.json"
    OUTPUT_DIR = "E:/FrankYcj/FinalTraj/Trajectory_Generation/output_copb"
    DAY = "Monday"

    GENERATION_MODE = "user_id_file"  # "num_users" 或 "user_id_file"
    
    NUM_USERS = 2
    
    USER_ID_FILE = "E:\\FrankYcj\\FinalTraj\\Trajectory_Generation_multi_agent\\user_id\\user_id_20251117_122412.json"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n加载数据: {PERSON_FILE}")
    persons = load_json(PERSON_FILE)
    print(f"  总人数: {len(persons)}")

    if GENERATION_MODE == "num_users":
        sampled_persons = random.sample(persons, min(NUM_USERS, len(persons)))
        
    elif GENERATION_MODE == "user_id_file":

        target_user_ids = load_json(USER_ID_FILE)
        if isinstance(target_user_ids, list):
            target_ids = target_user_ids
        elif isinstance(target_user_ids, dict) and "user_ids" in target_user_ids:
            target_ids = target_user_ids["user_ids"]
        else:
            print(f"  错误: 无法识别的user_id文件格式")
            return
        
        
        # 过滤出目标用户
        sampled_persons = [p for p in persons if p['user_id'] in target_ids]
        print(f"  匹配到的用户数: {len(sampled_persons)}")
        
        if len(sampled_persons) < len(target_ids):
            matched_ids = {p['user_id'] for p in sampled_persons}
            missing_ids = set(target_ids) - matched_ids
            print(f"  警告: {len(missing_ids)} 个user_id未找到")
            if len(missing_ids) <= 5:
                print(f"  未找到的ID: {list(missing_ids)}")
    else:
        print(f"  错误: 未知的生成模式 '{GENERATION_MODE}'")
        return
    
    client = create_client()
    
    print("\n" + "="*70)
    model_name = "Local Llama (LoRA fine-tuned)" if USE_LOCAL_LLAMA else "OpenAI GPT-4o"
    print(f"使用模型: {model_name}")
    print("="*70 + "\n")

    all_trajectories = []
    
    for idx, person in enumerate(sampled_persons, 1):
        print(f"\n[{idx}/{len(sampled_persons)}] 处理用户: {person['user_id']}")
        
        try:
            trajectory = generate_trajectory_for_person(client, person, DAY)
            all_trajectories.append(trajectory)
            
            time.sleep(1)
            
        except Exception as e:
            print(f"  错误: {e}")
            continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    simple_trajectories = [
        {
            "user_id": traj["user_id"],
            "schedule": traj["schedule"]
        }
        for traj in all_trajectories
    ]
    simple_output_file = os.path.join(OUTPUT_DIR, f"copb_{timestamp}.json")
    save_json(simple_trajectories, simple_output_file)
    print(f"\n✓ 轨迹文件保存到: {simple_output_file}")
    
    full_output_file = os.path.join(OUTPUT_DIR, f"copb_full_{timestamp}.json")
    save_json(all_trajectories, full_output_file)
    print(f"✓ 完整结果保存到: {full_output_file}")

    print("\n" + "="*70)
    print("生成统计:")
    print("="*70)
    print(f"  成功生成轨迹数: {len(all_trajectories)}")
    
    if all_trajectories:
        avg_activities = sum(len(t["schedule"]) for t in all_trajectories) / len(all_trajectories)
        print(f"  平均活动数: {avg_activities:.2f}")
        
        activity_counts = {}
        for traj in all_trajectories:
            for act in traj["schedule"]:
                activity = act["activity"]
                activity_counts[activity] = activity_counts.get(activity, 0) + 1
        
        print(f"\n  活动类型分布:")
        for activity, count in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {activity}: {count}")
    
    print("\n" + "="*70)
    print(" 生成完成!")
    print("="*70)


if __name__ == "__main__":
    main()
