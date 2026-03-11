"""
CoPB (Controllable Preference-Based) 基线推理脚本
使用 Theory of Planned Behavior (TPB) 框架，通过 API 生成日程
改编自原 CoPB 方法，适配本项目的10类活动类型

用法:
  python run_baseline_copb.py --data eval_data/california_50.jsonl
  python run_baseline_copb.py --data eval_data/arizona_50.jsonl
"""

import sys, os, json, re, argparse, time
from datetime import datetime, timedelta
from tqdm import tqdm
from openai import OpenAI

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
EVAL_DIR    = "/data/alice/cjtest/FinalTraj_arr/evaluation"
RESULTS_DIR = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/results"
sys.path.insert(0, EVAL_DIR)

# ─── API 配置（与 run_inference.py 相同）─────────────────────────────────────
API_KEY     = "sk-qyl51vYITpOoElayZ5gmNuIlsU2p3iNQnawX9G0RyMzOICym"
API_BASE_URL = "https://api.nuwaflux.com/v1"
API_MODEL   = "gpt-4o"

# ─── 活动类型（对应10类）────────────────────────────────────────────────────
ACTIVITIES = [
    "home", "work", "education", "shopping", "service",
    "medical", "dine_out", "socialize", "exercise", "dropoff_pickup"
]

ACTIVITY_DESCRIPTIONS = {
    "home":            "stay at home (rest, sleep, household chores)",
    "work":            "go to work or work from home",
    "education":       "attend school, classes, or university",
    "shopping":        "go shopping (retail, grocery)",
    "service":         "personal services (errands, government, banking)",
    "medical":         "medical appointment or pharmacy",
    "dine_out":        "eat at a restaurant or cafe",
    "socialize":       "visit friends or family, social gatherings",
    "exercise":        "gym, sports, outdoor recreation",
    "dropoff_pickup":  "drop off or pick up someone (e.g., children)",
}

ACTIVITY_LIST_STR = ", ".join(ACTIVITIES)


def add_minutes(time_str: str, minutes: int):
    """时间字符串加分钟，越过24:00时返回(end, True)"""
    if time_str == "24:00":
        return "24:00", True
    h, m = map(int, time_str.split(":"))
    total = h * 60 + m + minutes
    if total >= 1440:
        return "24:00", True
    return f"{total // 60:02d}:{total % 60:02d}", False


def build_global_prompt(profile: dict) -> list:
    """构建包含用户档案和生活方式的系统提示（TPB 框架）"""
    p = profile
    system_content = f"""You are simulating the daily behavior of a real person based on their profile. 
Use the Theory of Planned Behavior (TPB) to reason about their activities.

PERSON PROFILE:
- Age range: {p.get('age_range', 'Unknown')}
- Gender: {p.get('gender', 'Unknown')}
- Education: {p.get('education', 'Unknown')}
- Employment status: {p.get('employment_status', 'Unknown')}
- Work schedule: {p.get('work_schedule', 'Unknown')}
- Occupation: {p.get('occupation', 'Unknown')}
- Primary activity: {p.get('primary_activity', 'Unknown')}
- Work from home: {p.get('work_from_home', 'Unknown')}
- Driver on travel day: {p.get('driver_on_travel_day', 'Unknown')}

TPB FRAMEWORK:
- Attitude: Personal beliefs about each activity (beneficial vs. costly)
- Subjective Norm: Social expectations for this type of person
- Perceived Behavioral Control: Ability and opportunity to perform the activity

AVAILABLE ACTIVITIES: {ACTIVITY_LIST_STR}
"""
    return [{"role": "system", "content": system_content}]


def ask_confidence_ranking(client, conversation: list, current_time: str, history: list) -> list:
    """TPB 步骤1：对各活动类型进行完成置信度排名"""
    hist_str = ""
    if history:
        hist_str = "Already done today: " + ", ".join(
            f"{act} ({st}-{et})" for act, st, et in history
        ) + ".\n"

    question = (
        f"Current time: {current_time}.\n"
        f"{hist_str}"
        f"Based on TPB (Attitude + Subjective Norm + Perceived Behavioral Control), "
        f"rank your confidence of doing each activity NEXT from most to least likely.\n"
        f"Rank ALL activities: {ACTIVITY_LIST_STR}\n"
        f"Answer format: [activity1, activity2, ..., activity10]"
    )
    msgs = conversation + [{"role": "user", "content": question}]

    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=API_MODEL, messages=msgs, temperature=1.0, timeout=60,
            )
            raw = resp.choices[0].message.content.strip()
            # 解析列表
            m = re.search(r'\[([^\[\]]+)\]', raw)
            if m:
                items = [x.strip().strip('"\'') for x in m.group(1).split(",")]
                valid = [x for x in items if x.lower() in ACTIVITIES]
                if valid:
                    # 补全缺失项
                    missing = [a for a in ACTIVITIES if a not in valid]
                    return valid + missing
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return ACTIVITIES  # 默认顺序


def ask_duration(client, conversation: list, activity: str, history: list, current_time: str) -> int:
    """TPB 步骤2：询问该活动的持续时长（分钟）"""
    hist_str = ""
    if history:
        hist_str = "So far today: " + ", ".join(
            f"{act}({st}-{et})" for act, st, et in history
        ) + ". "

    question = (
        f"{hist_str}"
        f"You will now do '{activity}' starting at {current_time}. "
        f"Given your profile and remaining day, how many minutes will this take? "
        f"Answer with a single integer between 15 and 480."
    )
    msgs = conversation + [{"role": "user", "content": question}]

    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=API_MODEL, messages=msgs, temperature=0.7, timeout=60,
            )
            raw = resp.choices[0].message.content.strip()
            m = re.search(r'\b(\d{2,3})\b', raw)
            if m:
                mins = int(m.group(1))
                if 15 <= mins <= 480:
                    return mins
            # 尝试提取任何数字
            m_any = re.search(r'\b(\d+)\b', raw)
            if m_any:
                mins = int(m_any.group(1))
                return max(15, min(480, mins))
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return 60  # 默认60分钟


def generate_copb_schedule(client, sample: dict) -> list:
    """为一个用户生成完整日程（CoPB方式）"""
    profile     = sample["user_profile"]
    conversation = build_global_prompt(profile)

    history     = []    # [(activity, start_time, end_time)]
    current_time = "00:00"
    done        = False

    # 最多生成15个活动片段
    for i in range(15):
        if done:
            break

        # 获取置信度排名
        ranking = ask_confidence_ranking(client, conversation, current_time, history)

        # 选择排名最高的活动（跳过连续重复的 home）
        chosen = None
        for act in ranking:
            if act == "home" and history and history[-1][0] == "home":
                continue  # 避免连续在家
            chosen = act
            break
        if not chosen:
            chosen = "home"

        # 获取活动时长
        duration = ask_duration(client, conversation, chosen, history, current_time)

        # 计算结束时间
        end_time, crossed = add_minutes(current_time, duration)
        history.append((chosen, current_time, end_time))

        # 更新对话记录
        conversation = conversation + [
            {"role": "user",      "content": f"[{current_time}] Next activity?"},
            {"role": "assistant", "content": f"{chosen} for {duration} minutes ({current_time} - {end_time})"},
        ]

        if crossed or end_time == "24:00":
            done = True
        else:
            current_time = end_time

    # 补全剩余时间为 home
    if not done and current_time < "24:00":
        history.append(("home", current_time, "24:00"))

    # 转换为 schedule 格式
    schedule = [
        {"activity": act, "start_time": st, "end_time": et}
        for act, st, et in history
    ]

    # 确保24h覆盖
    if not schedule:
        return [{"activity": "home", "start_time": "00:00", "end_time": "24:00"}]
    if schedule[0]["start_time"] != "00:00":
        schedule.insert(0, {"activity": "home", "start_time": "00:00", "end_time": schedule[0]["start_time"]})
    if schedule[-1]["end_time"] != "24:00":
        schedule[-1]["end_time"] = "24:00"

    return schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    required=True, help="path to {state}_50.jsonl")
    parser.add_argument("--delay",   type=float, default=1.0, help="seconds between API calls")
    args = parser.parse_args()

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    # 读取数据
    samples = []
    with open(args.data) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} samples from {args.data}")

    ground_truth = [{"user_id": s["user_id"], "schedule": s["ground_truth_schedule"]} for s in samples]

    all_results = []
    for sample in tqdm(samples, desc="CoPB inference"):
        try:
            schedule = generate_copb_schedule(client, sample)
            all_results.append({"user_id": sample["user_id"], "schedule": schedule})
        except Exception as e:
            print(f"Failed for {sample['user_id']}: {e}")
            all_results.append({"user_id": sample["user_id"],
                                 "schedule": [{"activity": "home", "start_time": "00:00", "end_time": "24:00"}]})
        if args.delay > 0:
            time.sleep(args.delay)

    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    n        = len(all_results)
    data_tag = os.path.splitext(os.path.basename(args.data))[0]

    gen_path = os.path.join(RESULTS_DIR, f"generated_copb_{n}samples_{ts}.json")
    gt_path  = os.path.join(RESULTS_DIR, f"ground_truth_{n}samples_{ts}.json")

    with open(gen_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Generated {n} trajectories → {gen_path}")

    # 评估
    try:
        import evaluate_generated_trajectories
        eval_dir = os.path.join(RESULTS_DIR, f"eval_copb_{data_tag}_{ts}")
        metrics = evaluate_generated_trajectories.evaluate_trajectories(
            gen_path, gt_path, eval_dir, save_csv=True,
        )
        if metrics:
            print("\n=== Evaluation Results ===")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
    except Exception as e:
        print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
