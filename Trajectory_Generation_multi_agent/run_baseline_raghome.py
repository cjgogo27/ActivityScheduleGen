"""
RAGHome 基线推理脚本
使用检索增强生成（RAG）方法，利用同户家庭成员的已生成日程作为上下文
改编自 generate_trajectories_multiagent_rag.py

用法:
  python run_baseline_raghome.py --data eval_data/california_50.jsonl
  python run_baseline_raghome.py --data eval_data/arizona_50.jsonl
"""

import sys, os, json, re, argparse, time
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
EVAL_DIR    = "/data/alice/cjtest/FinalTraj_arr/evaluation"
RESULTS_DIR = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/results"
sys.path.insert(0, EVAL_DIR)

# ─── API 配置（与 run_inference.py 相同）─────────────────────────────────────
API_KEY      = "sk-qyl51vYITpOoElayZ5gmNuIlsU2p3iNQnawX9G0RyMzOICym"
API_BASE_URL = "https://api.nuwaflux.com/v1"
API_MODEL    = "gpt-4o"

ALLOWED_ACTIVITIES = {
    "home", "work", "education", "shopping", "service",
    "medical", "dine_out", "socialize", "exercise", "dropoff_pickup"
}

PLANNER_SYSTEM = """You are a daily schedule planner. Given a person's profile, generate a complete and realistic 24-hour schedule with exact start/end times in ONE step.

RULES:
- Use ONLY these activity types: home, work, education, shopping, service, medical, dine_out, socialize, exercise, dropoff_pickup
- Schedule MUST cover full 24 hours: start at 00:00, end at 24:00, NO gaps or overlaps
- Must start and end with "home"
- Consecutive activities must connect (end_time of one = start_time of next)
- Durations must be realistic based on the person's profile

OUTPUT FORMAT (strictly follow):
```json
{
  "reasoning": "Brief explanation of the schedule choices",
  "schedule": [
    {"activity": "home",  "start_time": "00:00", "end_time": "07:30"},
    {"activity": "work",  "start_time": "07:30", "end_time": "17:00"},
    {"activity": "home",  "start_time": "17:00", "end_time": "24:00"}
  ]
}
```"""


class HouseholdRAG:
    """简单 RAG 存储：保存已生成的家庭成员日程，作为后续成员生成的上下文"""
    def __init__(self):
        self._store = {}   # {household_id: {user_id: schedule_text}}

    def store(self, household_id: str, user_id: str, schedule: list):
        if household_id not in self._store:
            self._store[household_id] = {}
        lines = [f"  {s['start_time']}-{s['end_time']}: {s['activity']}" for s in schedule]
        self._store[household_id][user_id] = "\n".join(lines)

    def retrieve(self, household_id: str, exclude_uid: str = None) -> str:
        if household_id not in self._store:
            return ""
        parts = []
        for uid, sched_text in self._store[household_id].items():
            if uid == exclude_uid:
                continue
            parts.append(f"  Household member {uid}:\n{sched_text}")
        return "\n".join(parts)


def clean_json(text: str) -> str:
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    return text.strip()


def normalize_activity(act: str) -> str:
    alias = {
        "sleep": "home", "rest": "home", "sleeping": "home",
        "commute": "work", "dining": "dine_out", "restaurant": "dine_out",
        "grocery": "shopping", "errands": "service", "gym": "exercise",
        "school": "education", "doctor": "medical", "hospital": "medical",
        "childcare": "dropoff_pickup",
    }
    a = act.strip().lower()
    return alias.get(a, a) if alias.get(a, a) in ALLOWED_ACTIVITIES else "home"


def parse_schedule(text: str):
    text = clean_json(text)
    m = re.search(r'(\{.*\})', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            segs = obj.get("schedule", [])
            if segs:
                return [{"activity": normalize_activity(s["activity"]),
                          "start_time": s["start_time"],
                          "end_time":   s["end_time"]}
                         for s in segs
                         if all(k in s for k in ("activity", "start_time", "end_time"))]
        except Exception:
            pass
    return None


def generate_with_rag(client, sample: dict, rag: HouseholdRAG, household_id: str) -> list:
    """生成单个用户的日程，可选注入家庭成员上下文"""
    p = sample["user_profile"]
    user_content = (
        "Generate a complete 24-hour schedule for this person:\n\n"
        f"- Age range: {p.get('age_range', 'Unknown')}\n"
        f"- Gender: {p.get('gender', 'Unknown')}\n"
        f"- Education: {p.get('education', 'Unknown')}\n"
        f"- Employment status: {p.get('employment_status', 'Unknown')}\n"
        f"- Work schedule: {p.get('work_schedule', 'Unknown')}\n"
        f"- Occupation: {p.get('occupation', 'Unknown')}\n"
        f"- Primary activity: {p.get('primary_activity', 'Unknown')}\n"
        f"- Work from home: {p.get('work_from_home', 'Unknown')}\n"
        f"- Driver on travel day: {p.get('driver_on_travel_day', 'Unknown')}\n"
    )

    # 注入 RAG 上下文（如果有家庭成员已生成的日程）
    context = rag.retrieve(household_id, exclude_uid=sample["user_id"])
    if context:
        user_content += (
            f"\nOther household members' schedules for reference (consider coordination):\n"
            f"{context}\n"
        )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=API_MODEL, messages=messages, temperature=0.7,
                max_tokens=1500, timeout=60,
            )
            raw = resp.choices[0].message.content
            schedule = parse_schedule(raw)
            if schedule:
                return schedule
            time.sleep(2)
        except Exception as e:
            print(f"  API error: {e}")
            time.sleep(3)

    return [{"activity": "home", "start_time": "00:00", "end_time": "24:00"}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True, help="path to {state}_50.jsonl")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    samples = []
    with open(args.data) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} samples from {args.data}")

    # 按家庭分组（保持原序，同家庭的后续成员能看到前面成员的日程）
    household_order = []
    hh_groups = defaultdict(list)
    for s in samples:
        uid = s["user_id"]
        hh  = uid.rsplit("_", 1)[0]
        hh_groups[hh].append(s)
        if hh not in [h for h, _ in household_order]:
            household_order.append((hh, []))

    # 重建有序列表：按首次出现的家庭顺序
    seen_hh = []
    ordered_samples = []
    for s in samples:
        hh = s["user_id"].rsplit("_", 1)[0]
        if hh not in seen_hh:
            seen_hh.append(hh)
        # 追加到对应家庭
    # 按家庭顺序排列样本，每个家庭内按原始顺序
    hh_to_samples = defaultdict(list)
    for s in samples:
        hh = s["user_id"].rsplit("_", 1)[0]
        hh_to_samples[hh].append(s)

    ordered_samples = []
    for hh in seen_hh:
        ordered_samples.extend(hh_to_samples[hh])

    ground_truth = {s["user_id"]: s["ground_truth_schedule"] for s in samples}
    rag = HouseholdRAG()
    results_dict = {}

    for sample in tqdm(ordered_samples, desc="RAGHome inference"):
        uid = sample["user_id"]
        hh  = uid.rsplit("_", 1)[0]

        schedule = generate_with_rag(client, sample, rag, hh)
        results_dict[uid] = schedule
        rag.store(hh, uid, schedule)

        if args.delay > 0:
            time.sleep(args.delay)

    # 还原原始顺序
    all_results  = [{"user_id": s["user_id"], "schedule": results_dict[s["user_id"]]} for s in samples]
    ground_truth_list = [{"user_id": s["user_id"], "schedule": s["ground_truth_schedule"]} for s in samples]

    # 保存
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    n        = len(all_results)
    data_tag = os.path.splitext(os.path.basename(args.data))[0]

    gen_path = os.path.join(RESULTS_DIR, f"generated_raghome_{n}samples_{ts}.json")
    gt_path  = os.path.join(RESULTS_DIR, f"ground_truth_{n}samples_{ts}.json")

    with open(gen_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(gt_path, "w") as f:
        json.dump(ground_truth_list, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Generated {n} trajectories → {gen_path}")

    # 评估
    try:
        import evaluate_generated_trajectories
        eval_dir = os.path.join(RESULTS_DIR, f"eval_raghome_{data_tag}_{ts}")
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
