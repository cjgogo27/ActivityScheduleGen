"""
推理 + 评估脚本：多阶段 Multi-Agent Pipeline
  【本地模型】两阶段
    Stage 1 - Planner Agent   : user_profile → initial_schedule
    Stage 2 - Critic & Editor : user_profile + initial_schedule → final_schedule

  【API 后端】三阶段（参考 generate_trajectories_three_agent_pipeline.py）
    Stage 1 - API Planner      (gpt-5.2): user_profile → activity skeleton
    Stage 2 - API Trip Realizer(gpt-5.2): activity skeleton → timed initial_schedule
    Stage 3 - API Editor       (gpt-5.1): user_profile + initial_schedule → final_schedule

生成完成后自动调用 evaluate_generated_trajectories.py 计算评估指标。

用法：
  # Base 模型（两阶段），取前 50 条
  python run_inference.py --model base --num_samples 50

  # SFT 模型（两阶段），全量跑
  python run_inference.py --model sft

  # SFT 模型，跳过 Planner（使用存储的 initial_schedule）
  python run_inference.py --model sft --use_stored_initial

  # API 三阶段，取前 50 条
  python run_inference.py --model api --num_samples 50

  # API，跳过 Planner+Realizer（直接 Editor）
  python run_inference.py --model api --use_stored_initial --num_samples 50

输出目录：results/
  generated_<model>_<N>samples_<timestamp>.json
  ground_truth_<N>samples_<timestamp>.json
  evaluation_<model>_<timestamp>/
"""

import os, re, sys, json, argparse, time
from datetime import datetime
import torch

# ── OpenAI API 配置（API 推理后端） ──────────────────────────────────────────
API_KEY            = "sk-qyl51vYITpOoElayZ5gmNuIlsU2p3iNQnawX9G0RyMzOICym"
API_BASE_URL       = "https://api.nuwaflux.com/v1"
API_PLANNER_MODEL  = "gpt-5.2"
API_EDITOR_MODEL   = "gpt-5.2"
API_TIMEOUT        = 60

# ── torch 2.2.1 + transformers 5.0.0 兼容性修复 ──────────────────────────────
if hasattr(torch, 'is_autocast_enabled'):
    _orig_is_autocast_enabled = torch.is_autocast_enabled
    def _patched_is_autocast_enabled(device_type=None):
        return _orig_is_autocast_enabled()
    torch.is_autocast_enabled = _patched_is_autocast_enabled

# 将评估目录加入 path，以便 import eval_example
EVAL_DIR = "/data/alice/cjtest/FinalTraj_arr/evaluation"
sys.path.insert(0, EVAL_DIR)

# ─── 路径配置 ────────────────────────────────────────────────────────────────
BASE_MODEL_PATH  = "/data/alice/cjtest/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B"
SFT_MODEL_PATH   = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage2_sft_output_epoch10/final_model"
GRPO_MODEL_PATH  = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage3_grpo_output/final_model"

# ─── LLaMA 路径配置 ──────────────────────────────────────────────────────────
LLAMA_BASE_PATH      = "/data/alice/cjtest/FinalTraj_arr/finetune/models/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct"
LLAMA_SFT_PATH       = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage2_sft_llama_v2_epoch20/final_model"
LLAMA_GRPO_SFT_PATH  = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage2_sft_llama_v2_epoch20/final_model"
LLAMA_GRPO_PATH      = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage3_grpo_llama_v2_output/final_model"

DEFAULT_DATA    = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage1_training_data_3/metadata.jsonl"
RESULTS_DIR     = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/results"
HISTORY_CSV     = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/results/all_results_history.csv"

ALLOWED_ACTIVITIES = {
    "home", "work", "education", "shopping", "service",
    "medical", "dine_out", "socialize", "exercise", "dropoff_pickup"
}

# ─── Stage 1: Planner Agent prompt ───────────────────────────────────────────
# ─── Stage 1 Planner / Stage 1+2 合并 (api_merged) 共用指令 ──────────────────
PLANNER_REALIZER_INSTRUCTION = """You are a daily schedule planner. Given a person's profile, generate a complete and realistic 24-hour schedule with exact start/end times in ONE step.

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


# ─── Stage 1 (API 后端): Planner 指令 — 输出带 reasoning 的结构化 JSON ────────
PLANNER_INSTRUCTION_API = """You are a daily schedule skeleton planner. Generate a realistic activity sequence for a person based on their profile.

RULES:
- Use ONLY these activity types: home, work, education, shopping, service, medical, dine_out, socialize, exercise, dropoff_pickup
- The sequence MUST start and end with "home"
- Keep it realistic based on employment status and work schedule
- Do NOT assign times here — only list activities in order

OUTPUT FORMAT (strictly follow this):
```json
{
  "reasoning": "Brief explanation of why these activities are chosen",
  "activities": ["home", "work", "shopping", "home"]
}
```"""


# ─── Stage 2: Critic & Editor prompt（与 SFT 训练完全一致）──────────────────
EDITOR_INSTRUCTION = """You are a Critic & Editor Agent for daily schedule generation and refinement.

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


def build_planner_messages(user_profile: dict) -> list:
    """本地模型 Planner — profile → 带时间完整日程，与 api_merged 使用相同指令"""
    return [
        {"role": "system", "content": PLANNER_REALIZER_INSTRUCTION},
        {"role": "user",   "content": (
            "Generate a daily schedule for this person:\n"
            + json.dumps(user_profile, indent=2, ensure_ascii=False)
        )},
    ]


def build_planner_api_messages(user_profile: dict) -> list:
    """Stage 1 (API 后端): Planner — 生成结构化 {reasoning, activities} 骨架，包含全部静态属性"""
    age_range    = user_profile.get("age_range", "Unknown")
    gender       = user_profile.get("gender", "Unknown")
    race         = user_profile.get("race", "Unknown")
    education    = user_profile.get("education", "Unknown")
    employment   = user_profile.get("employment_status", "Unknown")
    work_sched   = user_profile.get("work_schedule", "Unknown")
    occupation   = user_profile.get("occupation", "Unknown")
    primary_act  = user_profile.get("primary_activity", "Unknown")
    wfh          = user_profile.get("work_from_home", "Unknown")
    driver       = user_profile.get("driver_on_travel_day", "Unknown")
    dist_work    = user_profile.get("distance_to_work_miles", "Unknown")
    work_state   = user_profile.get("work_state", "Unknown")

    user_content = (
        f"Generate an activity skeleton for this person:\n\n"
        f"- Age range: {age_range}\n"
        f"- Gender: {gender}\n"
        f"- Race: {race}\n"
        f"- Education: {education}\n"
        f"- Employment status: {employment}\n"
        f"- Work schedule: {work_sched}\n"
        f"- Occupation: {occupation}\n"
        f"- Primary activity: {primary_act}\n"
        f"- Work from home: {wfh}\n"
        f"- Driver on travel day: {driver}\n"
        f"- Distance to work (miles): {dist_work}\n"
        f"- Work state: {work_state}\n"
    )
    return [
        {"role": "system", "content": PLANNER_INSTRUCTION_API},
        {"role": "user",   "content": user_content},
    ]


def build_planner_realizer_messages(user_profile: dict) -> list:
    """api_merged: Planner+Realizer 合并 — profile → 带时间完整日程（一次调用）"""
    age_range   = user_profile.get("age_range", "Unknown")
    gender      = user_profile.get("gender", "Unknown")
    race        = user_profile.get("race", "Unknown")
    education   = user_profile.get("education", "Unknown")
    employment  = user_profile.get("employment_status", "Unknown")
    work_sched  = user_profile.get("work_schedule", "Unknown")
    occupation  = user_profile.get("occupation", "Unknown")
    primary_act = user_profile.get("primary_activity", "Unknown")
    wfh         = user_profile.get("work_from_home", "Unknown")
    driver      = user_profile.get("driver_on_travel_day", "Unknown")
    dist_work   = user_profile.get("distance_to_work_miles", "Unknown")
    work_state  = user_profile.get("work_state", "Unknown")

    user_content = (
        f"Generate a complete 24-hour schedule for this person:\n\n"
        f"- Age range: {age_range}\n"
        f"- Gender: {gender}\n"
        f"- Race: {race}\n"
        f"- Education: {education}\n"
        f"- Employment status: {employment}\n"
        f"- Work schedule: {work_sched}\n"
        f"- Occupation: {occupation}\n"
        f"- Primary activity: {primary_act}\n"
        f"- Work from home: {wfh}\n"
        f"- Driver on travel day: {driver}\n"
        f"- Distance to work (miles): {dist_work}\n"
        f"- Work state: {work_state}\n"
    )
    return [
        {"role": "system", "content": PLANNER_REALIZER_INSTRUCTION},
        {"role": "user",   "content": user_content},
    ]


def extract_planner_realizer_schedule(text: str):
    """从 api_merged 的 {reasoning, schedule} 输出中提取带时间的 schedule"""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    m = re.search(r'(\{.*\})', text, re.DOTALL)
    if not m:
        return None
    try:
        result = json.loads(m.group(1).strip())
        schedule = result.get("schedule", [])
        if isinstance(schedule, list) and schedule:
            cleaned = []
            for seg in schedule:
                if not all(k in seg for k in ("activity", "start_time", "end_time")):
                    return None
                act = _normalize_activity(seg["activity"])
                cleaned.append({"activity": act,
                                 "start_time": seg["start_time"],
                                 "end_time":   seg["end_time"]})
            return cleaned
        return None
    except Exception:
        return None


def build_editor_messages(user_profile: dict, initial_schedule: list) -> list:
    """Stage 2: Critic & Editor — 与 SFT 训练完全相同的 prompt"""
    user_content = (
        "**PERSON PROFILE:**\n"
        + json.dumps(user_profile, indent=2, ensure_ascii=False)
        + "\n\n**INITIAL SCHEDULE:**\n"
        + json.dumps(initial_schedule, indent=2, ensure_ascii=False)
    )
    return [
        {"role": "system", "content": EDITOR_INSTRUCTION},
        {"role": "user",   "content": user_content},
    ]


def extract_json_schedule(text: str):
    """从 [JSON]...[/JSON] 中提取 schedule（Editor 输出）"""
    m = re.search(r'\[JSON\](.*?)\[/JSON\]', text, re.DOTALL)
    if not m:
        return None
    try:
        result = json.loads(m.group(1).strip())
        return result if isinstance(result, list) else None
    except Exception:
        return None


def extract_planner_schedule(text: str):
    """从 Planner 输出中提取 schedule。
    支持两种格式：
      1. {reasoning, schedule} 对象（PLANNER_REALIZER_INSTRUCTION 统一输出）
      2. 纯 JSON 数组（兄容旧格式）
    """
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    alias = {
        "sleep": "home", "rest": "home", "sleeping": "home",
        "commute": "work", "commute to work": "work",
        "dining": "dine_out", "restaurant": "dine_out",
        "grocery": "shopping", "errands": "service",
        "gym": "exercise", "recreation": "exercise",
        "school": "education", "class": "education",
        "doctor": "medical", "hospital": "medical",
    }

    def _clean(segs):
        cleaned = []
        for seg in segs:
            act = seg.get("activity", "").lower()
            act = alias.get(act, act)
            if act not in ALLOWED_ACTIVITIES:
                act = "home"
            cleaned.append({
                "activity":   act,
                "start_time": seg.get("start_time", "00:00"),
                "end_time":   seg.get("end_time",   "24:00"),
            })
        return cleaned or None

    # 先尝试 {reasoning, schedule} 对象格式
    m_obj = re.search(r'(\{.*\})', text, re.DOTALL)
    if m_obj:
        try:
            result = json.loads(m_obj.group(1).strip())
            segs = result.get("schedule", [])
            if isinstance(segs, list) and segs:
                return _clean(segs)
        except Exception:
            pass

    # 回退：尝试纯 JSON 数组
    m_arr = re.search(r'(\[.*\])', text, re.DOTALL)
    if m_arr:
        try:
            result = json.loads(m_arr.group(1).strip())
            if isinstance(result, list) and result:
                return _clean(result)
        except Exception:
            pass

    return None


def _normalize_activity(act: str) -> str:
    """将别名映射到标准 ALLOWED_ACTIVITIES 中"""
    alias = {
        "sleep": "home", "rest": "home", "sleeping": "home",
        "commute": "work", "commute to work": "work",
        "dining": "dine_out", "restaurant": "dine_out", "eating out": "dine_out",
        "grocery": "shopping", "groceries": "shopping", "errands": "service",
        "gym": "exercise", "recreation": "exercise", "sports": "exercise",
        "school": "education", "class": "education", "university": "education",
        "doctor": "medical", "hospital": "medical", "clinic": "medical",
        "childcare": "dropoff_pickup", "pickup": "dropoff_pickup",
    }
    act = act.strip().lower()
    return alias.get(act, act) if alias.get(act, act) in ALLOWED_ACTIVITIES else "home"


def extract_planner_api(text: str):
    """从 API Planner 的结构化输出 {reasoning, activities} 中提取 activities 列表。
    返回形如 ["home","work",...,"home"] 的字符串列表，而非带时间的 schedule。"""
    # 去除 markdown 代码块
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    # 找 JSON 对象
    m = re.search(r'(\{.*\})', text, re.DOTALL)
    if not m:
        return None
    try:
        result = json.loads(m.group(1).strip())
        activities = result.get("activities", [])
        if not isinstance(activities, list) or not activities:
            return None
        # 只保留合法 activity
        valid = [_normalize_activity(a) for a in activities]
        # 确保首尾是 home
        if valid[0] != "home":
            valid.insert(0, "home")
        if valid[-1] != "home":
            valid.append("home")
        return valid if len(valid) >= 2 else None
    except Exception:
        return None


# ─── 模型加载 ─────────────────────────────────────────────────────────────────
def load_base_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[Base] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Base] 加载模型 (device_map=auto, bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    print("[Base] 加载完成 ✓")
    return tokenizer, model


def load_sft_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    print(f"[SFT] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[SFT] 加载 base 模型 ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"[SFT] 挂载 LoRA adapter: {SFT_MODEL_PATH} ...")
    model = PeftModel.from_pretrained(base_model, SFT_MODEL_PATH)
    model.eval()
    print("[SFT] 加载完成 ✓")
    return tokenizer, model


# ─── GRPO 单阶段推理 prompt（与 train_grpo.py 训练时完全一致）─────────────────
GRPO_SYSTEM_PROMPT = """You are a daily activity schedule generator. Your task is to generate a realistic daily schedule for a person based on their profile.

Output format:
[THOUGHT]
Brief reasoning about the person's schedule patterns.
[/THOUGHT]
[JSON]
[{"activity": "home", "start_time": "00:00", "end_time": "07:00"}, ...]
[/JSON]

The schedule must start at 00:00 and end at 24:00, covering the full day without gaps or overlaps."""


def build_grpo_messages(user_profile: dict) -> list:
    """GRPO 单阶段推理 — 与训练时 prompt 一致（仅 profile，无 initial_schedule）"""
    return [
        {"role": "system", "content": GRPO_SYSTEM_PROMPT},
        {"role": "user",   "content": "Generate a daily schedule for this person:\n"
                                       + json.dumps(user_profile, ensure_ascii=False)},
    ]


def load_grpo_model():
    """加载 GRPO 模型：base(CPU,bf16) → PeftModel SFT → merge → PeftModel GRPO → GPU"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[GRPO] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: 加载 base 模型（CPU，bf16，节省内存）
    print(f"[GRPO] 加载 base 模型（CPU, bf16）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        trust_remote_code=True,
    )

    # Step 2: 用 PeftModel.from_pretrained 加载 SFT LoRA (CPU)，然后 merge
    print(f"[GRPO] 加载并合并 SFT LoRA: {SFT_MODEL_PATH} ...")
    sft_model = PeftModel.from_pretrained(
        base_model, SFT_MODEL_PATH, device_map={"": "cpu"}
    )
    merged_model = sft_model.merge_and_unload()
    # 清除残留的 peft_config，防止 PeftModel 嵌套出现 CUDA 设备问题
    for _attr in ("peft_config", "active_adapters", "_peft_config"):
        if hasattr(merged_model, _attr):
            try:
                delattr(merged_model, _attr)
            except Exception:
                pass
    del sft_model, base_model
    print("  ✓ SFT LoRA 合并完成")

    # Step 3: 用 PeftModel.from_pretrained 加载 GRPO LoRA (CPU)
    print(f"[GRPO] 挂载 GRPO LoRA adapter: {GRPO_MODEL_PATH} ...")
    model = PeftModel.from_pretrained(
        merged_model, GRPO_MODEL_PATH, device_map={"": "cpu"}
    )
    del merged_model

    # Step 4: 整体移到 GPU
    print("[GRPO] 移动至 GPU ...")
    model = model.to("cuda")
    model.eval()
    print("[GRPO] 加载完成 ✓")
    return tokenizer, model


def load_sft_llama_model():
    """加载 LLaMA SFT 模型：LLaMA base + SFT LoRA（两阶段推理用）"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    print(f"[SFT-LLaMA] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_BASE_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[SFT-LLaMA] 加载 base 模型 ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_BASE_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"[SFT-LLaMA] 挂载 LoRA adapter: {LLAMA_SFT_PATH} ...")
    model = PeftModel.from_pretrained(base_model, LLAMA_SFT_PATH)
    model.eval()
    print("[SFT-LLaMA] 加载完成 ✓")
    return tokenizer, model


def load_grpo_llama_model():
    """加载 LLaMA GRPO 模型：base(CPU,bf16) → PeftModel SFT → merge → PeftModel GRPO → GPU"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"[GRPO-LLaMA] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_BASE_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[GRPO-LLaMA] 加载 base 模型（CPU, bf16）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_BASE_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"":"cpu"},
        trust_remote_code=True,
    )

    print(f"[GRPO-LLaMA] 加载并合并 SFT LoRA: {LLAMA_GRPO_SFT_PATH} ...")
    sft_model = PeftModel.from_pretrained(
        base_model, LLAMA_GRPO_SFT_PATH, device_map={"":"cpu"}
    )
    merged_model = sft_model.merge_and_unload()
    for _attr in ("peft_config", "active_adapters", "_peft_config"):
        if hasattr(merged_model, _attr):
            try:
                delattr(merged_model, _attr)
            except Exception:
                pass
    del sft_model, base_model
    print("  ✓ SFT LoRA 合并完成")

    print(f"[GRPO-LLaMA] 挂载 GRPO LoRA adapter: {LLAMA_GRPO_PATH} ...")
    model = PeftModel.from_pretrained(
        merged_model, LLAMA_GRPO_PATH, device_map={"":"cpu"}
    )
    del merged_model

    print("[GRPO-LLaMA] 移动至 GPU ...")
    model = model.to("cuda")
    model.eval()
    print("[GRPO-LLaMA] 加载完成 ✓")
    return tokenizer, model


def call_model(tokenizer, model, messages: list, max_new_tokens: int = 1024) -> str:
    """通用单次推理（本地模型）"""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def call_api(client, messages: list, model: str,
             max_tokens: int = 1024, temperature: float = 0.7,
             max_retries: int = 3) -> str:
    """通用 API 调用，支持 str / choices / dict 三种响应格式，带重试"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # 兼容三种响应格式（参考 generate_trajectories_three_agent_pipeline.py）
            if isinstance(response, str):
                return response
            elif hasattr(response, "choices"):
                return response.choices[0].message.content.strip()
            else:
                return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"    [API] 第 {attempt+1} 次调用失败: {str(e)[:120]}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2)
    return ""


def build_trip_realizer_messages(user_profile: dict, activities: list) -> list:
    """Stage 2 (API 后端): Trip Realizer — 活动序列 → 带时间的日程，包含全部静态属性"""
    age_range    = user_profile.get("age_range", "Unknown")
    gender       = user_profile.get("gender", "Unknown")
    race         = user_profile.get("race", "Unknown")
    education    = user_profile.get("education", "Unknown")
    employment   = user_profile.get("employment_status", "Unknown")
    work_sched   = user_profile.get("work_schedule", "Unknown")
    occupation   = user_profile.get("occupation", "Unknown")
    primary_act  = user_profile.get("primary_activity", "Unknown")
    wfh          = user_profile.get("work_from_home", "Unknown")
    driver       = user_profile.get("driver_on_travel_day", "Unknown")
    dist_work    = user_profile.get("distance_to_work_miles", "Unknown")
    work_state   = user_profile.get("work_state", "Unknown")
    acts_str     = " → ".join(activities)

    prompt = f"""You are a Trip Realizer Agent. Assign realistic start/end times to this activity sequence.

USER PROFILE:
- Age range: {age_range}
- Gender: {gender}
- Race: {race}
- Education: {education}
- Employment: {employment}
- Work schedule: {work_sched}
- Occupation: {occupation}
- Primary activity: {primary_act}
- Work from home: {wfh}
- Driver on travel day: {driver}
- Distance to work (miles): {dist_work}
- Work state: {work_state}

ACTIVITY SEQUENCE: {acts_str}

RULES:
1. Cover FULL 24 hours: 00:00 to 24:00, NO gaps or overlaps
2. Consecutive activities must connect seamlessly (end_time = next start_time)
3. Realistic durations: home(night) 6-9h, work(full-time) 8-9h, shopping/medical 1-2h
4. The first activity starts at 00:00, the last ends at 24:00

OUTPUT FORMAT:
```json
{{
  "schedule": [
    {{"activity": "home",  "start_time": "00:00", "end_time": "07:30"}},
    {{"activity": "work",  "start_time": "07:30", "end_time": "17:00"}},
    {{"activity": "home",  "start_time": "17:00", "end_time": "24:00"}}
  ]
}}
```"""
    return [
        {"role": "system", "content": "You assign realistic times to daily activities, ensuring full 24-hour coverage."},
        {"role": "user",   "content": prompt},
    ]


def extract_trip_realizer_schedule(text: str):
    """从 Trip Realizer 输出中提取带时间的 schedule"""
    # 去除 markdown
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    # 找 JSON 对象
    m = re.search(r'(\{.*\})', text, re.DOTALL)
    if not m:
        return None
    try:
        result = json.loads(m.group(1).strip())
        schedule = result.get("schedule", [])
        if isinstance(schedule, list) and schedule:
            # 基础验证
            for seg in schedule:
                if not all(k in seg for k in ("activity", "start_time", "end_time")):
                    return None
            return schedule
        return None
    except Exception:
        return None


def _make_fallback_schedule(activities: list) -> list:
    """当 Trip Realizer 失败时，按等分生成粗略日程"""
    if not activities:
        return [{"activity": "home", "start_time": "00:00", "end_time": "24:00"}]
    n = len(activities)
    mins_each = 1440 // n
    schedule = []
    cur = 0
    for i, act in enumerate(activities):
        start = cur
        end   = 1440 if i == n - 1 else cur + mins_each
        h_s, m_s = divmod(start, 60)
        h_e, m_e = divmod(end,   60)
        schedule.append({
            "activity":   act,
            "start_time": f"{h_s:02d}:{m_s:02d}",
            "end_time":   f"{h_e:02d}:{m_e:02d}",
        })
        cur = end
    return schedule


# ─── 历史汇总表追加 ────────────────────────────────────────────────────────────
def _append_to_history(results: dict, args, n: int, ts: str,
                       planner_fail: int, editor_fail: int,
                       gen_filename: str):
    """将本次评估结果追加到 HISTORY_CSV（固定路径），首次运行时自动写表头。"""
    import csv

    # 与 experiment_results.csv 相同的指标列顺序，再加上本次运行的上下文字段
    METRIC_COLS = [
        "accuracy", "f1_score", "edit_dist", "bleu_score", "data_jsd",
        "macro_int", "micro_int", "act_type", "uni_act_type",
        "traj_len", "macro_hour", "micro_hour",
    ]
    HEADER = (
        ["run_time", "model", "num_samples", "planner_fail", "editor_fail",
         "generated_file"]
        + METRIC_COLS
    )

    # results 的 key 有时带连字符（f1-score），统一处理为下划线
    def get_metric(key):
        val = results.get(key, results.get(key.replace("_", "-"), ""))
        return f"{val:.6f}" if isinstance(val, float) else str(val)

    row = (
        [ts, args.model.upper(), str(n), str(planner_fail), str(editor_fail),
         gen_filename]
        + [get_metric(k) for k in METRIC_COLS]
    )

    need_header = not os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(HEADER)
        writer.writerow(row)

    print(f"\n[历史汇总] 已追加到: {HISTORY_CSV}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, required=True,
                        choices=["base", "sft", "grpo", "sft_llama", "grpo_llama", "api", "api_merged", "api_single"],
                        help="推理后端：base/sft/grpo/sft_llama/grpo_llama（本地模型）| api（三阶段）| api_merged | api_single")
    parser.add_argument("--data",        type=str, default=DEFAULT_DATA,
                        help="输入数据路径（metadata.jsonl 格式）")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="取前 N 条，默认全部")
    parser.add_argument("--use_stored_initial", action="store_true",
                        help="跳过 Planner，直接用 metadata 里存储的 initial_schedule")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Editor 阶段最大生成 token 数（本地模型）")
    parser.add_argument("--api_planner_model", type=str, default=API_PLANNER_MODEL,
                        help=f"API Planner 使用的模型（默认 {API_PLANNER_MODEL}）")
    parser.add_argument("--api_editor_model",  type=str, default=API_EDITOR_MODEL,
                        help=f"API Editor 使用的模型（默认 {API_EDITOR_MODEL}）")
    parser.add_argument("--skip_editor", action="store_true",
                        help="跳过 Critic/Editor 阶段（消融：单智能体模式），Planner 输出直接作为最终结果")
    args = parser.parse_args()

    # 读取数据
    samples = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.num_samples:
        samples = samples[:args.num_samples]
    n = len(samples)

    is_api        = (args.model in ("api", "api_merged", "api_single"))
    is_api_merged = (args.model == "api_merged")   # Planner+Realizer 合并为一次调用
    is_api_single = (args.model == "api_single")   # 消融：无Planner/Realizer
    is_grpo       = (args.model in ("grpo", "grpo_llama"))  # GRPO 单阶段本地模型

    if is_api_single:
        pipeline_desc = "[消融 A5] 直接 Profile → API Editor（单阶段，无Planner/Realizer）"
    elif is_api_merged:
        pipeline_desc = "[消融 A3] API Planner+Realizer（合并）→ API Editor（两阶段）"
    elif is_api:
        pipeline_desc = ("Stored Initial → API Editor"
                         if args.use_stored_initial
                         else "API Planner → API Trip Realizer → API Editor")
    else:
        pipeline_desc = ("Stored Initial → Editor"
                         if args.use_stored_initial
                         else ("GRPO single-stage (profile → schedule)"
                               if is_grpo else "Planner → Editor"))

    print(f"\n数据来源: {args.data}")
    print(f"样本数量: {n}")
    print(f"Pipeline: {pipeline_desc} [{args.model.upper()}]")

    # 加载模型 / 初始化 API 客户端
    if is_api:
        from openai import OpenAI as _OpenAI
        api_client = _OpenAI(api_key=API_KEY, base_url=API_BASE_URL, timeout=API_TIMEOUT)
        tokenizer  = None
        model_obj  = None
        if not is_api_single:
            print(f"[API] Planner  模型: {args.api_planner_model}")
        print(f"[API] Editor   模型: {args.api_editor_model}")
    else:
        if args.model == "base":
            tokenizer, model_obj = load_base_model()
        elif args.model == "grpo":
            tokenizer, model_obj = load_grpo_model()
        elif args.model == "sft_llama":
            tokenizer, model_obj = load_sft_llama_model()
        elif args.model == "grpo_llama":
            tokenizer, model_obj = load_grpo_llama_model()
        else:
            tokenizer, model_obj = load_sft_model()
        api_client = None

    # 推理
    generated_results    = []
    ground_truth_results = []
    planner_fail = 0
    realizer_fail = 0
    editor_fail  = 0

    for i, sample in enumerate(samples):
        user_id        = sample.get("user_id", f"sample_{i}")
        user_profile   = sample["user_profile"]
        gt_sched       = sample["ground_truth_schedule"]
        stored_initial = sample.get("initial_schedule", [])

        t0 = time.time()
        sep = "─" * 60
        print(f"\n{sep}", flush=True)
        print(f"[{i+1}/{n}] user_id={user_id}", flush=True)
        print(f"  Profile: age={user_profile.get('age_range','?')}  "
              f"gender={user_profile.get('gender','?')}  "
              f"employment={user_profile.get('employment_status','?')}  "
              f"occupation={user_profile.get('occupation','?')}", flush=True)

        # ── Stage 1: Planner ──────────────────────────────────────────────────
        if is_api_single:
            # 消融实验：跳过全部前置 Agent，以空日程直接进 Editor
            initial_sched = []
            print(f"  [消融 A5] 跳过 Planner + Realizer，直接进入 Editor", flush=True)
        elif is_api_merged:
            # 消融实验：Planner + Realizer 合并为一次 API 调用
            print(f"  [Stage 1+2 / Merged ] API 生成带时间完整日程 ...", flush=True)
            t1 = time.time()
            try:
                merged_msgs = build_planner_realizer_messages(user_profile)
                merged_out  = call_api(api_client, merged_msgs,
                                       args.api_planner_model,
                                       max_tokens=1024, temperature=0.7)
                initial_sched = extract_planner_realizer_schedule(merged_out)
                if initial_sched:
                    acts = " → ".join(s["activity"] for s in initial_sched)
                    print(f"  [Stage 1+2 / Merged ] ✓ {len(initial_sched)} 段  ({time.time()-t1:.1f}s)", flush=True)
                    print(f"    {acts}", flush=True)
                else:
                    planner_fail += 1
                    print(f"  [Stage 1+2 / Merged ] ✗ 解析失败，使用回退日程  ({time.time()-t1:.1f}s)", flush=True)
                    print(f"    原始输出: {repr(merged_out[:200])}", flush=True)
                    initial_sched = stored_initial or [{"activity": "home",
                                                        "start_time": "00:00",
                                                        "end_time": "24:00"}]
            except Exception as e:
                planner_fail += 1
                print(f"  [Stage 1+2 / Merged ] ✗ 出错: {e}", flush=True)
                initial_sched = stored_initial or [{"activity": "home",
                                                    "start_time": "00:00",
                                                    "end_time": "24:00"}]
        elif args.use_stored_initial:
            initial_sched = stored_initial
            print(f"  [Stage 1 / Planner  ] 使用已存储的 initial_schedule ({len(initial_sched)} 段)", flush=True)
        elif is_api:
            # ── API Stage 1: Planner (活动骨架) ──────────────────────────────
            print(f"  [Stage 1 / Planner  ] API 生成活动骨架 ...", flush=True)
            t1 = time.time()
            activities = None
            try:
                planner_msgs = build_planner_api_messages(user_profile)
                planner_out  = call_api(api_client, planner_msgs,
                                        args.api_planner_model,
                                        max_tokens=512, temperature=0.7)
                activities = extract_planner_api(planner_out)
                if activities:
                    print(f"  [Stage 1 / Planner  ] ✓ {len(activities)} 个活动  ({time.time()-t1:.1f}s)",
                          flush=True)
                    print(f"    {' → '.join(activities)}", flush=True)
                else:
                    planner_fail += 1
                    print(f"  [Stage 1 / Planner  ] ✗ 解析失败  ({time.time()-t1:.1f}s)", flush=True)
                    print(f"    原始输出: {repr(planner_out[:200])}", flush=True)
            except Exception as e:
                planner_fail += 1
                print(f"  [Stage 1 / Planner  ] ✗ 出错: {e}", flush=True)

            # ── API Stage 2: Trip Realizer (分配时间) ─────────────────────────
            if activities:
                print(f"  [Stage 2 / Realizer ] API 分配时间 ...", flush=True)
                t2 = time.time()
                try:
                    realizer_msgs = build_trip_realizer_messages(user_profile, activities)
                    realizer_out  = call_api(api_client, realizer_msgs,
                                            args.api_planner_model,
                                            max_tokens=800, temperature=0.5)
                    initial_sched = extract_trip_realizer_schedule(realizer_out)
                    if initial_sched:
                        acts = " → ".join(s["activity"] for s in initial_sched)
                        print(f"  [Stage 2 / Realizer ] ✓ {len(initial_sched)} 段  ({time.time()-t2:.1f}s)",
                              flush=True)
                        print(f"    {acts}", flush=True)
                    else:
                        realizer_fail += 1
                        print(f"  [Stage 2 / Realizer ] ✗ 解析失败，使用等分回退  ({time.time()-t2:.1f}s)",
                              flush=True)
                        initial_sched = _make_fallback_schedule(activities)
                except Exception as e:
                    realizer_fail += 1
                    print(f"  [Stage 2 / Realizer ] ✗ 出错: {e}", flush=True)
                    initial_sched = _make_fallback_schedule(activities)
            else:
                # Planner 也失败，回退到存储日程
                initial_sched = stored_initial or [{"activity": "home",
                                                    "start_time": "00:00",
                                                    "end_time": "24:00"}]
                print(f"  [Stage 2 / Realizer ] 跳过（Planner 失败），使用回退日程", flush=True)
        elif not is_grpo:
            # ── 本地模型 Stage 1: Planner ─────────────────────────────────────
            print(f"  [Stage 1 / Planner  ] 生成初始日程 ...", flush=True)
            t1 = time.time()
            try:
                planner_msgs  = build_planner_messages(user_profile)
                planner_out   = call_model(tokenizer, model_obj, planner_msgs, max_new_tokens=512)
                initial_sched = extract_planner_schedule(planner_out)
                if initial_sched:
                    acts = " → ".join(s["activity"] for s in initial_sched)
                    print(f"  [Stage 1 / Planner  ] ✓ {len(initial_sched)} 段  ({time.time()-t1:.1f}s)", flush=True)
                    print(f"    {acts}", flush=True)
                else:
                    planner_fail += 1
                    print(f"  [Stage 1 / Planner  ] ✗ 解析失败，改用存储初始日程  ({time.time()-t1:.1f}s)", flush=True)
                    print(f"    原始输出: {repr(planner_out[:200])}", flush=True)
                    initial_sched = stored_initial or []
            except Exception as e:
                planner_fail += 1
                print(f"  [Stage 1 / Planner  ] ✗ 出错: {e}", flush=True)
                initial_sched = stored_initial or []
        else:
            initial_sched = []   # GRPO 单阶段，不需要 initial_sched

        # ── Stage Editor: Critic & Editor / GRPO 单阶段 ──────────────────────
        if is_grpo:
            # GRPO 单阶段：直接从 profile 生成日程，跳过 Editor
            print(f"  [GRPO / Single-stage] 生成日程 ...", flush=True)
            t_e = time.time()
            try:
                grpo_msgs   = build_grpo_messages(user_profile)
                grpo_out    = call_model(tokenizer, model_obj, grpo_msgs,
                                         max_new_tokens=args.max_new_tokens)
                final_sched = extract_json_schedule(grpo_out)
                if final_sched:
                    acts = " → ".join(s["activity"] for s in final_sched)
                    print(f"  [GRPO / Single-stage] ✓ {len(final_sched)} 段  ({time.time()-t_e:.1f}s)", flush=True)
                    print(f"    {acts}", flush=True)
                else:
                    editor_fail += 1
                    print(f"  [GRPO / Single-stage] ✗ 解析失败  ({time.time()-t_e:.1f}s)", flush=True)
                    print(f"    原始输出: {repr(grpo_out[:300])}", flush=True)
                    final_sched = []
            except Exception as e:
                editor_fail += 1
                print(f"  [GRPO / Single-stage] ✗ 出错: {e}", flush=True)
                final_sched = []
        elif args.skip_editor:
            # ── 消融：跳过 Editor，直接用 Planner 输出 ────────────────────────
            final_sched = initial_sched
            acts = " → ".join(s["activity"] for s in final_sched) if final_sched else "(空)"
            print(f"  [Single-Agent / Skip Editor] 直接使用 Planner 输出 ({len(final_sched)} 段): {acts}", flush=True)
        else:
            # ── 非 GRPO：走 Stage Editor ──────────────────────────────────────
            if is_api_single:
                stage_label = "消融实验 / Stage 1"
            elif is_api_merged:
                stage_label = "Stage 2"
            elif is_api and not args.use_stored_initial:
                stage_label = "Stage 3"
            else:
                stage_label = "Stage 2"
            print(f"  [{stage_label} / Editor  ] 精炼日程 ...", flush=True)
            t_e = time.time()
            try:
                editor_msgs = build_editor_messages(user_profile, initial_sched)
                if is_api:
                    editor_out = call_api(api_client, editor_msgs,
                                          args.api_editor_model,
                                          max_tokens=args.max_new_tokens, temperature=0.7)
                else:
                    editor_out = call_model(tokenizer, model_obj, editor_msgs,
                                            max_new_tokens=args.max_new_tokens)
                final_sched = extract_json_schedule(editor_out)

                if final_sched:
                    acts = " → ".join(s["activity"] for s in final_sched)
                    print(f"  [{stage_label} / Editor  ] ✓ {len(final_sched)} 段  ({time.time()-t_e:.1f}s)", flush=True)
                    print(f"    {acts}", flush=True)
                else:
                    editor_fail += 1
                    print(f"  [{stage_label} / Editor  ] ✗ 解析失败  ({time.time()-t_e:.1f}s)", flush=True)
                    print(f"    原始输出: {repr(editor_out[:300])}", flush=True)
                    final_sched = []
            except Exception as e:
                editor_fail += 1
                print(f"  [{stage_label} / Editor  ] ✗ 出错: {e}", flush=True)
                final_sched = []

        gt_acts = " → ".join(s["activity"] for s in gt_sched)
        print(f"  [Ground Truth       ] {len(gt_sched)} 段: {gt_acts}", flush=True)
        extra = f"  Realizer失败={realizer_fail}" if (is_api and not is_api_single) else ""
        print(f"  耗时: {time.time()-t0:.1f}s  |  累计进度: {i+1}/{n}  "
              f"Planner失败={planner_fail}{extra}  Editor失败={editor_fail}", flush=True)

        generated_results.append({"user_id": user_id, "schedule": final_sched})
        ground_truth_results.append({"user_id": user_id, "schedule": gt_sched})

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_tag = args.model + ("_single" if args.skip_editor else "")
    gen_path = os.path.join(RESULTS_DIR, f"generated_{model_tag}_{n}samples_{ts}.json")
    gt_path  = os.path.join(RESULTS_DIR, f"ground_truth_{n}samples_{ts}.json")

    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(generated_results, f, ensure_ascii=False, indent=2)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  模型: {args.model.upper()}   样本: {n}")
    print(f"  Planner 失败: {planner_fail}   Editor 解析失败: {editor_fail}")
    print(f"  生成结果: {gen_path}")
    print(f"  Ground Truth: {gt_path}")
    print(f"{'='*60}")

    # ── 自动评估 ──────────────────────────────────────────────────────────────
    print(f"\n[评估] 开始计算评估指标 ...")
    eval_out_dir = os.path.join(RESULTS_DIR, f"evaluation_{model_tag}_{ts}")
    os.makedirs(eval_out_dir, exist_ok=True)

    try:
        from evaluate_generated_trajectories import evaluate_trajectories
        results = evaluate_trajectories(
            generated_file = gen_path,
            original_file  = gt_path,
            output_dir     = eval_out_dir,
            save_csv       = True,
        )
        if results:
            print(f"\n{'='*60}")
            print(f"  评估结果摘要 [{args.model.upper()}]")
            print(f"{'='*60}")
            key_metrics = ["accuracy", "f1-score", "edit_dist", "act_type",
                           "macro_int", "micro_int", "bleu_score", "data_jsd"]
            for k in key_metrics:
                if k in results:
                    print(f"  {k:20s}: {results[k]:.6f}")
            print(f"{'='*60}")
            print(f"  详细报告: {eval_out_dir}")

            # ── 追加到固定历史汇总表 ──────────────────────────────────────────
            _append_to_history(results, args, n, ts,
                               planner_fail, editor_fail,
                               os.path.basename(gen_path))
    except Exception as e:
        print(f"[评估] 出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
