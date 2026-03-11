"""
Stage 3: GRPO Training for Trajectory Generation
- 加载 Stage 2 SFT 的 LoRA checkpoint，merge 后作为起点
- 使用 eval_example.py 中的 acc/f1/edit_dist 指标与 ground_truth_schedule 对比作为 reward
- Unsloth + vLLM 框架（参考 datawhalechina/self-llm Qwen3 GRPO 教程）
"""

import os, re, sys, json, random
import numpy as np
import torch
from datasets import Dataset

# ── torch 2.2.1 + transformers 5.0.0 兼容性修复 ──────────────────────────────
if hasattr(torch, 'is_autocast_enabled'):
    _orig_is_autocast_enabled = torch.is_autocast_enabled
    def _patched_is_autocast_enabled(device_type=None):
        return _orig_is_autocast_enabled()
    torch.is_autocast_enabled = _patched_is_autocast_enabled

# ─── 路径配置 ────────────────────────────────────────────────────────────────
BASE_MODEL_PATH = "/data/alice/cjtest/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B"
SFT_MODEL_PATH  = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage2_sft_output_epoch10/final_model"
DATA_PATH       = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage1_training_data_3/metadata.jsonl"
OUTPUT_DIR      = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage3_grpo_output"
MAX_SEQ_LENGTH  = 4096
MAX_PROMPT_LEN  = 2048
MAX_COMPLETION_LEN = 2048

# ─── Activity 编码（与 evaluate_generated_trajectories.py 一致）──────────────
ACTIVITY_NAME_CODE_MAPPING = {
    'home': 1, 'work': 2, 'education': 3, 'shopping': 4, 'service': 5,
    'medical': 6, 'dine_out': 7, 'socialize': 8, 'exercise': 9, 'dropoff_pickup': 10,
}
TIMESTEP_MINUTES = 15
N_TIMESTEPS = 96   # 24h / 15min = 96


# ─── 时间表 → 96 时间步整数数组 ───────────────────────────────────────────────
def time_to_minutes(time_str: str) -> int:
    if time_str in ("24:00", "23:59"):
        return 1440
    parts = time_str.strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])


def schedule_to_96_timesteps(schedule: list) -> np.ndarray:
    """将 [{'activity':str,'start_time':'HH:MM','end_time':'HH:MM'}, ...] 转成形状 (96,) 的整数数组"""
    timesteps = np.zeros(N_TIMESTEPS, dtype=int)
    if not schedule:
        return timesteps
    for slot_idx in range(N_TIMESTEPS):
        slot_start = slot_idx * TIMESTEP_MINUTES
        slot_end = (slot_idx + 1) * TIMESTEP_MINUTES
        activity_durations: dict = {}
        for seg in schedule:
            act_name = seg.get("activity", "home")
            seg_start = time_to_minutes(seg.get("start_time", "00:00"))
            seg_end   = time_to_minutes(seg.get("end_time",   "24:00"))
            overlap_start = max(slot_start, seg_start)
            overlap_end   = min(slot_end,   seg_end)
            if overlap_end > overlap_start:
                activity_durations[act_name] = activity_durations.get(act_name, 0) + (overlap_end - overlap_start)
        if activity_durations:
            dominant = max(activity_durations, key=activity_durations.get)
            timesteps[slot_idx] = ACTIVITY_NAME_CODE_MAPPING.get(dominant, 0)
    return timesteps


# ─── 从模型输出中提取 JSON 日程 ───────────────────────────────────────────────
_JSON_PAT = re.compile(r"\[JSON\](.*?)\[/JSON\]", re.DOTALL)

def extract_schedule_from_output(text: str):
    """从 [JSON]...[/JSON] 中提取日程列表，失败返回 None"""
    m = _JSON_PAT.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1).strip())
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass
    return None


# ─── 每样本相似度指标（对齐 eval_example.py：acc / macro_int_jsd / act_type_jsd / f1）
def _compute_int_single(arr: np.ndarray) -> np.ndarray:
    """
    对单条序列 (96,) 计算各活动的连续段长度分布。
    返回 act2int: shape (11, 96)，act2int[act, len-1] = 该活动出现长度为 len 的次数。
    对应 eval_example.py 中 compute_int()，但针对单条而非批量。
    """
    act2int = np.zeros((11, N_TIMESTEPS), dtype=float)
    curr_act = int(arr[0])
    curr_len = 1
    for j in range(1, len(arr)):
        a = int(arr[j])
        if a == curr_act:
            curr_len += 1
        else:
            act2int[curr_act, curr_len - 1] += 1
            curr_act, curr_len = a, 1
    act2int[curr_act, curr_len - 1] += 1
    return act2int


def compute_schedule_similarity(gen_schedule, gt_schedule) -> float:
    """
    返回 0~1 的相似度分数（越高越好），对齐 evaluate_generated_trajectories.py 的评估指标。

    权重分配（侧重 accuracy + 活动类型分布 + 活动段长分布）：
      accuracy        0.40  — 逐 slot 精确命中率
      act_type_sim    0.25  — 活动类型频率分布相似度  (1 - act_type_jsd)
      macro_int_sim   0.25  — 活动段长度分布相似度    (1 - macro_int_jsd)
      f1              0.10  — macro-F1（补充细粒度）
    """
    from scipy.spatial import distance as sp_dist
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        f1_score = None

    gen_arr = schedule_to_96_timesteps(gen_schedule)  # (96,)
    gt_arr  = schedule_to_96_timesteps(gt_schedule)   # (96,)

    # ── 1. accuracy ────────────────────────────────────────────────────────
    accuracy = float(np.sum(gen_arr == gt_arr)) / N_TIMESTEPS

    # ── 2. act_type_jsd（eval_example.py: act_type_jsd）─────────────────
    #   act2cnt[i] = 该活动类型在序列中占用的 slot 总数，shape (11,)
    gen_act_cnt = np.array([float(np.sum(gen_arr == i)) for i in range(11)])
    gt_act_cnt  = np.array([float(np.sum(gt_arr  == i)) for i in range(11)])
    gen_act_p = gen_act_cnt / (np.sum(gen_act_cnt) + 1e-9)
    gt_act_p  = gt_act_cnt  / (np.sum(gt_act_cnt)  + 1e-9)
    act_type_jsd = float(sp_dist.jensenshannon(gen_act_p, gt_act_p))
    act_type_sim = 1.0 - min(act_type_jsd, 1.0)

    # ── 3. macro_int_jsd（eval_example.py: macro_micro_int_jsd 的 macro 部分）
    #   对单样本：act2int shape (11, 96)，macro = 不区分活动类型的段长度分布
    gen_act2int = _compute_int_single(gen_arr)  # (11, 96)
    gt_act2int  = _compute_int_single(gt_arr)
    gen_int_sum = np.sum(gen_act2int)
    gt_int_sum  = np.sum(gt_act2int)
    if gen_int_sum > 0 and gt_int_sum > 0:
        gen_macro = np.sum(gen_act2int, axis=0) / gen_int_sum  # (96,)
        gt_macro  = np.sum(gt_act2int,  axis=0) / gt_int_sum
        macro_int_jsd = float(sp_dist.jensenshannon(gen_macro, gt_macro))
    else:
        macro_int_jsd = 1.0
    macro_int_sim = 1.0 - min(macro_int_jsd, 1.0)

    # ── 4. macro-F1 ────────────────────────────────────────────────────────
    if f1_score is not None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = float(f1_score(gt_arr, gen_arr, average='macro', zero_division=0))
    else:
        f1 = accuracy  # fallback

    # ── 加权组合 ────────────────────────────────────────────────────────────
    score = 0.40 * accuracy + 0.25 * act_type_sim + 0.25 * macro_int_sim + 0.10 * f1
    return float(score)


# ═══════════════════════════════════════════════════════════════════════════════
#  Reward Functions（GRPO 训练时调用）
#  completions: list[list[dict]]  每条为 [{"role":"assistant","content":"..."}]
#  ground_truth_schedule: list  每个元素是 GT 日程（Python list 或 JSON 字符串）
# ═══════════════════════════════════════════════════════════════════════════════

# ── TRL 0.27.1 调用约定：reward_fn(completions, **kwargs)
# completions: list[list[dict]]，kwargs 含 prompts + 数据集其他列
def reward_format(completions, prompts=None, ground_truth_schedule=None, **kwargs) -> list:
    """
    格式奖励：
    +1.0  含 [THOUGHT]...[/THOUGHT] 且含 [JSON]...[/JSON]
    +0.5  仅含其中一个
    +0.0  都没有
    """
    scores = []
    for completion in completions:
        resp = completion[0]["content"] if isinstance(completion, list) else completion
        has_thought = bool(re.search(r"\[THOUGHT\].*?\[/THOUGHT\]", resp, re.DOTALL))
        has_json    = bool(re.search(r"\[JSON\].*?\[/JSON\]",    resp, re.DOTALL))
        if has_thought and has_json:
            scores.append(1.0)
        elif has_thought or has_json:
            scores.append(0.5)
        else:
            scores.append(0.0)
    return scores


def reward_schedule_similarity(completions, prompts=None, ground_truth_schedule=None, **kwargs) -> list:
    """
    日程相似度奖励：将生成的日程与 ground_truth 逐 slot 对比（acc + F1 + edit_sim）。
    无法解析时返回 0.0。
    """
    if ground_truth_schedule is None:
        return [0.0] * len(completions)
    scores = []
    for completion, gt_raw in zip(completions, ground_truth_schedule):
        resp = completion[0]["content"] if isinstance(completion, list) else completion
        gen_schedule = extract_schedule_from_output(resp)
        if gen_schedule is None:
            scores.append(0.0)
            continue
        # ground_truth_schedule 可能是 Python list 或 JSON 字符串
        if isinstance(gt_raw, str):
            try:
                gt_schedule = json.loads(gt_raw)
            except Exception:
                scores.append(0.0)
                continue
        else:
            gt_schedule = gt_raw
        sim = compute_schedule_similarity(gen_schedule, gt_schedule)
        scores.append(sim)
    return scores


def reward_constraints(completions, prompts=None, ground_truth_schedule=None, **kwargs) -> list:
    """
    基本约束奖励（不依赖 GT，纯结构性验证）：
    - 日程覆盖全天（00:00 ~ 24:00）
    - 时间段连续不重叠
    - 每段时长合理（≥5 分钟，≤720 分钟）
    每项约束 +1/3，最终归到 [0, 1]。
    """
    scores = []
    for completion in completions:
        resp = completion[0]["content"] if isinstance(completion, list) else completion
        schedule = extract_schedule_from_output(resp)
        if not schedule:
            scores.append(0.0)
            continue
        try:
            segs = []
            for seg in schedule:
                start = time_to_minutes(seg.get("start_time", "00:00"))
                end   = time_to_minutes(seg.get("end_time",   "24:00"))
                segs.append((start, end))
            segs.sort(key=lambda x: x[0])

            covers_day    = (segs[0][0] == 0 and segs[-1][1] == 1440)
            continuous    = all(segs[i][1] == segs[i+1][0] for i in range(len(segs)-1))
            durations_ok  = all(5 <= (e - s) <= 720 for s, e in segs)

            score = (float(covers_day) + float(continuous) + float(durations_ok)) / 3.0
        except Exception:
            score = 0.0
        scores.append(score)
    return scores


# ─── 数据集构建 ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a daily activity schedule generator. Your task is to generate a realistic daily schedule for a person based on their profile.

Output format:
[THOUGHT]
Brief reasoning about the person's schedule patterns.
[/THOUGHT]
[JSON]
[{"activity": "home", "start_time": "00:00", "end_time": "07:00"}, ...]
[/JSON]

The schedule must start at 00:00 and end at 24:00, covering the full day without gaps or overlaps."""

def build_prompt(user_profile: dict) -> list:
    profile_str = json.dumps(user_profile, ensure_ascii=False)
    return [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"Generate a daily schedule for this person:\n{profile_str}"},
    ]


def load_grpo_dataset(data_path: str, max_samples: int = None) -> Dataset:
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append({
                "prompt": build_prompt(obj["user_profile"]),
                # ground_truth 保持为 Python list，reward_funcs 直接使用
                "ground_truth_schedule": obj["ground_truth_schedule"],
            })

    if max_samples:
        random.shuffle(records)
        records = records[:max_samples]

    print(f"[Dataset] Loaded {len(records)} samples from {data_path}")
    return Dataset.from_list(records)


# ─── 主训练流程（标准 transformers + peft + TRL，trajlla 环境直接可用）────────
def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import GRPOConfig, GRPOTrainer

    # 1. 加载 Tokenizer
    print(f"[1/4] Loading tokenizer from: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载 Base 模型 + 合并 SFT LoRA（全程 CPU）
    # 绕过 PeftModel.from_pretrained → load_peft_weights → safe_load_file 的 CUDA 问题，
    # 直接用 safetensors.torch.load_file(device="cpu") 手动加载 adapter 权重。
    print(f"[2/4] Loading base model + merging SFT LoRA (manual CPU load)...")
    from peft import PeftConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file as st_load_file

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype       = torch.float32,   # CPU merge 用 float32
        device_map        = {"": "cpu"},
        trust_remote_code = True,
    )

    # 读取 SFT adapter 配置并套上 LoRA 结构
    peft_config = PeftConfig.from_pretrained(SFT_MODEL_PATH)
    model = get_peft_model(base_model, peft_config)

    # 手动把 adapter 权重加载到 CPU（显式指定 device="cpu"，不经过 PEFT 内部路径）
    adapter_weights_path = os.path.join(SFT_MODEL_PATH, "adapter_model.safetensors")
    adapter_weights = st_load_file(adapter_weights_path, device="cpu")
    set_peft_model_state_dict(model, adapter_weights)
    del adapter_weights, base_model

    # Merge LoRA → 获得完整模型，再转 bfloat16
    model = model.merge_and_unload()
    model = model.to(torch.bfloat16)
    print("  ✓ SFT adapter merged on CPU.")

    # 3. 为 GRPO 重新添加 LoRA 层
    print("[3/4] Applying GRPO LoRA...")
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = 32,
        lora_alpha     = 64,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    # 4. 构建数据集
    dataset = load_grpo_dataset(DATA_PATH)

    # 5. GRPO 训练配置（纯 TRL，无 Unsloth/vLLM）
    print("[4/4] Starting GRPO training...")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = GRPOConfig(
        output_dir                  = OUTPUT_DIR,
        # 生成参数
        num_generations             = 4,           # 每条 prompt 采样 4 个候选（需满足 generation_batch_size % num_generations == 0）
        max_prompt_length           = MAX_PROMPT_LEN,
        max_completion_length       = MAX_COMPLETION_LEN,
        temperature                 = 1.0,
        top_p                       = 1.0,
        top_k                       = None,        # None = 禁用 top-k（transformers 不接受 -1）
        min_p                       = 0.1,
        # 训练超参
        learning_rate               = 5e-6,
        adam_beta1                  = 0.9,
        adam_beta2                  = 0.99,
        weight_decay                = 0.1,
        warmup_ratio                = 0.1,
        lr_scheduler_type           = "cosine",
        optim                       = "adamw_torch",  # trajlla 无 bitsandbytes adamw_8bit
        bf16                        = use_bf16,
        gradient_checkpointing      = True,
        logging_steps               = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        max_steps                   = 500,
        save_steps                  = 100,
        save_total_limit            = 3,
        report_to                   = "tensorboard",
        logging_dir                 = os.path.join(OUTPUT_DIR, "tensorboard_logs"),
        seed                        = 3407,
        remove_unused_columns       = False,   # 保留 ground_truth_schedule 列
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = [
            reward_format,               # 格式奖励  [0, 1]
            reward_schedule_similarity,  # GT 相似度 [0, 1]
            reward_constraints,          # 结构约束  [0, 1]
        ],
        args             = training_args,
        train_dataset    = dataset,
    )

    trainer.train()

    # 保存最终 LoRA adapter
    final_path = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n[Done] GRPO 训练完成，LoRA 已保存至: {final_path}")


if __name__ == "__main__":
    main()
