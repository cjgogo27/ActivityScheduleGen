"""
Stage 3 v2: GRPO Training — LLaMA-3.1-8B-Instruct
基于 stage2 v2 SFT (epoch20, LoRA r=16) 的结果做 GRPO
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
BASE_MODEL_PATH = "/data/alice/cjtest/FinalTraj_arr/finetune/models/Llama-3.1-8B-Instruct/LLM-Research/Meta-Llama-3___1-8B-Instruct"
SFT_MODEL_PATH  = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage2_sft_llama_v2_epoch20/final_model"
DATA_PATH       = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage1_training_data_3/metadata.jsonl"
OUTPUT_DIR      = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage3_grpo_llama_v2_output"
MAX_PROMPT_LEN     = 2048
MAX_COMPLETION_LEN = 2048

# ─── Activity 编码 ────────────────────────────────────────────────────────────
ACTIVITY_NAME_CODE_MAPPING = {
    'home': 1, 'work': 2, 'education': 3, 'shopping': 4, 'service': 5,
    'medical': 6, 'dine_out': 7, 'socialize': 8, 'exercise': 9, 'dropoff_pickup': 10,
}
TIMESTEP_MINUTES = 15
N_TIMESTEPS = 96


def time_to_minutes(time_str: str) -> int:
    try:
        s = str(time_str).strip()
        if not s:
            return 0
        if s in ("24:00", "23:59"):
            return 1440
        # 支持 "HH:MM"（冒号）和 "HH.MM"（模型有时生成小数点）两种格式
        for sep in (":", "."):
            if sep in s:
                parts = s.split(sep, 1)
                h = int(float(parts[0]))
                m = int(float(parts[1]))
                return min(h * 60 + m, 1440)
        # 纯数字：当作整点小时数
        return min(int(float(s)) * 60, 1440)
    except Exception:
        return 0


def schedule_to_96_timesteps(schedule: list) -> np.ndarray:
    timesteps = np.zeros(N_TIMESTEPS, dtype=int)
    if not schedule:
        return timesteps
    for slot_idx in range(N_TIMESTEPS):
        slot_start = slot_idx * TIMESTEP_MINUTES
        slot_end   = (slot_idx + 1) * TIMESTEP_MINUTES
        activity_durations: dict = {}
        for seg in schedule:
            act_name  = seg.get("activity", "home")
            seg_start = time_to_minutes(seg.get("start_time", "00:00"))
            seg_end   = time_to_minutes(seg.get("end_time",   "24:00"))
            ov_start  = max(slot_start, seg_start)
            ov_end    = min(slot_end,   seg_end)
            if ov_end > ov_start:
                activity_durations[act_name] = activity_durations.get(act_name, 0) + (ov_end - ov_start)
        if activity_durations:
            dominant = max(activity_durations, key=activity_durations.get)
            timesteps[slot_idx] = ACTIVITY_NAME_CODE_MAPPING.get(dominant, 0)
    return timesteps


_JSON_PAT = re.compile(r"\[JSON\](.*?)\[/JSON\]", re.DOTALL)

def extract_schedule_from_output(text: str):
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


def _compute_int_single(arr: np.ndarray) -> np.ndarray:
    act2int = np.zeros((11, N_TIMESTEPS), dtype=float)
    curr_act, curr_len = int(arr[0]), 1
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
    from scipy.spatial import distance as sp_dist
    try:
        from sklearn.metrics import f1_score
    except ImportError:
        f1_score = None

    gen_arr = schedule_to_96_timesteps(gen_schedule)
    gt_arr  = schedule_to_96_timesteps(gt_schedule)

    accuracy = float(np.sum(gen_arr == gt_arr)) / N_TIMESTEPS

    gen_act_p = np.array([float(np.sum(gen_arr == i)) for i in range(11)])
    gt_act_p  = np.array([float(np.sum(gt_arr  == i)) for i in range(11)])
    gen_act_p /= (gen_act_p.sum() + 1e-9)
    gt_act_p  /= (gt_act_p.sum()  + 1e-9)
    act_type_sim = 1.0 - min(float(sp_dist.jensenshannon(gen_act_p, gt_act_p)), 1.0)

    gen_act2int = _compute_int_single(gen_arr)
    gt_act2int  = _compute_int_single(gt_arr)
    gs, gs2 = np.sum(gen_act2int), np.sum(gt_act2int)
    if gs > 0 and gs2 > 0:
        macro_int_sim = 1.0 - min(
            float(sp_dist.jensenshannon(
                np.sum(gen_act2int, axis=0) / gs,
                np.sum(gt_act2int,  axis=0) / gs2
            )), 1.0)
    else:
        macro_int_sim = 0.0

    if f1_score is not None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = float(f1_score(gt_arr, gen_arr, average='macro', zero_division=0))
    else:
        f1 = accuracy

    return 0.40 * accuracy + 0.25 * act_type_sim + 0.25 * macro_int_sim + 0.10 * f1


# ═══════════════════════════════ Reward Functions ═══════════════════════════════

def reward_format(completions, prompts=None, ground_truth_schedule=None, **kwargs) -> list:
    scores = []
    for completion in completions:
        resp       = completion[0]["content"] if isinstance(completion, list) else completion
        has_thought = bool(re.search(r"\[THOUGHT\].*?\[/THOUGHT\]", resp, re.DOTALL))
        has_json    = bool(re.search(r"\[JSON\].*?\[/JSON\]",    resp, re.DOTALL))
        scores.append(1.0 if (has_thought and has_json) else 0.5 if (has_thought or has_json) else 0.0)
    return scores


def reward_schedule_similarity(completions, prompts=None, ground_truth_schedule=None, **kwargs) -> list:
    if ground_truth_schedule is None:
        return [0.0] * len(completions)
    scores = []
    for completion, gt_raw in zip(completions, ground_truth_schedule):
        resp         = completion[0]["content"] if isinstance(completion, list) else completion
        gen_schedule = extract_schedule_from_output(resp)
        if gen_schedule is None:
            scores.append(0.0); continue
        gt_schedule = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw
        scores.append(compute_schedule_similarity(gen_schedule, gt_schedule))
    return scores


def reward_constraints(completions, prompts=None, ground_truth_schedule=None, **kwargs) -> list:
    scores = []
    for completion in completions:
        resp     = completion[0]["content"] if isinstance(completion, list) else completion
        schedule = extract_schedule_from_output(resp)
        if not schedule:
            scores.append(0.0); continue
        try:
            segs = sorted(
                [(time_to_minutes(s.get("start_time","00:00")),
                  time_to_minutes(s.get("end_time","24:00"))) for s in schedule]
            )
            covers_day   = (segs[0][0] == 0 and segs[-1][1] == 1440)
            continuous   = all(segs[i][1] == segs[i+1][0] for i in range(len(segs)-1))
            durations_ok = all(5 <= (e-s) <= 720 for s, e in segs)
            scores.append((float(covers_day) + float(continuous) + float(durations_ok)) / 3.0)
        except Exception:
            scores.append(0.0)
    return scores


# ─── Dataset ──────────────────────────────────────────────────────────────────
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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Generate a daily schedule for this person:\n{profile_str}"},
    ]


def load_grpo_dataset(data_path: str) -> Dataset:
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append({
                "prompt":                 build_prompt(obj["user_profile"]),
                "ground_truth_schedule":  obj["ground_truth_schedule"],
            })
    print(f"[Dataset] Loaded {len(records)} samples from {data_path}")
    return Dataset.from_list(records)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType, PeftConfig, set_peft_model_state_dict
    from safetensors.torch import load_file as st_load_file
    from trl import GRPOConfig, GRPOTrainer

    print(f"[1/4] Loading tokenizer from: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[2/4] Loading base model + merging SFT v2 LoRA (CPU)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float32, device_map={"": "cpu"}
    )
    peft_config     = PeftConfig.from_pretrained(SFT_MODEL_PATH)
    model           = get_peft_model(base_model, peft_config)
    adapter_weights = st_load_file(os.path.join(SFT_MODEL_PATH, "adapter_model.safetensors"), device="cpu")
    set_peft_model_state_dict(model, adapter_weights)
    del adapter_weights, base_model
    model = model.merge_and_unload().to(torch.bfloat16)
    print("  ✓ SFT v2 adapter merged on CPU.")

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

    dataset = load_grpo_dataset(DATA_PATH)

    print("[4/4] Starting GRPO v2 training...")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = GRPOConfig(
        output_dir                  = OUTPUT_DIR,
        num_generations             = 4,
        max_prompt_length           = MAX_PROMPT_LEN,
        max_completion_length       = MAX_COMPLETION_LEN,
        temperature                 = 0.8,
        top_p                       = 1.0,
        top_k                       = 0,
        min_p                       = 0.1,
        beta                        = 0.04,
        learning_rate               = 5e-6,
        adam_beta1                  = 0.9,
        adam_beta2                  = 0.99,
        weight_decay                = 0.1,
        warmup_ratio                = 0.1,
        lr_scheduler_type           = "cosine",
        optim                       = "adamw_torch",
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
        remove_unused_columns       = False,
    )

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = [reward_format, reward_schedule_similarity, reward_constraints],
        args             = training_args,
        train_dataset    = dataset,
    )

    trainer.train()

    final_path = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n[Done] GRPO v2 训练完成，已保存至: {final_path}")


if __name__ == "__main__":
    main()
