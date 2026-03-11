"""
评估脚本：对比 Base 模型 vs SFT 模型的轨迹生成质量

用法：
  # 测试 base 模型（全部 200 条）
  python eval_base_sft.py --model base

  # 测试 SFT 模型（前 50 条）
  python eval_base_sft.py --model sft --num_samples 50

  # 指定输出文件
  python eval_base_sft.py --model sft --num_samples 50 --output results_sft.json
"""

import os, re, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

# ─── 路径配置 ────────────────────────────────────────────────────────────────
BASE_MODEL_PATH = "/data/alice/cjtest/FinalTraj_arr/finetune/models/Qwen3-8B/Qwen/Qwen3-8B"
SFT_MODEL_PATH  = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage2_sft_output_epoch10/final_model"
DATA_PATH       = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/stage1_training_data_3/metadata.jsonl"
OUTPUT_DIR      = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/evaluation"

# ─── 与 train_grpo.py 完全相同的常量 ─────────────────────────────────────────
ACTIVITY_NAME_CODE_MAPPING = {
    'home': 1, 'work': 2, 'education': 3, 'shopping': 4, 'service': 5,
    'medical': 6, 'dine_out': 7, 'socialize': 8, 'exercise': 9, 'dropoff_pickup': 10,
}
TIMESTEP_MINUTES = 15
N_TIMESTEPS = 96

SYSTEM_PROMPT = """You are a daily activity schedule generator. Your task is to generate a realistic daily schedule for a person based on their profile.

Output format:
[THOUGHT]
Brief reasoning about the person's schedule patterns.
[/THOUGHT]
[JSON]
[{"activity": "home", "start_time": "00:00", "end_time": "07:00"}, ...]
[/JSON]

The schedule must start at 00:00 and end at 24:00, covering the full day without gaps or overlaps."""


# ─── 工具函数（与 train_grpo.py 完全一致）────────────────────────────────────
def time_to_minutes(time_str: str) -> int:
    if time_str in ("24:00", "23:59"):
        return 1440
    parts = time_str.strip().split(":")
    return int(parts[0]) * 60 + int(parts[1])


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
            overlap   = max(0, min(seg_end, slot_end) - max(seg_start, slot_start))
            if overlap > 0:
                activity_durations[act_name] = activity_durations.get(act_name, 0) + overlap
        if activity_durations:
            dominant = max(activity_durations, key=activity_durations.get)
            timesteps[slot_idx] = ACTIVITY_NAME_CODE_MAPPING.get(dominant, 0)
    return timesteps


def _compute_int_single(arr: np.ndarray) -> np.ndarray:
    n_acts = len(ACTIVITY_NAME_CODE_MAPPING) + 1  # 0-10
    result = np.zeros((n_acts, N_TIMESTEPS), dtype=float)
    for t in range(N_TIMESTEPS):
        act = int(arr[t])
        if act < n_acts:
            result[act, t] = 1.0
    return result


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    from scipy.special import rel_entr
    p = p + 1e-10;  p = p / p.sum()
    q = q + 1e-10;  q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m)))


def compute_schedule_similarity(gen_schedule: list, gt_schedule: list) -> dict:
    """返回详细 metrics dict"""
    try:
        gen_arr = schedule_to_96_timesteps(gen_schedule)
        gt_arr  = schedule_to_96_timesteps(gt_schedule)

        # accuracy（slot-level）
        accuracy = float(np.mean(gen_arr == gt_arr))

        # act_type_sim（1 - JSD of activity frequency）
        n_acts = len(ACTIVITY_NAME_CODE_MAPPING) + 1
        gen_freq = np.zeros(n_acts)
        gt_freq  = np.zeros(n_acts)
        for v in gen_arr:
            gen_freq[int(v)] += 1
        for v in gt_arr:
            gt_freq[int(v)] += 1
        act_type_sim = max(0.0, 1.0 - _jsd(gen_freq, gt_freq))

        # macro_int_sim（1 - mean JSD over activity dimension）
        gen_int = _compute_int_single(gen_arr)
        gt_int  = _compute_int_single(gt_arr)
        jsd_per_act = [_jsd(gen_int[a], gt_int[a]) for a in range(n_acts)]
        macro_int_sim = max(0.0, 1.0 - float(np.mean(jsd_per_act)))

        # macro F1
        from sklearn.metrics import f1_score
        f1 = float(f1_score(gt_arr, gen_arr, average="macro", zero_division=0))

        # 综合得分（与 reward 函数一致）
        score = 0.40 * accuracy + 0.25 * act_type_sim + 0.25 * macro_int_sim + 0.10 * f1

        return {
            "score":          round(score, 4),
            "accuracy":       round(accuracy, 4),
            "act_type_sim":   round(act_type_sim, 4),
            "macro_int_sim":  round(macro_int_sim, 4),
            "f1":             round(f1, 4),
        }
    except Exception as e:
        return {"score": 0.0, "accuracy": 0.0, "act_type_sim": 0.0,
                "macro_int_sim": 0.0, "f1": 0.0, "error": str(e)}


def extract_schedule(text: str):
    """从模型输出中提取 [JSON]...[/JSON] 里的 schedule"""
    m = re.search(r'\[JSON\](.*?)\[/JSON\]', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except Exception:
        return None


# ─── 模型加载 ─────────────────────────────────────────────────────────────────
def load_base_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[Base] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Base] 加载模型（device_map=auto, bf16）...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map        = "auto",
        torch_dtype       = torch.bfloat16,
        trust_remote_code = True,
    )
    model.eval()
    print("[Base] 模型加载完成 ✓")
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
        device_map        = "auto",
        torch_dtype       = torch.bfloat16,
        trust_remote_code = True,
    )

    print(f"[SFT] 加载 LoRA adapter: {SFT_MODEL_PATH} ...")
    model = PeftModel.from_pretrained(base_model, SFT_MODEL_PATH)
    model.eval()
    print("[SFT] 模型加载完成 ✓")
    return tokenizer, model


# ─── 推理 ─────────────────────────────────────────────────────────────────────
def build_prompt_messages(user_profile: dict) -> list:
    profile_str = json.dumps(user_profile, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Generate a daily schedule for this person:\n{profile_str}"},
    ]


def run_inference(tokenizer, model, messages: list,
                  max_new_tokens: int = 2048,
                  temperature: float = 0.7) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens  = max_new_tokens,
            temperature     = temperature,
            top_p           = 0.9,
            do_sample       = True,
            pad_token_id    = tokenizer.pad_token_id,
            eos_token_id    = tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str, required=True,
                        choices=["base", "sft"], help="base 或 sft")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="测试样本数，默认全部")
    parser.add_argument("--output",      type=str, default=None,
                        help="结果输出 JSON 文件路径")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    # 加载模型
    if args.model == "base":
        tokenizer, model = load_base_model()
    else:
        tokenizer, model = load_sft_model()

    # 读取数据
    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.num_samples:
        samples = samples[:args.num_samples]
    print(f"\n共 {len(samples)} 条测试样本")

    # 逐条推理并评估
    results = []
    metric_keys = ["score", "accuracy", "act_type_sim", "macro_int_sim", "f1"]
    totals = {k: 0.0 for k in metric_keys}
    parse_fail = 0

    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating [{args.model}]")):
        messages  = build_prompt_messages(sample["user_profile"])
        gt_sched  = sample["ground_truth_schedule"]
        user_id   = sample.get("user_id", f"sample_{i}")

        try:
            response   = run_inference(tokenizer, model, messages,
                                       max_new_tokens=args.max_new_tokens)
            gen_sched  = extract_schedule(response)

            if gen_sched is None:
                parse_fail += 1
                metrics = {"score": 0.0, "accuracy": 0.0, "act_type_sim": 0.0,
                           "macro_int_sim": 0.0, "f1": 0.0, "parse_fail": True}
            else:
                metrics = compute_schedule_similarity(gen_sched, gt_sched)
                metrics["parse_fail"] = False

            for k in metric_keys:
                totals[k] += metrics.get(k, 0.0)

            results.append({
                "user_id":        user_id,
                "metrics":        metrics,
                "generated":      gen_sched,
                "ground_truth":   gt_sched,
            })

            # 每 10 条打印一次进度
            if (i + 1) % 10 == 0:
                n = i + 1
                print(f"\n  [{n}/{len(samples)}] 累计均值 | "
                      f"score={totals['score']/n:.4f}  "
                      f"acc={totals['accuracy']/n:.4f}  "
                      f"act={totals['act_type_sim']/n:.4f}  "
                      f"int={totals['macro_int_sim']/n:.4f}  "
                      f"f1={totals['f1']/n:.4f}  "
                      f"parse_fail={parse_fail}")

        except Exception as e:
            print(f"\nSample {i} ({user_id}) 出错: {e}")
            results.append({"user_id": user_id, "error": str(e)})

    # 汇总
    n = len(samples)
    summary = {
        "model":       args.model,
        "num_samples": n,
        "parse_fail":  parse_fail,
        "avg": {k: round(totals[k] / n, 4) for k in metric_keys},
    }

    print("\n" + "=" * 60)
    print(f"  模型: {args.model.upper()}    样本数: {n}    解析失败: {parse_fail}")
    print(f"  综合得分  (score)       : {summary['avg']['score']:.4f}")
    print(f"  准确率    (accuracy)    : {summary['avg']['accuracy']:.4f}")
    print(f"  活动类型  (act_type_sim): {summary['avg']['act_type_sim']:.4f}")
    print(f"  宏观强度  (macro_int)   : {summary['avg']['macro_int_sim']:.4f}")
    print(f"  Macro-F1  (f1)          : {summary['avg']['f1']:.4f}")
    print("=" * 60)

    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = args.output or os.path.join(
        OUTPUT_DIR, f"eval_{args.model}_{n}samples.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[Done] 结果已保存至: {out_path}")


if __name__ == "__main__":
    main()
