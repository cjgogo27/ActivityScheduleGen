"""
LSTPM 基线推理脚本
从 metadata.jsonl 读取用户信息，使用预训练的 LSTPM 模型生成日程
输出格式与 run_inference.py 兼容，并自动调用评估函数

用法:
  python run_baseline_lstpm.py --data eval_data/california_50.jsonl
  python run_baseline_lstpm.py --data eval_data/arizona_50.jsonl
"""

import sys, os, json, argparse
from datetime import datetime
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
LSTPM_DIR   = "/data/alice/cjtest/FinalTraj/Trajectory_Generation_tradition2"
EVAL_DIR    = "/data/alice/cjtest/FinalTraj_arr/evaluation"
RESULTS_DIR = "/data/alice/cjtest/FinalTraj_arr/Trajectory_Generation_multi_agent/results"

sys.path.insert(0, LSTPM_DIR)
sys.path.insert(0, EVAL_DIR)

from lstpm_model import create_lstpm_model
from lstpm_data_loader import (
    InferenceDataset, collate_inference_fn,
    ACTIVITY_TYPES, IDX_TO_ACTIVITY
)


def hour_to_time_string(hour) -> str:
    h = int(hour) % 24
    return f"{h:02d}:00"


def generate_schedule_from_sequence(activities, times):
    schedule = []
    for i in range(len(activities) - 1):
        idx  = activities[i].item() if hasattr(activities[i], "item") else int(activities[i])
        name = IDX_TO_ACTIVITY.get(idx, "home")
        sh   = times[i].item() if hasattr(times[i], "item") else float(times[i])
        eh   = times[i+1].item() if hasattr(times[i+1], "item") else float(times[i+1])
        if eh <= sh:
            eh = sh + 1
        schedule.append({
            "activity":   name,
            "start_time": hour_to_time_string(sh),
            "end_time":   hour_to_time_string(eh),
        })
    last_idx  = activities[-1].item() if hasattr(activities[-1], "item") else int(activities[-1])
    last_name = IDX_TO_ACTIVITY.get(last_idx, "home")
    last_h    = times[-1].item() if hasattr(times[-1], "item") else float(times[-1])
    schedule.append({"activity": last_name, "start_time": hour_to_time_string(last_h), "end_time": "24:00"})
    return schedule


def post_process_schedule(schedule):
    if not schedule:
        return [{"activity": "home", "start_time": "00:00", "end_time": "24:00"}]
    merged = []
    cur = schedule[0].copy()
    for seg in schedule[1:]:
        if seg["activity"] == cur["activity"]:
            cur["end_time"] = seg["end_time"]
        else:
            merged.append(cur)
            cur = seg.copy()
    merged.append(cur)
    if merged[0]["activity"] != "home":
        merged.insert(0, {"activity": "home", "start_time": "00:00", "end_time": merged[0]["start_time"]})
    if merged[-1]["activity"] != "home":
        merged.append({"activity": "home", "start_time": merged[-1]["end_time"], "end_time": "24:00"})
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True,  help="path to {state}_50.jsonl")
    parser.add_argument("--gpu",         default="4",    help="CUDA device id")
    parser.add_argument("--batch",       type=int, default=16)
    parser.add_argument("--max_len",     type=int, default=15)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 加载 user_id 映射（训练时建立，推理时使用 0 作为未知用户的回退）────────
    mapping_path = os.path.join(LSTPM_DIR, "user_id_mapping.json")
    with open(mapping_path) as f:
        user_id_mapping = json.load(f)
    num_users = len(user_id_mapping)
    print(f"Loaded user_id mapping: {num_users} training users")

    # ── 创建并加载模型 ────────────────────────────────────────────────────────
    model = create_lstpm_model(num_users=num_users, activity_list=ACTIVITY_TYPES, device=device)
    ckpt_path = os.path.join(LSTPM_DIR, "checkpoints", "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")

    # ── 读取 metadata.jsonl ───────────────────────────────────────────────────
    samples = []
    with open(args.data) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"Loaded {len(samples)} samples from {args.data}")

    persons_list = []
    for s in samples:
        p = dict(s["user_profile"])
        p["user_id"] = s["user_id"]
        persons_list.append(p)

    ground_truth = [{"user_id": s["user_id"], "schedule": s["ground_truth_schedule"]} for s in samples]

    # ── 推理 ──────────────────────────────────────────────────────────────────
    dataset = InferenceDataset(persons_list, user_id_mapping, seq_len=3)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                         collate_fn=collate_inference_fn)

    all_results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="LSTPM inference"):
            user_ids  = batch["user_ids"]
            user_idxs = batch["user_idx"].to(device)
            start_act = batch["start_activities"].to(device)
            start_t   = batch["start_times"].to(device)

            acts_seq, times_seq = model.generate_sequence(
                user_idxs, start_act, start_t,
                max_len=args.max_len, temperature=args.temperature,
            )

            for i, uid in enumerate(user_ids):
                raw = generate_schedule_from_sequence(
                    acts_seq[i].cpu().tolist(),
                    times_seq[i].cpu().tolist(),
                )
                all_results.append({"user_id": uid, "schedule": post_process_schedule(raw)})

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    n        = len(all_results)
    data_tag = os.path.splitext(os.path.basename(args.data))[0]

    gen_path = os.path.join(RESULTS_DIR, f"generated_lstpm_{n}samples_{ts}.json")
    gt_path  = os.path.join(RESULTS_DIR, f"ground_truth_{n}samples_{ts}.json")

    with open(gen_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Generated {n} trajectories → {gen_path}")

    # ── 评估 ──────────────────────────────────────────────────────────────────
    try:
        import evaluate_generated_trajectories
        eval_dir = os.path.join(RESULTS_DIR, f"eval_lstpm_{data_tag}_{ts}")
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
