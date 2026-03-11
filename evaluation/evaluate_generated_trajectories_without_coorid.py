import json
import numpy as np
from datetime import datetime
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_example import evaluation

ACTIVITY_NAME_CODE_MAPPING = {
    'home': 1, 'work': 2, 'education': 3, 'shopping': 4, 'service': 5,
    'medical': 6, 'dine_out': 7, 'socialize': 8, 'exercise': 9, 'dropoff_pickup': 10,
}
TIMESTEP_MINUTES = 15
N_TIMESTEPS = 96

def time_to_minutes(time_str):
    if time_str == "24:00":
        return 1440
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])

def schedule_to_96_timesteps(schedule):
    timesteps = np.zeros(N_TIMESTEPS, dtype=int)
    for slot_idx in range(N_TIMESTEPS):
        slot_start = slot_idx * TIMESTEP_MINUTES
        slot_end = (slot_idx + 1) * TIMESTEP_MINUTES
        activity_durations = {}
        for segment in schedule:
            activity_name = segment['activity']
            seg_start = time_to_minutes(segment['start_time'])
            seg_end = time_to_minutes(segment['end_time'])
            overlap_start = max(slot_start, seg_start)
            overlap_end = min(slot_end, seg_end)
            if overlap_end > overlap_start:
                if activity_name not in activity_durations:
                    activity_durations[activity_name] = 0
                activity_durations[activity_name] += overlap_end - overlap_start
        if activity_durations:
            dominant_activity = max(activity_durations, key=activity_durations.get)
            timesteps[slot_idx] = ACTIVITY_NAME_CODE_MAPPING.get(dominant_activity, 0)
    return timesteps

def get_activity_name(code):
    code_to_name = {v: k for k, v in ACTIVITY_NAME_CODE_MAPPING.items()}
    return code_to_name.get(code, 'unknown')

def format_timesteps(timesteps):
    lines = []
    current_activity = None
    start_step = 0
    for i in range(len(timesteps)):
        if timesteps[i] != current_activity:
            if current_activity is not None:
                start_time = f"{start_step//4:02d}:{(start_step%4)*15:02d}"
                end_time = f"{i//4:02d}:{(i%4)*15:02d}"
                lines.append(f"{start_time}-{end_time}: {get_activity_name(current_activity)}")
            current_activity = timesteps[i]
            start_step = i
    if current_activity is not None:
        start_time = f"{start_step//4:02d}:{(start_step%4)*15:02d}"
        lines.append(f"{start_time}-24:00: {get_activity_name(current_activity)}")
    return "; ".join(lines)

def save_results_to_csv(results, generated_file, original_file, num_users, csv_path):
    """
    将评估结果追加保存到CSV文件中，并添加轨迹类型列
    
    Args:
        results: 评估结果字典
        generated_file: 生成的轨迹文件路径
        original_file: 原始轨迹文件路径
        num_users: 评估的用户数
        csv_path: CSV文件保存路径
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # 获取文件名（不含路径）
    gen_filename = os.path.basename(generated_file)
    
    # 判断轨迹类型
    if 'Trajectory_Generation_multi_agent' in generated_file:
        trajectory_type = 'Multi_agent'
    elif 'Trajectory_Generation_Household' in generated_file:
        trajectory_type = 'Household'
    else:
        trajectory_type = 'Personal'
    
    orig_filename = os.path.basename(original_file)
    
    # 构建记录行 - type放在第一列
    record = {
        'type': trajectory_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'generated_file': gen_filename,
        'original_file': orig_filename,
        'num_users': num_users,
        'accuracy': results.get('accuracy', np.nan),
        'f1_score': results.get('f1-score', np.nan),
        'edit_dist': results.get('edit_dist', np.nan),
        'bleu_score': results.get('bleu_score', np.nan),
        'data_jsd': results.get('data_jsd', np.nan),
        'macro_int': results.get('macro_int', np.nan),
        'micro_int': results.get('micro_int', np.nan),
        'act_type': results.get('act_type', np.nan),
        'uni_act_type': results.get('uni_act_type', np.nan),
        'traj_len': results.get('traj_len', np.nan),
        'macro_hour': results.get('macro_hour', np.nan),
        'micro_hour': results.get('micro_hour', np.nan)
    }
    
    # 检查CSV文件是否存在
    if os.path.exists(csv_path):
        # 追加到现有CSV
        df_existing = pd.read_csv(csv_path)
        df_new = pd.DataFrame([record])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        # 创建新CSV
        df_combined = pd.DataFrame([record])
    
    # 确保列的顺序：type 在第一列
    columns_order = ['type', 'timestamp', 'generated_file', 'original_file', 'num_users',
                     'accuracy', 'f1_score', 'edit_dist', 'bleu_score', 'data_jsd',
                     'macro_int', 'micro_int', 'act_type', 'uni_act_type', 'traj_len',
                     'macro_hour', 'micro_hour']
    
    # 只保留存在的列
    existing_columns = [col for col in columns_order if col in df_combined.columns]
    df_combined = df_combined[existing_columns]
    
    # 保存CSV
    df_combined.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✓ Results saved to CSV: {csv_path}")
    print(f"  Trajectory Type: {trajectory_type}")
    print(f"  Total experiments recorded: {len(df_combined)}")


def evaluate_trajectories(generated_file, original_file, output_dir='evaluation_results_9', save_csv=True):
    with open(generated_file, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    original_dict = {item['user_id']: item['schedule'] for item in original_data if 'user_id' in item}
    
    matched_users = []
    gen_sequences = []
    tar_sequences = []
    
    for gen_item in generated_data:
        user_id = gen_item.get('user_id')
        if user_id in original_dict:
            gen_seq = schedule_to_96_timesteps(gen_item.get('schedule', []))
            tar_seq = schedule_to_96_timesteps(original_dict[user_id])
            gen_sequences.append(gen_seq)
            tar_sequences.append(tar_seq)
            matched_users.append(user_id)
    
    if not matched_users:
        return None
    
    gen_seq_array = np.array(gen_sequences)
    tar_seq_array = np.array(tar_sequences)
    results = evaluation(gen_seq_array, tar_seq_array, N_TIMESTEPS)
    
    # 保存结果到CSV
    if save_csv:
        csv_path = os.path.join(output_dir, 'experiment_results_9.csv')
        save_results_to_csv(results, generated_file, original_file, len(matched_users), csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'evaluation_{timestamp}.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Trajectory Evaluation Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Generated File: {generated_file}\n")
        f.write(f"Original File: {original_file}\n")
        f.write(f"Users: {len(matched_users)}\n")
        f.write(f"Timesteps: {N_TIMESTEPS} ({TIMESTEP_MINUTES}min each)\n\n")
        
        f.write("Metrics:\n")
        f.write("-"*80 + "\n")
        for metric, value in results.items():
            f.write(f"{metric:20s}: {value:.6f}\n")
        f.write("\n" + "="*80 + "\n\n")
        
        f.write("User Trajectories:\n")
        f.write("="*80 + "\n\n")
        for idx, user_id in enumerate(matched_users):
            gen_seq = gen_sequences[idx]
            tar_seq = tar_sequences[idx]
            matches = np.sum(gen_seq == tar_seq)
            match_rate = matches / len(gen_seq)
            
            f.write(f"User {idx+1}: {user_id}\n")
            f.write(f"Match Rate: {match_rate:.2%}\n")
            f.write(f"Generated: {format_timesteps(gen_seq)}\n")
            f.write(f"Original:  {format_timesteps(tar_seq)}\n")
            f.write("-"*80 + "\n")
    
    print(f"Evaluation complete. Results saved to: {output_file}")
    return results

if __name__ == "__main__":
    GENERATED_FILE = r"E:\mayue\FinalTraj\Trajectory_Generation_multi_agent\output_trajectories\all_trajectories_20251215_183240.json"
    ORIGINAL_FILE = r"E:\mayue\FinalTraj\Oklahoma\processed_data\all_user_schedules.json"
    OUTPUT_DIR = r"E:\mayue\FinalTraj\evaluation\evaluation_results_finetune"
    
    evaluate_trajectories(GENERATED_FILE, ORIGINAL_FILE, OUTPUT_DIR, save_csv=True)
