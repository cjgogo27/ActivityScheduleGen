#!/usr/bin/env python3
"""
为所有41个用户生成竖向条形图可视化对比
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Font settings removed - using default English fonts
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# 文件路径
ORIGINAL_TRAJECTORIES_FILE = r"E:\mayue\FinalTraj\California\processed_data\all_user_schedules.json"
GENERATED_TRAJECTORIES_FILE = r"E:\mayue\FinalTraj\Trajectory_Generation_multi_agent\output_trajectories\all_trajectories_20251215_163347.json"
OUTPUT_DIR = r"E:\mayue\FinalTraj\evaluation\visualization_analysis_finetune"

TIMESTEP_MINUTES = 15
N_TIMESTEPS = 96

# 活动类型到颜色的映射
ACTIVITY_COLORS = {
    'home': '#90EE90',       # 浅绿色
    'work': '#FFB6C1',       # 浅粉色
    'education': '#87CEEB',  # 天蓝色
    'shopping': '#FFD700',   # 金色
    'service': '#DDA0DD',    # 梅红色
    'medical': '#FF6347',    # 番茄红
    'dine_out': '#FFA500',   # 橙色
    'socialize': '#9370DB',  # 中紫色
    'exercise': '#32CD32',   # 酸橙绿
    'dropoff_pickup': '#FF69B4'  # 热粉色
}

ACTIVITY_NAME_CODE_MAPPING = {
    'home': 1, 'work': 2, 'education': 3, 'shopping': 4, 'service': 5,
    'medical': 6, 'dine_out': 7, 'socialize': 8, 'exercise': 9, 'dropoff_pickup': 10,
}


def time_to_minutes(time_str):
    """将时间字符串转换为分钟数"""
    if time_str == "24:00":
        return 1440
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])


def schedule_to_96_timesteps(schedule):
    """将schedule转换为96个时间步"""
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


def calculate_accuracy(gen_seq, tar_seq):
    """计算准确率"""
    return np.sum(gen_seq == tar_seq) / len(gen_seq)


def create_all_users_vertical_comparison(original_dict, generated_trajectories, output_dir):
    """创建所有41个用户的竖向对比图"""
    
    # 匹配用户并计算准确率
    matched_users = []
    for gen_item in generated_trajectories:
        user_id = gen_item['user_id']
        if user_id in original_dict:
            gen_seq = schedule_to_96_timesteps(gen_item['schedule'])
            tar_seq = schedule_to_96_timesteps(original_dict[user_id])
            accuracy = calculate_accuracy(gen_seq, tar_seq)
            
            matched_users.append({
                'user_id': user_id,
                'accuracy': accuracy,
                'gen_schedule': gen_item['schedule'],
                'orig_schedule': original_dict[user_id]
            })
    
    # 按准确率排序 (从低到高)
    matched_users.sort(key=lambda x: x['accuracy'])
    
    num_users = len(matched_users)
    print(f"Matched {num_users} users")
    
    # 创建大图 - 两个子图并排
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14))
    
    # Set titles
    fig.suptitle(f'Trajectory Comparison for All Users (n={num_users}, sorted by accuracy from low to high)', fontsize=22, fontweight='bold', y=0.995)
    ax1.set_title('True (Original)', fontsize=20, fontweight='bold', pad=10)
    ax2.set_title('Predicted (Generated)', fontsize=20, fontweight='bold', pad=10)
    
    # 处理每个用户
    for idx, user_data in enumerate(matched_users):
        user_id = user_data['user_id']
        accuracy = user_data['accuracy']
        
        orig_schedule = user_data['orig_schedule']
        gen_schedule = user_data['gen_schedule']
        
        # 绘制原始轨迹 (True)
        for seg in orig_schedule:
            start = time_to_minutes(seg['start_time'])
            end = time_to_minutes(seg['end_time'])
            activity = seg['activity']
            color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
            
            start_hour = start / 60
            end_hour = end / 60
            height = end_hour - start_hour
            
            ax1.bar(idx + 1, height, bottom=start_hour, width=0.9,
                   color=color, edgecolor='black', linewidth=0.3)
        
        # 绘制生成轨迹 (Predicted)
        for seg in gen_schedule:
            start = time_to_minutes(seg['start_time'])
            end = time_to_minutes(seg['end_time'])
            activity = seg['activity']
            color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
            
            start_hour = start / 60
            end_hour = end / 60
            height = end_hour - start_hour
            
            ax2.bar(idx + 1, height, bottom=start_hour, width=0.9,
                   color=color, edgecolor='black', linewidth=0.3)
    
    # 设置ax1
    ax1.set_ylim(0, 24)
    ax1.set_xlim(0, num_users + 1)
    ax1.set_ylabel('Time of Day', fontsize=18, fontweight='bold')
    ax1.set_xlabel('User (sorted by accuracy)', fontsize=18, fontweight='bold')
    
    # 设置x轴刻度 - 每5个用户标一次
    x_ticks = list(range(1, num_users + 1, 5))
    if num_users not in x_ticks:
        x_ticks.append(num_users)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([str(i) for i in x_ticks], fontsize=12)
    
    ax1.set_yticks([0, 4, 8, 12, 16, 20, 24])
    ax1.set_yticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=14)
    ax1.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax1.invert_yaxis()
    
    # 设置ax2
    ax2.set_ylim(0, 24)
    ax2.set_xlim(0, num_users + 1)
    ax2.set_ylabel('Time of Day', fontsize=18, fontweight='bold')
    ax2.set_xlabel('User (sorted by accuracy)', fontsize=18, fontweight='bold')
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([str(i) for i in x_ticks], fontsize=12)
    ax2.set_yticks([0, 4, 8, 12, 16, 20, 24])
    ax2.set_yticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=14)
    ax2.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    ax2.invert_yaxis()
    
    # 添加准确率范围标注
    low_acc_count = sum(1 for u in matched_users if u['accuracy'] < 0.5)
    mid_acc_count = sum(1 for u in matched_users if 0.5 <= u['accuracy'] < 0.7)
    high_acc_count = sum(1 for u in matched_users if u['accuracy'] >= 0.7)
    
    avg_accuracy = np.mean([u['accuracy'] for u in matched_users])
    
    # 在图上添加准确率区间分隔线
    if low_acc_count > 0:
        ax1.axvline(x=low_acc_count + 0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.axvline(x=low_acc_count + 0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    if low_acc_count + mid_acc_count < num_users:
        ax1.axvline(x=low_acc_count + mid_acc_count + 0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5)
        ax2.axvline(x=low_acc_count + mid_acc_count + 0.5, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    
    # 添加图例
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=activity)
                      for activity, color in ACTIVITY_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, fontsize=13,
              bbox_to_anchor=(0.5, 0.97), frameon=True, shadow=True)
    
    # Add statistics
    stats_text = f"Average Accuracy: {avg_accuracy:.2%} | Low (<50%): {low_acc_count} | Medium (50-70%): {mid_acc_count} | High (≥70%): {high_acc_count}"
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=14, style='italic', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    output_file = os.path.join(output_dir, 'all_41_users_vertical_comparison.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Image saved: {output_file}")
    print(f"\nStatistics:")
    print(f"  Average Accuracy: {avg_accuracy:.2%}")
    print(f"  Low Accuracy (<50%): {low_acc_count} users")
    print(f"  Medium Accuracy (50-70%): {mid_acc_count} users")
    print(f"  High Accuracy (≥70%): {high_acc_count} users")
    
    return output_file


def main():
    """主函数"""
    print("="*80)
    print("Generating Vertical Bar Chart Comparison for All 41 Users")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    with open(ORIGINAL_TRAJECTORIES_FILE, 'r', encoding='utf-8') as f:
        original_trajectories = json.load(f)
    original_dict = {t['user_id']: t['schedule'] for t in original_trajectories}
    print(f"  Original trajectories: {len(original_dict)} users")
    
    with open(GENERATED_TRAJECTORIES_FILE, 'r', encoding='utf-8') as f:
        generated_trajectories = json.load(f)
    print(f"  Generated trajectories: {len(generated_trajectories)} users")
    
    # Generate comparison chart
    print("\nGenerating vertical comparison chart...")
    output_file = create_all_users_vertical_comparison(original_dict, generated_trajectories, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
