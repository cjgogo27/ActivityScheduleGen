#!/usr/bin/env python3
"""
轨迹对比可视化分析脚本
用于分析生成轨迹与真实轨迹的差异，找出具体问题
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
import os
import pandas as pd
from collections import Counter

# Font settings removed - using default English fonts
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False

# 活动映射
ACTIVITY_NAME_CODE_MAPPING = {
    'home': 1, 'work': 2, 'education': 3, 'shopping': 4, 'service': 5,
    'medical': 6, 'dine_out': 7, 'socialize': 8, 'exercise': 9, 'dropoff_pickup': 10,
}

# 活动颜色映射
ACTIVITY_COLORS = {
    'home': '#87CEEB',      # 天蓝色
    'work': '#FF6B6B',      # 红色
    'education': '#4ECDC4', # 青色
    'shopping': '#FFD93D',  # 黄色
    'service': '#95E1D3',   # 薄荷绿
    'medical': '#F38181',   # 粉红色
    'dine_out': '#FFA07A',  # 浅橙色
    'socialize': '#DDA0DD', # 紫色
    'exercise': '#98D8C8',  # 浅绿色
    'dropoff_pickup': '#F7DC6F', # 浅黄色
}


def time_to_minutes(time_str):
    """将时间字符串转换为分钟数"""
    if time_str == "24:00":
        return 1440
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])


def minutes_to_time(minutes):
    """将分钟数转换为时间字符串"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def load_trajectories(generated_file, original_file):
    """加载生成轨迹和原始轨迹"""
    with open(generated_file, 'r', encoding='utf-8') as f:
        generated_data = json.load(f)
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 构建字典
    gen_dict = {item['user_id']: item['schedule'] for item in generated_data}
    orig_dict = {item['user_id']: item['schedule'] for item in original_data if 'user_id' in item}
    
    return gen_dict, orig_dict


def visualize_single_comparison(user_id, gen_schedule, orig_schedule, save_path=None):
    """可视化单个用户的轨迹对比"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))
    
    # 绘制生成轨迹
    for segment in gen_schedule:
        activity = segment['activity']
        start = time_to_minutes(segment['start_time'])
        end = time_to_minutes(segment['end_time'])
        duration = end - start
        
        color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
        ax1.barh(0, duration, left=start, height=0.8, 
                color=color, edgecolor='black', linewidth=0.5)
        
        # 添加活动标签
        mid_point = start + duration / 2
        if duration > 60:  # 只在较长的活动段上显示文字
            ax1.text(mid_point, 0, activity, 
                    ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax1.set_xlim(0, 1440)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([0])
    ax1.set_yticklabels(['Generated Trajectory'])
    ax1.set_xlabel('Time (minutes)', fontsize=11)
    ax1.set_title(f'User {user_id} - Trajectory Comparison', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 设置x轴刻度为小时
    hour_ticks = [i * 60 for i in range(0, 25, 3)]
    ax1.set_xticks(hour_ticks)
    ax1.set_xticklabels([f"{i:02d}:00" for i in range(0, 25, 3)])
    
    # 绘制真实轨迹
    for segment in orig_schedule:
        activity = segment['activity']
        start = time_to_minutes(segment['start_time'])
        end = time_to_minutes(segment['end_time'])
        duration = end - start
        
        color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
        ax2.barh(0, duration, left=start, height=0.8, 
                color=color, edgecolor='black', linewidth=0.5)
        
        # 添加活动标签
        mid_point = start + duration / 2
        if duration > 60:
            ax2.text(mid_point, 0, activity, 
                    ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlim(0, 1440)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([0])
    ax2.set_yticklabels(['True Trajectory'])
    ax2.set_xlabel('Time (minutes)', fontsize=11)
    ax2.grid(axis='x', alpha=0.3)
    
    ax2.set_xticks(hour_ticks)
    ax2.set_xticklabels([f"{i:02d}:00" for i in range(0, 25, 3)])
    
    # 添加图例
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=activity)
                      for activity, color in ACTIVITY_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper right', ncol=2, 
              bbox_to_anchor=(0.98, 0.98), fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_problem_patterns(gen_dict, orig_dict, output_dir):
    """分析问题模式"""
    print("\n" + "="*80)
    print("Problem Pattern Analysis")
    print("="*80)
    
    problems = {
        'Work time mismatch': [],
        'Generated unnecessary work': [],
        'Missing real work': [],
        'Activity type error': [],
        'Time period completely mismatched': [],
        'Activities oversimplified': [],
        'Night shift work not recognized': [],
    }
    
    for user_id in gen_dict:
        if user_id not in orig_dict:
            continue
        
        gen_schedule = gen_dict[user_id]
        orig_schedule = orig_dict[user_id]
        
        # 提取活动统计
        gen_activities = [s['activity'] for s in gen_schedule]
        orig_activities = [s['activity'] for s in orig_schedule]
        
        gen_work_count = gen_activities.count('work')
        orig_work_count = orig_activities.count('work')
        
        # Problem 1: Generated work when there should be none
        if gen_work_count > 0 and orig_work_count == 0:
            problems['Generated unnecessary work'].append(user_id)
        
        # Problem 2: Missing real work
        if orig_work_count > 0 and gen_work_count == 0:
            problems['Missing real work'].append(user_id)
        
        # Problem 3: Work time mismatch
        if gen_work_count > 0 and orig_work_count > 0:
            gen_work_times = [(time_to_minutes(s['start_time']), time_to_minutes(s['end_time'])) 
                             for s in gen_schedule if s['activity'] == 'work']
            orig_work_times = [(time_to_minutes(s['start_time']), time_to_minutes(s['end_time'])) 
                              for s in orig_schedule if s['activity'] == 'work']
            
            # Check for night shift
            for start, end in orig_work_times:
                if start < 6*60 or end > 20*60 or start > 14*60:  # Before 6am or after 8pm, or starts in afternoon
                    problems['Night shift work not recognized'].append(user_id)
                    break
            
            # Check work time differences
            if gen_work_times and orig_work_times:
                gen_start, gen_end = gen_work_times[0]
                orig_start, orig_end = orig_work_times[0]
                
                if abs(gen_start - orig_start) > 120 or abs(gen_end - orig_end) > 120:
                    problems['Work time mismatch'].append(user_id)
        
        # Problem 4: Activity type diversity difference
        gen_unique = len(set(gen_activities))
        orig_unique = len(set(orig_activities))
        
        if gen_unique < orig_unique - 1:
            problems['Activities oversimplified'].append(user_id)
        
        # Problem 5: Activity types completely mismatched
        gen_set = set(gen_activities) - {'home'}
        orig_set = set(orig_activities) - {'home'}
        
        if gen_set and orig_set and len(gen_set & orig_set) == 0:
            problems['Activity type error'].append(user_id)
    
    # Print analysis results
    total_users = len([u for u in gen_dict if u in orig_dict])
    
    print(f"\nTotal users: {total_users}")
    print("\nProblem distribution:")
    for problem_type, users in problems.items():
        count = len(users)
        percentage = (count / total_users) * 100 if total_users > 0 else 0
        print(f"  {problem_type}: {count} users ({percentage:.1f}%)")
        if count > 0 and count <= 5:
            print(f"    Example users: {', '.join(users[:5])}")
    
    # Save problem user list
    problem_file = os.path.join(output_dir, 'problem_analysis.json')
    with open(problem_file, 'w', encoding='utf-8') as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Problem analysis saved to: {problem_file}")
    
    return problems


def visualize_activity_distribution(gen_dict, orig_dict, output_dir):
    """可视化活动分布对比"""
    gen_activities = []
    orig_activities = []
    
    for user_id in gen_dict:
        if user_id in orig_dict:
            gen_activities.extend([s['activity'] for s in gen_dict[user_id]])
            orig_activities.extend([s['activity'] for s in orig_dict[user_id]])
    
    gen_counter = Counter(gen_activities)
    orig_counter = Counter(orig_activities)
    
    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    activities = list(ACTIVITY_COLORS.keys())
    gen_counts = [gen_counter.get(act, 0) for act in activities]
    orig_counts = [orig_counter.get(act, 0) for act in activities]
    
    colors = [ACTIVITY_COLORS[act] for act in activities]
    
    ax1.bar(activities, gen_counts, color=colors, edgecolor='black')
    ax1.set_title('Generated Trajectory - Activity Distribution', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Activity Type', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(activities, orig_counts, color=colors, edgecolor='black')
    ax2.set_title('True Trajectory - Activity Distribution', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Activity Type', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'activity_distribution_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Activity distribution comparison saved to: {save_path}")
    
    # Print statistics
    print("\nActivity distribution comparison:")
    print(f"{'Activity Type':<15} {'Generated':<10} {'True':<10} {'Difference':<10}")
    print("-" * 50)
    for act in activities:
        gen_c = gen_counter.get(act, 0)
        orig_c = orig_counter.get(act, 0)
        diff = gen_c - orig_c
        diff_str = f"{diff:+d}" if diff != 0 else "0"
        print(f"{act:<15} {gen_c:<10} {orig_c:<10} {diff_str:<10}")


def visualize_work_time_distribution(gen_dict, orig_dict, output_dir):
    """可视化工作时间分布"""
    gen_work_starts = []
    gen_work_ends = []
    orig_work_starts = []
    orig_work_ends = []
    
    for user_id in gen_dict:
        if user_id in orig_dict:
            # 生成轨迹的工作时间
            for s in gen_dict[user_id]:
                if s['activity'] == 'work':
                    gen_work_starts.append(time_to_minutes(s['start_time']) / 60)
                    gen_work_ends.append(time_to_minutes(s['end_time']) / 60)
            
            # 真实轨迹的工作时间
            for s in orig_dict[user_id]:
                if s['activity'] == 'work':
                    orig_work_starts.append(time_to_minutes(s['start_time']) / 60)
                    orig_work_ends.append(time_to_minutes(s['end_time']) / 60)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Generated trajectory - work start time
    ax1.hist(gen_work_starts, bins=24, range=(0, 24), color='#FF6B6B', alpha=0.7, edgecolor='black')
    ax1.set_title('Generated Trajectory - Work Start Time Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Hour', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axvline(x=np.mean(gen_work_starts) if gen_work_starts else 0, 
                color='red', linestyle='--', linewidth=2, label=f'Average: {np.mean(gen_work_starts):.1f}h' if gen_work_starts else 'N/A')
    ax1.legend()
    
    # True trajectory - work start time
    ax2.hist(orig_work_starts, bins=24, range=(0, 24), color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax2.set_title('True Trajectory - Work Start Time Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hour', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axvline(x=np.mean(orig_work_starts) if orig_work_starts else 0, 
                color='blue', linestyle='--', linewidth=2, label=f'Average: {np.mean(orig_work_starts):.1f}h' if orig_work_starts else 'N/A')
    ax2.legend()
    
    # Generated trajectory - work end time
    ax3.hist(gen_work_ends, bins=24, range=(0, 24), color='#FF6B6B', alpha=0.7, edgecolor='black')
    ax3.set_title('Generated Trajectory - Work End Time Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Hour', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axvline(x=np.mean(gen_work_ends) if gen_work_ends else 0, 
                color='red', linestyle='--', linewidth=2, label=f'Average: {np.mean(gen_work_ends):.1f}h' if gen_work_ends else 'N/A')
    ax3.legend()
    
    # True trajectory - work end time
    ax4.hist(orig_work_ends, bins=24, range=(0, 24), color='#4ECDC4', alpha=0.7, edgecolor='black')
    ax4.set_title('True Trajectory - Work End Time Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Hour', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axvline(x=np.mean(orig_work_ends) if orig_work_ends else 0, 
                color='blue', linestyle='--', linewidth=2, label=f'Average: {np.mean(orig_work_ends):.1f}h' if orig_work_ends else 'N/A')
    ax4.legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'work_time_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Work time distribution saved to: {save_path}")
    
    # Print statistics
    if gen_work_starts and orig_work_starts:
        print(f"\nWork time statistics:")
        print(f"  Generated trajectory - Average start time: {np.mean(gen_work_starts):.2f}h (std: {np.std(gen_work_starts):.2f}h)")
        print(f"  True trajectory - Average start time: {np.mean(orig_work_starts):.2f}h (std: {np.std(orig_work_starts):.2f}h)")
        print(f"  Generated trajectory - Average end time: {np.mean(gen_work_ends):.2f}h (std: {np.std(gen_work_ends):.2f}h)")
        print(f"  True trajectory - Average end time: {np.mean(orig_work_ends):.2f}h (std: {np.std(orig_work_ends):.2f}h)")


def main():
    """主函数"""
    print("="*80)
    print("Trajectory Comparison Visualization Analysis")
    print("="*80)
    
    # Configuration
    GENERATED_FILE = r"E:\mayue\FinalTraj\Trajectory_Generation_multi_agent\output_trajectories\all_trajectories_20251215_163347.json"
    ORIGINAL_FILE = r"E:\mayue\FinalTraj\California\processed_data\all_user_schedules.json"
    OUTPUT_DIR = r"E:\mayue\FinalTraj\evaluation\visualization_analysis_finetune"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    print(f"\nLoading data...")
    gen_dict, orig_dict = load_trajectories(GENERATED_FILE, ORIGINAL_FILE)
    print(f"  Generated trajectory users: {len(gen_dict)}")
    print(f"  True trajectory users: {len(orig_dict)}")
    print(f"  Matched users: {len([u for u in gen_dict if u in orig_dict])}")
    
    # 1. Problem pattern analysis
    problems = analyze_problem_patterns(gen_dict, orig_dict, OUTPUT_DIR)
    
    # 2. Activity distribution comparison
    visualize_activity_distribution(gen_dict, orig_dict, OUTPUT_DIR)
    
    # 3. Work time distribution
    visualize_work_time_distribution(gen_dict, orig_dict, OUTPUT_DIR)
    
    # 4. Generate detailed comparison charts for individual users
    print(f"\nGenerating detailed comparison charts for individual users...")
    user_viz_dir = os.path.join(OUTPUT_DIR, 'user_comparisons')
    os.makedirs(user_viz_dir, exist_ok=True)
    
    # Select some representative users
    sample_users = []
    
    # Add problem users
    if problems['Night shift work not recognized']:
        sample_users.extend(problems['Night shift work not recognized'][:3])
    if problems['Generated unnecessary work']:
        sample_users.extend(problems['Generated unnecessary work'][:3])
    if problems['Work time mismatch']:
        sample_users.extend(problems['Work time mismatch'][:3])
    
    # Remove duplicates
    sample_users = list(set(sample_users))[:10]
    
    for user_id in sample_users:
        if user_id in gen_dict and user_id in orig_dict:
            save_path = os.path.join(user_viz_dir, f'{user_id}_comparison.png')
            visualize_single_comparison(user_id, gen_dict[user_id], orig_dict[user_id], save_path)
    
    print(f"  ✓ Generated detailed comparison charts for {len(sample_users)} users")
    print(f"  Save path: {user_viz_dir}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  - problem_analysis.json: Problem user list")
    print(f"  - activity_distribution_comparison.png: Activity distribution comparison")
    print(f"  - work_time_distribution.png: Work time distribution")
    print(f"  - user_comparisons/: Individual user detailed comparison charts")


if __name__ == "__main__":
    main()
