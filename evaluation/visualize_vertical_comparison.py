#!/usr/bin/env python3
"""
用竖向条形图可视化低准确率用户的轨迹对比
参考vertical bar chart样式
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
ANALYSIS_FILE = r"E:\FrankYcj\FinalTraj\evaluation\visualization_analysis\low_accuracy_users_analysis.json"
OUTPUT_DIR = r"E:\FrankYcj\FinalTraj\evaluation\visualization_analysis\low_accuracy_users"

# 活动类型到颜色的映射 (与其他可视化脚本保持一致)
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


def parse_schedule(schedule_str):
    """解析schedule字符串"""
    activities = []
    segments = schedule_str.split('; ')
    for seg in segments:
        if ': ' not in seg:
            continue
        time_part, activity = seg.split(': ')
        start_time, end_time = time_part.split('-')
        activities.append({
            'activity': activity,
            'start_time': start_time,
            'end_time': end_time
        })
    return activities


def time_to_minutes(time_str):
    """将时间字符串转换为分钟数"""
    if time_str == "24:00":
        return 1440
    parts = time_str.split(':')
    return int(parts[0]) * 60 + int(parts[1])


def create_vertical_comparison(all_users_data, output_dir):
    """创建竖向对比图 - True vs Predicted"""
    num_users = len(all_users_data)
    
    # 创建图形 - 两个子图并排
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # 设置标题
    ax1.set_title('True', fontsize=20, fontweight='bold')
    ax2.set_title('Predicted', fontsize=20, fontweight='bold')
    
    # 处理每个用户
    for idx, user_data in enumerate(all_users_data):
        user_id = user_data['user_id']
        
        # 解析轨迹
        orig_schedule = parse_schedule(user_data['trajectories']['original'])
        gen_schedule = parse_schedule(user_data['trajectories']['generated'])
        
        # 绘制原始轨迹 (True)
        for seg in orig_schedule:
            start = time_to_minutes(seg['start_time'])
            end = time_to_minutes(seg['end_time'])
            activity = seg['activity']
            color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
            
            # 转换为小时刻度
            start_hour = start / 60
            end_hour = end / 60
            height = end_hour - start_hour
            
            ax1.bar(idx + 1, height, bottom=start_hour, width=0.8,
                   color=color, edgecolor='black', linewidth=0.5)
        
        # 绘制生成轨迹 (Predicted)
        for seg in gen_schedule:
            start = time_to_minutes(seg['start_time'])
            end = time_to_minutes(seg['end_time'])
            activity = seg['activity']
            color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
            
            start_hour = start / 60
            end_hour = end / 60
            height = end_hour - start_hour
            
            ax2.bar(idx + 1, height, bottom=start_hour, width=0.8,
                   color=color, edgecolor='black', linewidth=0.5)
    
    # 设置ax1
    ax1.set_ylim(0, 24)
    ax1.set_xlim(0, num_users + 1)
    ax1.set_ylabel('Time of Day', fontsize=16)
    ax1.set_xlabel('User', fontsize=16)
    ax1.set_xticks(range(1, num_users + 1))
    ax1.set_xticklabels(range(1, num_users + 1), fontsize=12)
    ax1.set_yticks([0, 4, 8, 12, 16, 20, 24])
    ax1.set_yticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=12)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.invert_yaxis()  # 反转y轴,使00:00在顶部
    
    # 设置ax2
    ax2.set_ylim(0, 24)
    ax2.set_xlim(0, num_users + 1)
    ax2.set_ylabel('Time of Day', fontsize=16)
    ax2.set_xlabel('User', fontsize=16)
    ax2.set_xticks(range(1, num_users + 1))
    ax2.set_xticklabels(range(1, num_users + 1), fontsize=12)
    ax2.set_yticks([0, 4, 8, 12, 16, 20, 24])
    ax2.set_yticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.invert_yaxis()
    
    # 添加图例
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=activity)
                      for activity, color in ACTIVITY_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper right', ncol=1, fontsize=12,
              bbox_to_anchor=(0.98, 0.95))
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'vertical_comparison.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_file


def create_single_user_vertical(user_data, user_idx, output_dir):
    """为单个用户创建竖向对比图"""
    user_id = user_data['user_id']
    accuracy = user_data['accuracy']
    
    # 解析轨迹
    orig_schedule = parse_schedule(user_data['trajectories']['original'])
    gen_schedule = parse_schedule(user_data['trajectories']['generated'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.suptitle(f'User {user_idx}: {user_id} (Accuracy: {accuracy:.2%})', fontsize=16, fontweight='bold')
    
    # Set titles
    ax1.set_title('Original', fontsize=14, fontweight='bold')
    ax2.set_title('Generated', fontsize=14, fontweight='bold')
    
    # 绘制原始轨迹
    for seg in orig_schedule:
        start = time_to_minutes(seg['start_time'])
        end = time_to_minutes(seg['end_time'])
        activity = seg['activity']
        color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
        
        start_hour = start / 60
        end_hour = end / 60
        height = end_hour - start_hour
        
        bar = ax1.bar(0.5, height, bottom=start_hour, width=0.6,
               color=color, edgecolor='black', linewidth=1)
        
        # 添加活动标签
        if height > 1:  # 只在足够高的条上显示文字
            mid_hour = (start_hour + end_hour) / 2
            ax1.text(0.5, mid_hour, activity, ha='center', va='center', 
                    fontsize=10, rotation=0, fontweight='bold')
    
    # 绘制生成轨迹
    for seg in gen_schedule:
        start = time_to_minutes(seg['start_time'])
        end = time_to_minutes(seg['end_time'])
        activity = seg['activity']
        color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
        
        start_hour = start / 60
        end_hour = end / 60
        height = end_hour - start_hour
        
        bar = ax2.bar(0.5, height, bottom=start_hour, width=0.6,
               color=color, edgecolor='black', linewidth=1)
        
        if height > 1:
            mid_hour = (start_hour + end_hour) / 2
            ax2.text(0.5, mid_hour, activity, ha='center', va='center', 
                    fontsize=10, rotation=0, fontweight='bold')
    
    # 设置ax1
    ax1.set_ylim(0, 24)
    ax1.set_xlim(0, 1)
    ax1.set_ylabel('Time of Day', fontsize=14)
    ax1.set_xticks([])
    ax1.set_yticks([0, 4, 8, 12, 16, 20, 24])
    ax1.set_yticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=12)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.invert_yaxis()
    
    # 设置ax2
    ax2.set_ylim(0, 24)
    ax2.set_xlim(0, 1)
    ax2.set_ylabel('Time of Day', fontsize=14)
    ax2.set_xticks([])
    ax2.set_yticks([0, 4, 8, 12, 16, 20, 24])
    ax2.set_yticklabels(['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'], fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.invert_yaxis()
    
    # Add user information
    person_info = user_data['person_info']
    info_text = f"Employment: {person_info['employment_status']} | Work Schedule: {person_info['work_schedule']} | Occupation: {person_info['occupation']}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    output_file = os.path.join(output_dir, f'vertical_user_{user_idx:02d}_{user_id}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def main():
    """主函数"""
    print("="*80)
    print("Generating Vertical Bar Chart Visualization")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load analysis data
    print("\nLoading analysis data...")
    with open(ANALYSIS_FILE, 'r', encoding='utf-8') as f:
        all_users_data = json.load(f)
    
    print(f"Found {len(all_users_data)} users")
    
    # Generate summary comparison chart
    print("\nGenerating summary vertical comparison chart...")
    summary_file = create_vertical_comparison(all_users_data, OUTPUT_DIR)
    print(f"  ✓ Summary comparison chart: {summary_file}")
    
    # Generate individual vertical comparison charts for each user
    print("\nGenerating individual user vertical comparison charts...")
    for idx, user_data in enumerate(all_users_data, 1):
        output_file = create_single_user_vertical(user_data, idx, OUTPUT_DIR)
        print(f"  ✓ {idx}/{len(all_users_data)}: {user_data['user_id']}")
    
    print("\n" + "="*80)
    print("Vertical visualization complete!")
    print(f"All images saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
