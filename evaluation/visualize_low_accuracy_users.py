#!/usr/bin/env python3
"""
为准确率低的用户生成可视化对比图表
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
import numpy as np
import os

# Font settings removed - using default English fonts
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

# 文件路径
ANALYSIS_FILE = r"E:\FrankYcj\FinalTraj\evaluation\visualization_analysis\low_accuracy_users_analysis.json"
OUTPUT_DIR = r"E:\FrankYcj\FinalTraj\evaluation\visualization_analysis\low_accuracy_users"

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


def plot_user_comparison(user_data, user_idx, output_dir):
    """为单个用户绘制对比图"""
    user_id = user_data['user_id']
    accuracy = user_data['accuracy']
    
    # 解析轨迹
    orig_schedule = parse_schedule(user_data['trajectories']['original'])
    gen_schedule = parse_schedule(user_data['trajectories']['generated'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))
    fig.suptitle(f'User {user_idx}: {user_id} (Accuracy: {accuracy:.2%})', fontsize=20, fontweight='bold')
    
    # Plot original trajectory
    ax1.set_xlim(0, 1440)  # 0-24 hours (minutes)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Time', fontsize=16)
    ax1.set_ylabel('Original Trajectory', fontsize=16)
    ax1.set_yticks([])
    
    for seg in orig_schedule:
        start = time_to_minutes(seg['start_time'])
        end = time_to_minutes(seg['end_time'])
        activity = seg['activity']
        color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
        
        ax1.barh(0.5, end - start, left=start, height=0.5, 
                color=color, edgecolor='black', linewidth=1)
        
        # 添加活动标签
        mid = (start + end) / 2
        width = end - start
        if width > 60:  # 只在较宽的段上显示文字
            ax1.text(mid, 0.5, activity, ha='center', va='center', fontsize=12)
    
    # 设置x轴刻度 (每2小时)
    ax1.set_xticks([i * 120 for i in range(13)])
    ax1.set_xticklabels([f'{i*2:02d}:00' for i in range(13)], fontsize=14)
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Plot generated trajectory
    ax2.set_xlim(0, 1440)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time', fontsize=16)
    ax2.set_ylabel('Generated Trajectory', fontsize=16)
    ax2.set_yticks([])
    
    for seg in gen_schedule:
        start = time_to_minutes(seg['start_time'])
        end = time_to_minutes(seg['end_time'])
        activity = seg['activity']
        color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
        
        ax2.barh(0.5, end - start, left=start, height=0.5, 
                color=color, edgecolor='black', linewidth=1)
        
        # 添加活动标签
        mid = (start + end) / 2
        width = end - start
        if width > 60:
            ax2.text(mid, 0.5, activity, ha='center', va='center', fontsize=12)
    
    ax2.set_xticks([i * 120 for i in range(13)])
    ax2.set_xticklabels([f'{i*2:02d}:00' for i in range(13)], fontsize=14)
    ax2.grid(True, axis='x', alpha=0.3)
    
    # 添加图例
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=activity)
                      for activity, color in ACTIVITY_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=12)
    
    # Add user information
    person_info = user_data['person_info']
    info_text = f"Employment: {person_info['employment_status']} | Work Schedule: {person_info['work_schedule']} | Occupation: {person_info['occupation']}"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=13, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # 保存图片
    output_file = os.path.join(output_dir, f'user_{user_idx:02d}_{user_id}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def create_summary_table(all_users_data, output_dir):
    """创建汇总对比表"""
    num_users = len(all_users_data)
    
    # 创建一个大图,每个用户占2行
    fig = plt.figure(figsize=(18, 3 * num_users))
    
    for idx, user_data in enumerate(all_users_data):
        user_id = user_data['user_id']
        accuracy = user_data['accuracy']
        
        # 解析轨迹
        orig_schedule = parse_schedule(user_data['trajectories']['original'])
        gen_schedule = parse_schedule(user_data['trajectories']['generated'])
        
        # 原始轨迹
        ax1 = plt.subplot(num_users * 2, 1, idx * 2 + 1)
        ax1.set_xlim(0, 1440)
        ax1.set_ylim(0, 1)
        ax1.set_yticks([])
        
        if idx == 0:
            ax1.set_title('All Low Accuracy Users Comparison (Original vs Generated)', fontsize=18, fontweight='bold', pad=20)
        
        # 用户标签
        ax1.text(-100, 0.5, f'{idx+1}. {user_id}\n({accuracy:.1%})', 
                ha='right', va='center', fontsize=13, fontweight='bold')
        
        for seg in orig_schedule:
            start = time_to_minutes(seg['start_time'])
            end = time_to_minutes(seg['end_time'])
            activity = seg['activity']
            color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
            
            ax1.barh(0.5, end - start, left=start, height=0.6, 
                    color=color, edgecolor='black', linewidth=0.5)
            
            mid = (start + end) / 2
            width = end - start
            if width > 40:
                ax1.text(mid, 0.5, activity, ha='center', va='center', fontsize=10)
        
        if idx == 0:
            ax1.set_xticks([i * 120 for i in range(13)])
            ax1.set_xticklabels([f'{i*2:02d}:00' for i in range(13)], fontsize=12)
        else:
            ax1.set_xticks([])
        
        ax1.grid(True, axis='x', alpha=0.2)
        ax1.set_ylabel('Original', fontsize=12, rotation=0, labelpad=30)
        
        # 生成轨迹
        ax2 = plt.subplot(num_users * 2, 1, idx * 2 + 2)
        ax2.set_xlim(0, 1440)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        
        for seg in gen_schedule:
            start = time_to_minutes(seg['start_time'])
            end = time_to_minutes(seg['end_time'])
            activity = seg['activity']
            color = ACTIVITY_COLORS.get(activity, '#CCCCCC')
            
            ax2.barh(0.5, end - start, left=start, height=0.6, 
                    color=color, edgecolor='black', linewidth=0.5)
            
            mid = (start + end) / 2
            width = end - start
            if width > 40:
                ax2.text(mid, 0.5, activity, ha='center', va='center', fontsize=10)
        
        if idx == num_users - 1:
            ax2.set_xticks([i * 120 for i in range(13)])
            ax2.set_xticklabels([f'{i*2:02d}:00' for i in range(13)], fontsize=12)
            ax2.set_xlabel('Time', fontsize=14)
        else:
            ax2.set_xticks([])
        
        ax2.grid(True, axis='x', alpha=0.2)
        ax2.set_ylabel('Generated', fontsize=12, rotation=0, labelpad=30)
    
    # 添加图例
    legend_elements = [mpatches.Patch(facecolor=color, edgecolor='black', label=activity)
                      for activity, color in ACTIVITY_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper right', ncol=5, fontsize=13, 
              bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'all_users_comparison.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_file


def create_info_table(all_users_data, output_dir):
    """创建用户信息表格"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['#', 'User ID', 'Accuracy', 'Employment', 'Work Schedule', 'Occupation', 'Main Issues']
    table_data = []
    
    for idx, user_data in enumerate(all_users_data, 1):
        person_info = user_data['person_info']
        
        # 提取主要问题
        issues = user_data.get('issues', [])
        main_issues = []
        for issue in issues:
            if issue['severity'] == 'high':
                main_issues.append(issue['type'])
        main_issue_str = '\n'.join(main_issues[:2]) if main_issues else 'No obvious issues'
        
        row = [
            str(idx),
            user_data['user_id'],
            f"{user_data['accuracy']:.1%}",
            person_info['employment_status'] or 'N/A',
            person_info['work_schedule'] or 'N/A',
            (person_info['occupation'] or 'N/A')[:30],  # 截断长文本
            main_issue_str
        ]
        table_data.append(row)
    
    # 创建表格
    table = ax.table(cellText=table_data, colLabels=headers, 
                    cellLoc='left', loc='center',
                    colWidths=[0.05, 0.15, 0.08, 0.12, 0.12, 0.25, 0.23])
    
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2)
    
    # 设置表头样式
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # 设置行颜色
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')
    
    plt.title('Low Accuracy Users Information Summary', fontsize=18, fontweight='bold', pad=20)
    
    output_file = os.path.join(output_dir, 'users_info_table.png')
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    return output_file


def main():
    """主函数"""
    print("="*80)
    print("Generating Visualization Comparison for Low Accuracy Users")
    print("="*80)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load analysis data
    print("\nLoading analysis data...")
    with open(ANALYSIS_FILE, 'r', encoding='utf-8') as f:
        all_users_data = json.load(f)
    
    print(f"Found {len(all_users_data)} users")
    
    # Generate individual user comparison charts
    print("\nGenerating individual user comparison charts...")
    for idx, user_data in enumerate(all_users_data, 1):
        output_file = plot_user_comparison(user_data, idx, OUTPUT_DIR)
        print(f"  ✓ {idx}/{len(all_users_data)}: {user_data['user_id']}")
    
    # Generate summary comparison table
    print("\nGenerating summary comparison table...")
    summary_file = create_summary_table(all_users_data, OUTPUT_DIR)
    print(f"  ✓ Summary comparison table: {summary_file}")
    
    # Generate information table
    print("\nGenerating user information table...")
    info_table_file = create_info_table(all_users_data, OUTPUT_DIR)
    print(f"  ✓ Information table: {info_table_file}")
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"All images saved to: {OUTPUT_DIR}")
    print("="*80)
    
    # List generated files
    print("\nGenerated files:")
    files = os.listdir(OUTPUT_DIR)
    for f in sorted(files):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
