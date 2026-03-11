import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
import os

# Set English font and style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 创建输出目录
output_dir = r'E:\FrankYcj\FinalTraj\data_analysis'
graphs_dir = r'E:\FrankYcj\FinalTraj\data_analysis\graphs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

# 定义州的轨迹数据文件
states_data = {
    'Arizona': r'E:\FrankYcj\FinalTraj\Arizona\processed_data\Arizona_trajectories_processed.json',
    'California': r'E:\FrankYcj\FinalTraj\California\processed_data\california_trajectories_processed.json',
    'Georgia': r'E:\FrankYcj\FinalTraj\Georgia\processed_data\Georgia_trajectories_processed.json',
    'Oklahoma': r'E:\FrankYcj\FinalTraj\Oklahoma\processed_data\Oklahoma_trajectories_processed.json',
    'Wisconsin': r'E:\FrankYcj\FinalTraj\wisconsin\processed_data\Wisconsin_trajectories_processed.json'
}

# 活动类别映射
activity_names = {
    0: 'unknown',
    1: 'home',
    2: 'work',
    3: 'education',
    4: 'shopping',
    5: 'service',
    6: 'medical',
    7: 'dine_out',
    8: 'socialize',
    9: 'exercise',
    10: 'dropoff_pickup'
}

print("=" * 80)
print("活动链模式分析 - 多州比较")
print("=" * 80)

# 读取所有州的数据
all_states_data = {}
for state, filepath in states_data.items():
    print(f"\n正在读取 {state} 的数据...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_states_data[state] = data
            print(f"  - 用户数: {len(data)}")
            total_trajectories = sum(len(user['trajectories']) for user in data)
            print(f"  - 总轨迹段数: {total_trajectories}")
    except Exception as e:
        print(f"  - 读取失败: {e}")

print("\n" + "=" * 80)

# ============================================================================
# 分析函数定义
# ============================================================================

def extract_activity_chain(user):
    """提取用户的活动链"""
    activities = []
    for traj in user['trajectories']:
        activity = traj.get('activity', 'unknown')
        activities.append(activity)
    return activities

def get_activity_sequence(user):
    """获取活动序列（去除连续重复）"""
    chain = extract_activity_chain(user)
    if not chain:
        return []
    
    sequence = [chain[0]]
    for activity in chain[1:]:
        if activity != sequence[-1]:
            sequence.append(activity)
    return sequence

def analyze_state_patterns(state_name, users_data):
    """分析单个州的活动模式"""
    print(f"\n分析 {state_name} 的活动模式...")
    
    results = {
        'state': state_name,
        'num_users': len(users_data),
        'activity_counts': Counter(),
        'activity_durations': defaultdict(list),
        'activity_sequences': [],
        'trip_purposes': Counter(),
        'hourly_activities': defaultdict(lambda: Counter()),
        'chain_lengths': [],
        'sequence_lengths': []
    }
    
    for user in users_data:
        trajectories = user.get('trajectories', [])
        
        # 活动链长度
        results['chain_lengths'].append(len(trajectories))
        
        # 活动序列
        sequence = get_activity_sequence(user)
        results['sequence_lengths'].append(len(sequence))
        results['activity_sequences'].append(tuple(sequence))
        
        for traj in trajectories:
            activity = traj.get('activity', 'unknown')
            duration = traj.get('duration', 0)
            hour = traj.get('hour', 0)
            home_based = traj.get('home_based', 'unknown')
            
            # 统计活动次数
            results['activity_counts'][activity] += 1
            
            # 统计活动时长
            results['activity_durations'][activity].append(duration)
            
            # 统计出行目的
            results['trip_purposes'][home_based] += 1
            
            # 统计每小时的活动分布
            results['hourly_activities'][hour][activity] += 1
    
    # 计算平均时长
    results['avg_durations'] = {
        activity: np.mean(durations) 
        for activity, durations in results['activity_durations'].items()
    }
    
    # 统计最常见的活动序列
    results['top_sequences'] = Counter(results['activity_sequences']).most_common(20)
    
    return results

# ============================================================================
# 分析所有州的数据
# ============================================================================

states_results = {}
for state, users_data in all_states_data.items():
    states_results[state] = analyze_state_patterns(state, users_data)

# ============================================================================
# 1. 活动分布对比
# ============================================================================

print("\n" + "=" * 80)
print("1. 生成活动分布对比图...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comparison of Activity Type Distribution Across States', fontsize=16, fontweight='bold')

for idx, (state, results) in enumerate(states_results.items()):
    ax = axes[idx // 3, idx % 3]
    
    activity_counts = results['activity_counts']
    sorted_activities = sorted(activity_counts.items(), key=lambda x: x[1], reverse=True)
    
    activities = [act for act, _ in sorted_activities]
    counts = [cnt for _, cnt in sorted_activities]
    
    bars = ax.barh(activities, counts, color=sns.color_palette("husl", len(activities)))
    ax.set_xlabel('Number of Activities', fontsize=10)
    ax.set_title(f'{state} (N={results["num_users"]})', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (act, cnt) in enumerate(sorted_activities):
        ax.text(cnt, i, f' {cnt:,}', va='center', fontsize=9)

# 移除多余的子图
if len(states_results) < 6:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '01_activity_distribution_by_state.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 01_activity_distribution_by_state.png")
plt.close()

# ============================================================================
# 2. 活动时长分布对比
# ============================================================================

print("\n2. 生成活动时长分布对比图...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(' Comparison of Average Activity Duration Across States (Minutes)', fontsize=16, fontweight='bold')

for idx, (state, results) in enumerate(states_results.items()):
    ax = axes[idx // 3, idx % 3]
    
    avg_durations = results['avg_durations']
    sorted_durations = sorted(avg_durations.items(), key=lambda x: x[1], reverse=True)
    
    activities = [act for act, _ in sorted_durations]
    durations = [dur for _, dur in sorted_durations]
    
    bars = ax.barh(activities, durations, color=sns.color_palette("viridis", len(activities)))
    ax.set_xlabel('Average Duration (Minutes)', fontsize=10)
    ax.set_title(f'{state}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (act, dur) in enumerate(sorted_durations):
        ax.text(dur, i, f' {dur:.1f}', va='center', fontsize=9)

# 移除多余的子图
if len(states_results) < 6:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '02_activity_duration_by_state.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 02_activity_duration_by_state.png")
plt.close()

# ============================================================================
# 3. 活动链长度分布
# ============================================================================

print("\n3. 生成活动链长度分布图...")

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

for state, results in states_results.items():
    chain_lengths = results['chain_lengths']
    ax.hist(chain_lengths, bins=50, alpha=0.5, label=state, edgecolor='black')

ax.set_xlabel('Activity Number', fontsize=12)
ax.set_ylabel('User Count', fontsize=12)
ax.set_title('Comparison of Activity Chain Length Distribution Across States', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '03_chain_length_distribution.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 03_chain_length_distribution.png")
plt.close()

# ============================================================================
# 4. 活动序列长度分布（去重后）
# ============================================================================

print("\n4. 生成活动序列长度分布图...")

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

sequence_stats = []
for state, results in states_results.items():
    seq_lengths = results['sequence_lengths']
    ax.hist(seq_lengths, bins=30, alpha=0.5, label=state, edgecolor='black')
    sequence_stats.append({
        'state': state,
        'mean': np.mean(seq_lengths),
        'median': np.median(seq_lengths),
        'std': np.std(seq_lengths)
    })

ax.set_xlabel('Activity Sequence Length (Without Consecutive Duplicates)', fontsize=12)
ax.set_ylabel('User Count', fontsize=12)
ax.set_title('Comparison of Activity Sequence Length Distribution Across States', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '04_sequence_length_distribution.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 04_sequence_length_distribution.png")
plt.close()

# ============================================================================
# 5. 出行目的分布（Home-based类型）
# ============================================================================

print("\n5. 生成出行目的分布图...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Distribution of Travel Purposes Across States (Home-based)', fontsize=16, fontweight='bold')

for idx, (state, results) in enumerate(states_results.items()):
    ax = axes[idx // 3, idx % 3]
    
    trip_purposes = results['trip_purposes']
    sorted_purposes = sorted(trip_purposes.items(), key=lambda x: x[1], reverse=True)
    
    purposes = [p for p, _ in sorted_purposes]
    counts = [c for _, c in sorted_purposes]
    
    colors = sns.color_palette("Set2", len(purposes))
    wedges, texts, autotexts = ax.pie(counts, labels=purposes, autopct='%1.1f%%',
                                        startangle=90, colors=colors)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax.set_title(f'{state}', fontsize=12, fontweight='bold')

# 移除多余的子图
if len(states_results) < 6:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '05_trip_purpose_distribution.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 05_trip_purpose_distribution.png")
plt.close()

# ============================================================================
# 6. 24小时活动分布热力图
# ============================================================================

print("\n6. 生成24小时活动分布热力图...")

for state, results in states_results.items():
    hourly_data = results['hourly_activities']
    
    # 创建数据矩阵
    hours = range(24)
    main_activities = ['home', 'work', 'shopping', 'education', 'socialize', 'exercise', 'dine_out']
    
    matrix = []
    for activity in main_activities:
        row = [hourly_data[hour].get(activity, 0) for hour in hours]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # 归一化（按行）
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除以零
    matrix_norm = matrix / row_sums * 100
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    sns.heatmap(matrix_norm, annot=False, fmt='.1f', cmap='YlOrRd',
                xticklabels=hours, yticklabels=main_activities,
                cbar_kws={'label': 'Percentage (%)'}, ax=ax)

    ax.set_xlabel('Hour', fontsize=12)
    ax.set_ylabel('Activity Type', fontsize=12)
    ax.set_title(f'{state} - Activity Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, f'06_hourly_heatmap_{state}.png'), dpi=300, bbox_inches='tight')
    print(f"  保存: 06_hourly_heatmap_{state}.png")
    plt.close()

# ============================================================================
# 7. 跨州活动占比对比
# ============================================================================

print("\n7. 生成跨州活动占比对比图...")

# 收集所有活动类型
all_activities = set()
for results in states_results.values():
    all_activities.update(results['activity_counts'].keys())

all_activities = sorted(list(all_activities))

# 创建数据矩阵
activity_matrix = []
for activity in all_activities:
    row = []
    for state in states_results.keys():
        total = sum(states_results[state]['activity_counts'].values())
        count = states_results[state]['activity_counts'].get(activity, 0)
        percentage = (count / total * 100) if total > 0 else 0
        row.append(percentage)
    activity_matrix.append(row)

activity_matrix = np.array(activity_matrix)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))

sns.heatmap(activity_matrix, annot=True, fmt='.1f', cmap='coolwarm',
            xticklabels=list(states_results.keys()),
            yticklabels=all_activities,
            cbar_kws={'label': 'Percentage (%)'}, ax=ax)

ax.set_xlabel('States', fontsize=12)
ax.set_ylabel('Activity Types', fontsize=12)
ax.set_title('Comparison of Activity Type Distribution Across States', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '07_cross_state_activity_comparison.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 07_cross_state_activity_comparison.png")
plt.close()

# ============================================================================
# 8. 最常见的活动序列模式
# ============================================================================

print("\n8. 分析最常见的活动序列模式...")

for state, results in states_results.items():
    top_sequences = results['top_sequences'][:10]
    
    print(f"\n{state} - Top 10 活动序列:")
    for i, (seq, count) in enumerate(top_sequences, 1):
        seq_str = ' → '.join(seq)
        print(f"  {i}. [{count:,}次] {seq_str}")

# ============================================================================
# 9. 生成统计报告
# ============================================================================

print("\n9. 生成统计报告...")

report_lines = []
report_lines.append("=" * 100)
report_lines.append("活动链模式分析报告")
report_lines.append("=" * 100)
report_lines.append("")

for state, results in states_results.items():
    report_lines.append(f"\n{'=' * 100}")
    report_lines.append(f"{state} 州")
    report_lines.append(f"{'=' * 100}")
    report_lines.append(f"用户数量: {results['num_users']:,}")
    report_lines.append(f"总轨迹段数: {sum(results['activity_counts'].values()):,}")
    report_lines.append(f"平均活动链长度: {np.mean(results['chain_lengths']):.2f}")
    report_lines.append(f"平均活动序列长度: {np.mean(results['sequence_lengths']):.2f}")
    
    report_lines.append(f"\n活动类型分布 (Top 10):")
    sorted_activities = sorted(results['activity_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
    for activity, count in sorted_activities:
        percentage = count / sum(results['activity_counts'].values()) * 100
        report_lines.append(f"  {activity:15s}: {count:8,} ({percentage:5.2f}%)")
    
    report_lines.append(f"\n平均活动时长 (Top 10):")
    sorted_durations = sorted(results['avg_durations'].items(), key=lambda x: x[1], reverse=True)[:10]
    for activity, duration in sorted_durations:
        report_lines.append(f"  {activity:15s}: {duration:8.1f} 分钟")
    
    report_lines.append(f"\n出行目的分布:")
    sorted_purposes = sorted(results['trip_purposes'].items(), key=lambda x: x[1], reverse=True)
    for purpose, count in sorted_purposes:
        percentage = count / sum(results['trip_purposes'].values()) * 100
        report_lines.append(f"  {purpose:10s}: {count:8,} ({percentage:5.2f}%)")

report_lines.append(f"\n\n{'=' * 100}")
report_lines.append("跨州对比总结")
report_lines.append(f"{'=' * 100}")

# 活动序列长度对比
report_lines.append(f"\n活动序列长度统计:")
report_lines.append(f"{'州':15s} {'平均':>10s} {'中位数':>10s} {'标准差':>10s}")
report_lines.append("-" * 50)
for stat in sequence_stats:
    report_lines.append(f"{stat['state']:15s} {stat['mean']:10.2f} {stat['median']:10.2f} {stat['std']:10.2f}")

# 保存报告
report_file = os.path.join(output_dir, 'activity_chain_analysis_report.txt')
with open(report_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"  保存: activity_chain_analysis_report.txt")

# ============================================================================
# 10. 保存统计数据为CSV
# ============================================================================

print("\n10. 保存统计数据为CSV...")

# 活动分布统计
activity_dist_data = []
for state, results in states_results.items():
    total = sum(results['activity_counts'].values())
    for activity, count in results['activity_counts'].items():
        activity_dist_data.append({
            'State': state,
            'Activity': activity,
            'Count': count,
            'Percentage': count / total * 100 if total > 0 else 0
        })

df_activity_dist = pd.DataFrame(activity_dist_data)
df_activity_dist.to_csv(os.path.join(output_dir, 'activity_distribution.csv'), index=False, encoding='utf-8-sig')
print(f"  保存: activity_distribution.csv")

# 活动时长统计
duration_data = []
for state, results in states_results.items():
    for activity, duration in results['avg_durations'].items():
        duration_data.append({
            'State': state,
            'Activity': activity,
            'Average_Duration_Minutes': duration
        })

df_duration = pd.DataFrame(duration_data)
df_duration.to_csv(os.path.join(output_dir, 'activity_durations.csv'), index=False, encoding='utf-8-sig')
print(f"  保存: activity_durations.csv")

# 序列长度统计
seq_length_data = []
for state, results in states_results.items():
    for length in results['sequence_lengths']:
        seq_length_data.append({
            'State': state,
            'Sequence_Length': length
        })

df_seq_length = pd.DataFrame(seq_length_data)
df_seq_length.to_csv(os.path.join(output_dir, 'sequence_lengths.csv'), index=False, encoding='utf-8-sig')
print(f"  保存: sequence_lengths.csv")

print("\n" + "=" * 100)
print("分析完成！")
print(f"图表保存在: {graphs_dir}")
print(f"数据保存在: {output_dir}")
print("=" * 100)
