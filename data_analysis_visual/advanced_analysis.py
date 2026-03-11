import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# 设置样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("Set2")

# 输出目录
graphs_dir = r'E:\FrankYcj\FinalTraj\data_analysis\graphs'
data_dir = r'E:\FrankYcj\FinalTraj\data_analysis'

print("=" * 80)
print("生成高级对比分析图表")
print("=" * 80)

# 读取已经处理好的CSV数据
print("\n正在读取数据...")
df_activity_dist = pd.read_csv(os.path.join(data_dir, 'activity_distribution.csv'))
df_duration = pd.read_csv(os.path.join(data_dir, 'activity_durations.csv'))
df_seq_length = pd.read_csv(os.path.join(data_dir, 'sequence_lengths.csv'))

# ============================================================================
# 1. 各州活动占比雷达图对比
# ============================================================================

print("\n1. 生成活动占比雷达图...")

# 选择主要活动
main_activities = ['home', 'work', 'shopping', 'socialize', 'dine_out', 
                   'service', 'exercise', 'medical', 'education', 'dropoff_pickup']

# 准备数据
states = df_activity_dist['State'].unique()
radar_data = {}

for state in states:
    state_data = df_activity_dist[df_activity_dist['State'] == state]
    percentages = []
    for activity in main_activities:
        act_data = state_data[state_data['Activity'] == activity]
        if len(act_data) > 0:
            percentages.append(act_data['Percentage'].values[0])
        else:
            percentages.append(0)
    radar_data[state] = percentages

# 创建雷达图
fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
fig.suptitle('Activity Pattern Radar Chart by State', fontsize=16, fontweight='bold')

angles = np.linspace(0, 2 * np.pi, len(main_activities), endpoint=False).tolist()
angles += angles[:1]  # 闭合

for idx, (state, values) in enumerate(radar_data.items()):
    ax = axes[idx // 3, idx % 3]
    
    values += values[:1]  # 闭合
    
    ax.plot(angles, values, 'o-', linewidth=2, label=state)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(main_activities, size=8)
    ax.set_ylim(0, max([max(v) for v in radar_data.values()]) * 1.1)
    ax.set_title(state, fontsize=12, fontweight='bold', pad=20)
    ax.grid(True)

# 移除多余的子图
if len(radar_data) < 6:
    axes[1, 2].remove()

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '08_activity_radar_charts.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 08_activity_radar_charts.png")
plt.close()

# ============================================================================
# 2. 活动时长箱线图对比
# ============================================================================

print("\n2. 生成活动时长箱线图...")

# 选择主要活动进行对比
selected_activities = ['home', 'work', 'shopping', 'socialize', 'dine_out', 'exercise']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Activity Duration Comparison Across States', fontsize=16, fontweight='bold')

for idx, activity in enumerate(selected_activities):
    ax = axes[idx // 3, idx % 3]
    
    activity_data = df_duration[df_duration['Activity'] == activity]
    
    states_list = activity_data['State'].tolist()
    durations = activity_data['Average_Duration_Minutes'].tolist()
    
    bars = ax.bar(states_list, durations, color=sns.color_palette("viridis", len(states_list)))
    
    ax.set_ylabel('Minutes', fontsize=10)
    ax.set_title(f'{activity.capitalize()}', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for i, (state, duration) in enumerate(zip(states_list, durations)):
        ax.text(i, duration, f'{duration:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '08_duration_comparison.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 08_duration_comparison.png")
plt.close()

# ============================================================================
# 3. 序列长度分布箱线图
# ============================================================================

print("\n3. 生成序列长度分布箱线图...")

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# 准备箱线图数据
data_for_boxplot = []
labels_for_boxplot = []

for state in states:
    state_data = df_seq_length[df_seq_length['State'] == state]['Sequence_Length'].values
    data_for_boxplot.append(state_data)
    labels_for_boxplot.append(state)

# 创建箱线图
bp = ax.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True,
                showmeans=True, meanline=True)

# 美化箱线图
colors = sns.color_palette("Set2", len(data_for_boxplot))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Sequence Length', fontsize=12)
ax.set_xlabel('State', fontsize=12)
ax.set_title('Activity Sequence Length Distribution by State', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 添加统计信息
for i, state in enumerate(labels_for_boxplot, 1):
    state_data = df_seq_length[df_seq_length['State'] == state]['Sequence_Length']
    median = state_data.median()
    mean = state_data.mean()
    ax.text(i, median - 0.5, f'Med: {median:.1f}', ha='center', fontsize=8)
    ax.text(i, mean + 0.5, f'Mean: {mean:.1f}', ha='center', fontsize=8, color='red')

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '09_sequence_length_boxplot.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 09_sequence_length_boxplot.png")
plt.close()

# ============================================================================
# 4. 活动类型占比堆叠条形图
# ============================================================================

print("\n4. 生成活动占比堆叠条形图...")

# 准备数据
activity_pivot = df_activity_dist.pivot_table(
    index='State',
    columns='Activity',
    values='Percentage',
    fill_value=0
)

# 只保留主要活动
activity_pivot = activity_pivot[[col for col in main_activities if col in activity_pivot.columns]]

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

activity_pivot.plot(kind='bar', stacked=True, ax=ax, 
                    color=sns.color_palette("husl", len(activity_pivot.columns)))

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xlabel('State', fontsize=12)
ax.set_title('Activity Type Distribution by State', fontsize=14, fontweight='bold')
ax.legend(title='Activity', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '10_activity_stacked_bar.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 10_activity_stacked_bar.png")
plt.close()

# ============================================================================
# 5. Top 3 活动对比（分组条形图）
# ============================================================================

print("\n5. 生成 Top 3 活动对比图...")

top_activities = ['home', 'shopping', 'work']

# 准备数据
comparison_data = []
for state in states:
    for activity in top_activities:
        state_act_data = df_activity_dist[
            (df_activity_dist['State'] == state) & 
            (df_activity_dist['Activity'] == activity)
        ]
        if len(state_act_data) > 0:
            comparison_data.append({
                'State': state,
                'Activity': activity,
                'Percentage': state_act_data['Percentage'].values[0]
            })

df_comparison = pd.DataFrame(comparison_data)

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# 分组条形图
x = np.arange(len(states))
width = 0.25

for i, activity in enumerate(top_activities):
    activity_data = df_comparison[df_comparison['Activity'] == activity]
    percentages = [activity_data[activity_data['State'] == state]['Percentage'].values[0] 
                   for state in states]
    
    ax.bar(x + i * width, percentages, width, label=activity.capitalize())

ax.set_xlabel('State', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Top 3 Activities Comparison Across States', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(states, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '11_top3_activities_comparison.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 11_top3_activities_comparison.png")
plt.close()

# ============================================================================
# 6. 生成统计摘要表格图
# ============================================================================

print("\n6. 生成统计摘要表格...")

# 计算每个州的关键统计数据
summary_stats = []

for state in states:
    state_dist = df_activity_dist[df_activity_dist['State'] == state]
    state_seq = df_seq_length[df_seq_length['State'] == state]
    
    # Top 3 activities
    top3 = state_dist.nlargest(3, 'Percentage')['Activity'].tolist()
    
    # Sequence length stats
    mean_seq = state_seq['Sequence_Length'].mean()
    median_seq = state_seq['Sequence_Length'].median()
    
    summary_stats.append({
        'State': state,
        'Top Activity': top3[0] if len(top3) > 0 else 'N/A',
        'Mean Seq Length': f'{mean_seq:.2f}',
        'Median Seq Length': f'{median_seq:.0f}',
        'Top 3 Activities': ', '.join(top3)
    })

df_summary = pd.DataFrame(summary_stats)

fig, ax = plt.subplots(1, 1, figsize=(14, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=df_summary.values,
                colLabels=df_summary.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.15, 0.15, 0.15, 0.4])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 设置表头样式
for i in range(len(df_summary.columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 交替行颜色
for i in range(1, len(df_summary) + 1):
    if i % 2 == 0:
        for j in range(len(df_summary.columns)):
            table[(i, j)].set_facecolor('#f0f0f0')

plt.title('Summary Statistics by State', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(graphs_dir, '12_summary_table.png'), dpi=300, bbox_inches='tight')
print(f"  保存: 12_summary_table.png")
plt.close()

print("\n" + "=" * 80)
print("高级分析图表生成完成！")
print(f"所有图表已保存至: {graphs_dir}")
print("=" * 80)
