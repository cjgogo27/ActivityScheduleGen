import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import os

# Set font to support international characters
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create output directory
os.makedirs('graphs_EN', exist_ok=True)

print("\n" + "="*80)
print("COMPREHENSIVE ACTIVITY CHAIN ANALYSIS - ENGLISH VERSION")
print("="*80 + "\n")

# Load data from all states
def load_trajectories(state_name, file_path):
    """Load trajectory data for a specific state"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {state_name}: {len(data)} users")
    return data

# Load data from all states
states_data = {
    'Arizona': load_trajectories('Arizona', '../Arizona/processed_data/Arizona_trajectories_processed.json'),
    'California': load_trajectories('California', '../California/processed_data/california_trajectories_processed.json'),
    'Georgia': load_trajectories('Georgia', '../Georgia/processed_data/Georgia_trajectories_processed.json'),
    'Oklahoma': load_trajectories('Oklahoma', '../Oklahoma/processed_data/Oklahoma_trajectories_processed.json'),
    'Wisconsin': load_trajectories('Wisconsin', '../wisconsin/processed_data/wisconsin_trajectories_processed.json')
}

print("\n" + "-"*80)

# ==================== Analysis 1: Activity Type Distribution ====================
print("Running Analysis 1: Activity Type Distribution...")
activity_counts = {}

for state, data in states_data.items():
    activities = []
    # Handle both dict and list data structures
    if isinstance(data, dict):
        for user_id, user_data in data.items():
            for segment in user_data['trajectory']:
                activities.append(segment['activity'])
    else:  # list
        for user_data in data:
            for segment in user_data['trajectory']:
                activities.append(segment['activity'])
    activity_counts[state] = Counter(activities)

# Create DataFrame
df_activity = pd.DataFrame(activity_counts).fillna(0)

# Plot
fig, ax = plt.subplots(figsize=(14, 8))
df_activity.plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Activity Type Distribution by State', fontsize=16, fontweight='bold')
ax.set_xlabel('Activity Type', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.legend(title='State', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('graphs_EN/activity_distribution_by_state.png', dpi=300, bbox_inches='tight')
plt.close()

df_activity.to_csv('activity_distribution.csv')
print("✓ Analysis 1 completed\n")

# ==================== Analysis 2: Activity Duration ====================
print("Running Analysis 2: Activity Duration...")
duration_data = defaultdict(list)

for state, data in states_data.items():
    if isinstance(data, dict):
        for user_id, user_data in data.items():
            for segment in user_data['trajectory']:
                duration_data[segment['activity']].append(segment['duration'])
    else:  # list
        for user_data in data:
            for segment in user_data['trajectory']:
                duration_data[segment['activity']].append(segment['duration'])

# Calculate statistics
stats = []
for activity, durations in duration_data.items():
    stats.append({
        'Activity': activity,
        'Mean Duration (min)': np.mean(durations),
        'Median Duration (min)': np.median(durations),
        'Std Dev (min)': np.std(durations),
        'Total Count': len(durations)
    })

df_duration = pd.DataFrame(stats).sort_values('Mean Duration (min)', ascending=False)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
activities = df_duration['Activity'].tolist()
means = df_duration['Mean Duration (min)'].tolist()

bars = ax.bar(activities, means, color=sns.color_palette('viridis', len(activities)))
ax.set_title('Average Activity Duration by Type', fontsize=16, fontweight='bold')
ax.set_xlabel('Activity Type', fontsize=12)
ax.set_ylabel('Average Duration (minutes)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('graphs_EN/activity_duration_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

df_duration.to_csv('activity_durations.csv', index=False)
print("✓ Analysis 2 completed\n")

# ==================== Analysis 3: Activity Sequence Patterns ====================
print("Running Analysis 3: Activity Sequence Patterns...")
sequence_lengths = defaultdict(list)
common_patterns = defaultdict(Counter)

for state, data in states_data.items():
    if isinstance(data, dict):
        for user_id, user_data in data.items():
            trajectory = user_data['trajectory']
            sequence_lengths[state].append(len(trajectory))
            
            # Extract activity sequence
            activity_seq = [seg['activity'] for seg in trajectory]
            # Get top 3-activity patterns
            for i in range(len(activity_seq) - 2):
                pattern = tuple(activity_seq[i:i+3])
                common_patterns[state][pattern] += 1
    else:  # list
        for user_data in data:
            trajectory = user_data['trajectory']
            sequence_lengths[state].append(len(trajectory))
            
            # Extract activity sequence
            activity_seq = [seg['activity'] for seg in trajectory]
            # Get top 3-activity patterns
            for i in range(len(activity_seq) - 2):
                pattern = tuple(activity_seq[i:i+3])
                common_patterns[state][pattern] += 1

# Plot sequence length distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (state, lengths) in enumerate(sequence_lengths.items()):
    ax = axes[idx]
    ax.hist(lengths, bins=30, color=sns.color_palette('Set2')[idx], edgecolor='black', alpha=0.7)
    ax.set_title(f'{state}\nAvg: {np.mean(lengths):.1f} activities', fontsize=12, fontweight='bold')
    ax.set_xlabel('Number of Activities', fontsize=10)
    ax.set_ylabel('User Count', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

axes[5].axis('off')

plt.suptitle('Activity Chain Length Distribution by State', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('graphs_EN/sequence_length_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Save statistics
seq_stats = []
for state, lengths in sequence_lengths.items():
    seq_stats.append({
        'State': state,
        'Avg Length': np.mean(lengths),
        'Median Length': np.median(lengths),
        'Min Length': np.min(lengths),
        'Max Length': np.max(lengths)
    })
pd.DataFrame(seq_stats).to_csv('sequence_lengths.csv', index=False)

print("✓ Analysis 3 completed\n")

# ==================== Analysis 4: Hourly Activity Heatmap ====================
print("Running Analysis 4: Hourly Activity Patterns...")
hourly_activity = defaultdict(lambda: defaultdict(int))

for state, data in states_data.items():
    if isinstance(data, dict):
        for user_id, user_data in data.items():
            for segment in user_data['trajectory']:
                # Parse start time
                time_parts = segment['start_time'].split(':')
                hour = int(time_parts[0])
                activity = segment['activity']
                hourly_activity[state][(hour, activity)] += 1
    else:  # list
        for user_data in data:
            for segment in user_data['trajectory']:
                # Parse start time
                time_parts = segment['start_time'].split(':')
                hour = int(time_parts[0])
                activity = segment['activity']
                hourly_activity[state][(hour, activity)] += 1

# Create heatmap for each state
for state in states_data.keys():
    # Prepare data matrix
    activities = sorted(set(act for (h, act) in hourly_activity[state].keys()))
    hours = range(24)
    matrix = np.zeros((len(activities), 24))
    
    for (hour, activity), count in hourly_activity[state].items():
        act_idx = activities.index(activity)
        matrix[act_idx, hour] = count
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(matrix, xticklabels=hours, yticklabels=activities, 
               cmap='YlOrRd', annot=False, fmt='g', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(f'{state} - Hourly Activity Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Activity Type', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'graphs_EN/hourly_heatmap_{state.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("✓ Analysis 4 completed\n")

# ==================== Analysis 5: Trip Purpose Distribution ====================
print("Running Analysis 5: Trip Purpose Distribution...")
purpose_counts = defaultdict(Counter)

for state, data in states_data.items():
    if isinstance(data, dict):
        for user_id, user_data in data.items():
            for segment in user_data['trajectory']:
                if 'home_based_type' in segment and segment['home_based_type']:
                    purpose_counts[state][segment['home_based_type']] += 1
    else:  # list
        for user_data in data:
            for segment in user_data['trajectory']:
                if 'home_based_type' in segment and segment['home_based_type']:
                    purpose_counts[state][segment['home_based_type']] += 1

# Create DataFrame
df_purpose = pd.DataFrame(purpose_counts).fillna(0)

# Plot
fig, ax = plt.subplots(figsize=(12, 7))
df_purpose.plot(kind='bar', ax=ax, width=0.8, color=sns.color_palette('Set3'))
ax.set_title('Trip Purpose Distribution by State (Home-Based Types)', fontsize=16, fontweight='bold')
ax.set_xlabel('Trip Purpose', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.legend(title='State', fontsize=10, loc='best')
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('graphs_EN/trip_purpose_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Analysis 5 completed\n")

# ==================== NEW Analysis 6: Arrival/Start Time for Different Activities ====================
print("Running Analysis 6: Arrival/Start Time Distribution...")
arrival_times = defaultdict(list)

# Collect start times for each activity type
for state, data in states_data.items():
    if isinstance(data, dict):
        for user_id, user_data in data.items():
            for segment in user_data['trajectory']:
                time_parts = segment['start_time'].split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                time_decimal = hour + minute / 60.0  # Convert to decimal hours
                arrival_times[segment['activity']].append(time_decimal)
    else:  # list
        for user_data in data:
            for segment in user_data['trajectory']:
                time_parts = segment['start_time'].split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                time_decimal = hour + minute / 60.0  # Convert to decimal hours
                arrival_times[segment['activity']].append(time_decimal)

# Create subplots for major activity types
major_activities = ['work', 'home', 'shopping', 'education', 'dine_out', 'socialize']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, activity in enumerate(major_activities):
    if activity in arrival_times and len(arrival_times[activity]) > 0:
        ax = axes[idx]
        times = arrival_times[activity]
        ax.hist(times, bins=48, color=sns.color_palette('tab10')[idx], 
               edgecolor='black', alpha=0.7)
        ax.set_title(f'{activity.capitalize()}\n(n={len(times):,} trips)', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Hour of Day', fontsize=10)
        ax.set_ylabel('Number of Arrivals', fontsize=10)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean line
        mean_time = np.mean(times)
        ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {int(mean_time):02d}:{int((mean_time % 1) * 60):02d}')
        ax.legend(fontsize=8)

plt.suptitle('Arrival/Start Time Distribution by Activity Type', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('graphs_EN/arrival_time_by_activity.png', dpi=300, bbox_inches='tight')
plt.close()

# Create overall heatmap
activities = sorted(arrival_times.keys())
hour_bins = range(25)  # 0-24 hours
matrix = np.zeros((len(activities), 24))

for act_idx, activity in enumerate(activities):
    times = arrival_times[activity]
    hist, _ = np.histogram(times, bins=hour_bins)
    matrix[act_idx, :] = hist

# Normalize by row for better visualization
matrix_norm = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-10)

fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(matrix_norm, xticklabels=range(24), yticklabels=activities,
           cmap='YlOrRd', annot=False, fmt='.2f', ax=ax, 
           cbar_kws={'label': 'Proportion of Daily Arrivals'})
ax.set_title('Activity Start Time Heatmap (Normalized by Activity)', fontsize=16, fontweight='bold')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Activity Type', fontsize=12)
plt.tight_layout()
plt.savefig('graphs_EN/arrival_time_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate statistics
stats = []
for activity, times in arrival_times.items():
    mean_time = np.mean(times)
    median_time = np.median(times)
    std_time = np.std(times)
    stats.append({
        'Activity': activity,
        'Count': len(times),
        'Mean Start Time': f"{int(mean_time):02d}:{int((mean_time % 1) * 60):02d}",
        'Median Start Time': f"{int(median_time):02d}:{int((median_time % 1) * 60):02d}",
        'Std Dev (hours)': f"{std_time:.2f}"
    })

df_arrival = pd.DataFrame(stats).sort_values('Count', ascending=False)
df_arrival.to_csv('arrival_time_statistics.csv', index=False)

print("✓ Analysis 6 completed\n")

# ==================== NEW Analysis 7: Activity Count per User ====================
print("Running Analysis 7: Activity Count per User...")
user_activity_counts = defaultdict(list)

# Count activities for each user in each state
for state, data in states_data.items():
    if isinstance(data, dict):
        for user_id, user_data in data.items():
            activity_count = len(user_data['trajectory'])
            user_activity_counts[state].append(activity_count)
    else:  # list
        for user_data in data:
            activity_count = len(user_data['trajectory'])
            user_activity_counts[state].append(activity_count)

# Create box plot comparison
fig, ax = plt.subplots(figsize=(12, 7))

data_to_plot = [user_activity_counts[state] for state in states_data.keys()]
bp = ax.boxplot(data_to_plot, labels=list(states_data.keys()), patch_artist=True)

# Color the boxes
colors = sns.color_palette('Set2', len(states_data))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('Activity Count per User by State', fontsize=16, fontweight='bold')
ax.set_xlabel('State', fontsize=12)
ax.set_ylabel('Number of Activities per User', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graphs_EN/activity_count_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Create histogram for each state
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (state, counts) in enumerate(user_activity_counts.items()):
    ax = axes[idx]
    ax.hist(counts, bins=30, color=sns.color_palette('Set2')[idx], 
           edgecolor='black', alpha=0.7)
    ax.set_title(f'{state}\nMean: {np.mean(counts):.1f} | Median: {np.median(counts):.0f}', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Activities per User', fontsize=10)
    ax.set_ylabel('Number of Users', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(np.mean(counts), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(np.median(counts), color='blue', linestyle='--', linewidth=2, label='Median')
    ax.legend(fontsize=8)

axes[5].axis('off')

plt.suptitle('Distribution of Activities per User by State', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('graphs_EN/activity_count_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Generate statistics
stats = []
for state, counts in user_activity_counts.items():
    stats.append({
        'State': state,
        'Total Users': len(counts),
        'Mean Activities': f"{np.mean(counts):.2f}",
        'Median Activities': int(np.median(counts)),
        'Min Activities': int(np.min(counts)),
        'Max Activities': int(np.max(counts)),
        'Std Dev': f"{np.std(counts):.2f}",
        '25th Percentile': int(np.percentile(counts, 25)),
        '75th Percentile': int(np.percentile(counts, 75))
    })

df_activity_count = pd.DataFrame(stats)
df_activity_count.to_csv('activity_count_per_user.csv', index=False)

# Create summary visualization
fig, ax = plt.subplots(figsize=(12, 7))

states = [s['State'] for s in stats]
means = [float(s['Mean Activities']) for s in stats]
stds = [float(s['Std Dev']) for s in stats]

bars = ax.bar(states, means, yerr=stds, capsize=10, 
              color=sns.color_palette('Set2', len(states)), 
              edgecolor='black', alpha=0.7)

# Add value labels on bars
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
            f'{mean:.1f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('Average Activities per User by State (with Std Dev)', fontsize=16, fontweight='bold')
ax.set_xlabel('State', fontsize=12)
ax.set_ylabel('Average Number of Activities', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('graphs_EN/activity_count_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Analysis 7 completed\n")

# ==================== Generate Analysis Report ====================
print("Generating comprehensive report...")

report = []
report.append("="*80)
report.append("ACTIVITY CHAIN ANALYSIS REPORT - NHTS 2017 MULTI-STATE COMPARISON")
report.append("="*80)
report.append("")

# Data overview
report.append("DATA OVERVIEW")
report.append("-" * 80)
total_users = sum(len(data) for data in states_data.values())
total_segments = 0
for data in states_data.values():
    if isinstance(data, dict):
        total_segments += sum(len(user_data['trajectory']) for user_data in data.values())
    else:  # list
        total_segments += sum(len(user_data['trajectory']) for user_data in data)

report.append(f"Total Number of States: {len(states_data)}")
report.append(f"Total Number of Users: {total_users:,}")
report.append(f"Total Number of Trajectory Segments: {total_segments:,}")
report.append("")

for state, data in states_data.items():
    user_count = len(data)
    if isinstance(data, dict):
        segment_count = sum(len(user_data['trajectory']) for user_data in data.values())
    else:  # list
        segment_count = sum(len(user_data['trajectory']) for user_data in data)
    report.append(f"  {state:12} - Users: {user_count:>7,} | Segments: {segment_count:>9,}")

report.append("")
report.append("="*80)
report.append("ANALYSIS OUTPUTS")
report.append("="*80)
report.append("")
report.append("Generated Visualizations:")
report.append("  1. Activity distribution by state (bar chart)")
report.append("  2. Activity duration analysis (bar chart)")
report.append("  3. Activity chain length distribution (histograms)")
report.append("  4. Hourly activity heatmaps (5 states)")
report.append("  5. Trip purpose distribution (bar chart)")
report.append("  6. Arrival/start time by activity (histograms + heatmap)")
report.append("  7. Activity count per user (box plot + histograms + summary)")
report.append("")
report.append("Generated Data Files:")
report.append("  - activity_distribution.csv")
report.append("  - activity_durations.csv")
report.append("  - sequence_lengths.csv")
report.append("  - arrival_time_statistics.csv")
report.append("  - activity_count_per_user.csv")
report.append("")
report.append("="*80)
report.append(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append("="*80)

report_text = "\n".join(report)

# Save report
with open('comprehensive_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)

print("\n" + "="*80)
print("✓ ALL ANALYSES COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nOutput location: {os.path.abspath('graphs_EN')}")
print("\nGenerated files:")
print("  - 15 PNG charts (300 DPI)")
print("  - 5 CSV data files")
print("  - 1 comprehensive report (TXT)")
print("\n" + "="*80)
