#!/usr/bin/env python3
"""
分析准确率低的用户 - 提取个人和家庭信息,分析原始轨迹合理性和预测失败原因
"""

import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import os

# 文件路径
PERSON_STATIC_FILE = r"/data/mayue/cjy/Other_method/FinalTraj/Oklahoma/processed_data/oklahoma_person_static.json"
HOUSEHOLD_STATIC_FILE = r"/data/mayue/cjy/Other_method/FinalTraj/Oklahoma/processed_data/oklahoma_household_static.json"
ORIGINAL_TRAJECTORIES_FILE = r"/data/mayue/cjy/Other_method/FinalTraj/Oklahoma/processed_data/all_user_schedules.json"
GENERATED_TRAJECTORIES_FILE = r"/data/mayue/cjy/Other_method/FinalTraj/Trajectory_Generation_multi_agent/output_trajectories/all_trajectories_20251211_231750.json"
OUTPUT_DIR = r"/data/mayue/cjy/Other_method/FinalTraj/evaluation/visualization_analysis_oklahoma_v3"

TIMESTEP_MINUTES = 15
N_TIMESTEPS = 96

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


def format_schedule(schedule):
    """格式化schedule为可读字符串"""
    parts = []
    for seg in schedule:
        parts.append(f"{seg['start_time']}-{seg['end_time']}: {seg['activity']}")
    return "; ".join(parts)


def analyze_trajectory_mismatch(gen_schedule, orig_schedule, person_info, household_info):
    """分析轨迹不匹配的原因"""
    issues = []
    
    # 1. 就业状态分析
    employment = person_info.get('employment_status', 'Unknown')
    work_schedule = person_info.get('work_schedule', 'Unknown')
    
    # 检查真实轨迹中是否有工作
    has_real_work = any(seg['activity'] == 'work' for seg in orig_schedule)
    has_gen_work = any(seg['activity'] == 'work' for seg in gen_schedule)
    
    # 分析工作相关问题
    if has_gen_work and not has_real_work:
        issues.append({
            'type': '错误生成工作',
            'detail': f"就业状态: {employment}, 但真实轨迹无工作活动",
            'severity': 'high'
        })
    elif has_real_work and not has_gen_work:
        issues.append({
            'type': '缺失工作活动',
            'detail': f"就业状态: {employment}, 但生成轨迹无工作活动",
            'severity': 'high'
        })
    
    # 2. 工作时间分析
    if has_real_work and has_gen_work:
        real_work_times = []
        gen_work_times = []
        
        for seg in orig_schedule:
            if seg['activity'] == 'work':
                real_work_times.append((seg['start_time'], seg['end_time']))
        
        for seg in gen_schedule:
            if seg['activity'] == 'work':
                gen_work_times.append((seg['start_time'], seg['end_time']))
        
        # 检查是否是夜班
        is_night_shift = any(
            time_to_minutes(start) >= 18*60 or time_to_minutes(start) < 6*60
            for start, _ in real_work_times
        )
        
        if is_night_shift:
            issues.append({
                'type': '夜班/非标准工作时间',
                'detail': f"真实工作时间: {real_work_times}, 工作时间表: {work_schedule}",
                'severity': 'high'
            })
        
        # 检查工作时间差异
        if real_work_times and gen_work_times:
            real_start = time_to_minutes(real_work_times[0][0])
            gen_start = time_to_minutes(gen_work_times[0][0])
            time_diff = abs(real_start - gen_start)
            
            if time_diff > 120:  # 超过2小时差异
                issues.append({
                    'type': '工作开始时间差异大',
                    'detail': f"真实: {real_work_times[0][0]}, 生成: {gen_work_times[0][0]}, 差异: {time_diff}分钟",
                    'severity': 'medium'
                })
    
    # 3. 活动复杂度分析
    real_activities = set(seg['activity'] for seg in orig_schedule)
    gen_activities = set(seg['activity'] for seg in gen_schedule)
    
    if len(real_activities) > len(gen_activities) + 2:
        issues.append({
            'type': '活动类型过于简化',
            'detail': f"真实活动类型: {len(real_activities)}, 生成活动类型: {len(gen_activities)}",
            'severity': 'low'
        })
    
    # 4. 家庭相关分析
    if household_info:
        household_size = household_info.get('household_size', 0)
        num_children = household_info.get('number_of_children', 0)
        
        # 检查是否有接送活动
        has_real_dropoff = any(seg['activity'] == 'dropoff_pickup' for seg in orig_schedule)
        has_gen_dropoff = any(seg['activity'] == 'dropoff_pickup' for seg in gen_schedule)
        
        if num_children > 0 and has_real_dropoff and not has_gen_dropoff:
            issues.append({
                'type': '缺失接送孩子活动',
                'detail': f"家庭有{num_children}个孩子,但未生成接送活动",
                'severity': 'medium'
            })
    
    # 5. 其他特殊活动分析
    special_activities = ['medical', 'education', 'dine_out', 'exercise']
    for activity in special_activities:
        has_real = any(seg['activity'] == activity for seg in orig_schedule)
        has_gen = any(seg['activity'] == activity for seg in gen_schedule)
        
        if has_real and not has_gen:
            issues.append({
                'type': f'缺失{activity}活动',
                'detail': f"真实轨迹有{activity},但生成轨迹没有",
                'severity': 'low'
            })
    
    return issues


def main():
    """主函数"""
    print("="*80)
    print("准确率低的用户深度分析")
    print("="*80)
    
    # 加载数据
    print("\n加载数据...")
    with open(PERSON_STATIC_FILE, 'r', encoding='utf-8') as f:
        person_data = json.load(f)
    person_dict = {p['user_id']: p for p in person_data}
    
    with open(HOUSEHOLD_STATIC_FILE, 'r', encoding='utf-8') as f:
        household_data = json.load(f)
    household_dict = {h['household_id']: h for h in household_data}
    
    with open(ORIGINAL_TRAJECTORIES_FILE, 'r', encoding='utf-8') as f:
        original_trajectories = json.load(f)
    original_dict = {t['user_id']: t['schedule'] for t in original_trajectories}
    
    with open(GENERATED_TRAJECTORIES_FILE, 'r', encoding='utf-8') as f:
        generated_trajectories = json.load(f)
    
    # 计算每个用户的准确率
    print("\n计算准确率...")
    user_accuracies = []
    
    for gen_item in generated_trajectories:
        user_id = gen_item['user_id']
        if user_id in original_dict:
            gen_seq = schedule_to_96_timesteps(gen_item['schedule'])
            tar_seq = schedule_to_96_timesteps(original_dict[user_id])
            accuracy = calculate_accuracy(gen_seq, tar_seq)
            user_accuracies.append({
                'user_id': user_id,
                'accuracy': accuracy,
                'gen_schedule': gen_item['schedule'],
                'orig_schedule': original_dict[user_id]
            })
    
    # 按准确率排序
    user_accuracies.sort(key=lambda x: x['accuracy'])
    
    # 选择准确率最低的10个用户
    low_accuracy_users = user_accuracies[:10]
    
    print(f"\n找到准确率最低的{len(low_accuracy_users)}个用户:")
    for item in low_accuracy_users:
        print(f"  {item['user_id']}: {item['accuracy']:.2%}")
    
    # 详细分析每个低准确率用户
    detailed_analysis = []
    
    for item in low_accuracy_users:
        user_id = item['user_id']
        person_info = person_dict.get(user_id, {})
        
        # 获取家庭信息
        household_id = person_info.get('household_id')
        household_info = household_dict.get(household_id, {}) if household_id else {}
        
        # 分析不匹配原因
        issues = analyze_trajectory_mismatch(
            item['gen_schedule'],
            item['orig_schedule'],
            person_info,
            household_info
        )
        
        analysis = {
            'user_id': user_id,
            'accuracy': item['accuracy'],
            'person_info': {
                'age_range': person_info.get('age_range'),
                'gender': person_info.get('gender'),
                'employment_status': person_info.get('employment_status'),
                'work_schedule': person_info.get('work_schedule'),
                'occupation': person_info.get('occupation'),
                'education': person_info.get('education'),
                'relationship': person_info.get('relationship'),
                'work_from_home': person_info.get('work_from_home'),
                'distance_to_work_miles': person_info.get('distance_to_work_miles'),
                'driver_on_travel_day': person_info.get('driver_on_travel_day')
            },
            'household_info': {
                'household_id': household_id,
                'household_size': household_info.get('household_size'),
                'number_of_children': household_info.get('number_of_children'),
                'household_income': household_info.get('household_income'),
                'number_of_vehicles': household_info.get('number_of_vehicles'),
                'home_ownership': household_info.get('home_ownership')
            },
            'trajectories': {
                'original': format_schedule(item['orig_schedule']),
                'generated': format_schedule(item['gen_schedule'])
            },
            'issues': issues
        }
        
        detailed_analysis.append(analysis)
    
    # 保存详细分析结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'low_accuracy_users_analysis.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 详细分析已保存到: {output_file}")
    
    # 生成可读报告
    report_file = os.path.join(OUTPUT_DIR, 'low_accuracy_users_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("准确率最低用户深度分析报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"分析用户数: {len(detailed_analysis)}\n\n")
        
        for idx, analysis in enumerate(detailed_analysis, 1):
            f.write("\n" + "="*80 + "\n")
            f.write(f"用户 {idx}: {analysis['user_id']}\n")
            f.write(f"准确率: {analysis['accuracy']:.2%}\n")
            f.write("="*80 + "\n\n")
            
            # 个人信息
            f.write("个人信息:\n")
            f.write("-"*80 + "\n")
            for key, value in analysis['person_info'].items():
                f.write(f"  {key}: {value}\n")
            
            # 家庭信息
            f.write("\n家庭信息:\n")
            f.write("-"*80 + "\n")
            for key, value in analysis['household_info'].items():
                f.write(f"  {key}: {value}\n")
            
            # 轨迹对比
            f.write("\n轨迹对比:\n")
            f.write("-"*80 + "\n")
            f.write(f"原始轨迹:\n  {analysis['trajectories']['original']}\n\n")
            f.write(f"生成轨迹:\n  {analysis['trajectories']['generated']}\n")
            
            # 问题分析
            f.write("\n问题分析:\n")
            f.write("-"*80 + "\n")
            if analysis['issues']:
                for issue in analysis['issues']:
                    f.write(f"  [{issue['severity'].upper()}] {issue['type']}\n")
                    f.write(f"    详情: {issue['detail']}\n\n")
            else:
                f.write("  未发现明显问题\n")
            
            # 合理性分析
            f.write("\n合理性分析:\n")
            f.write("-"*80 + "\n")
            
            # 基于个人信息判断原始轨迹的合理性
            employment = analysis['person_info']['employment_status']
            work_schedule = analysis['person_info']['work_schedule']
            orig_schedule = analysis['trajectories']['original']
            
            f.write(f"就业状态: {employment}\n")
            f.write(f"工作时间表: {work_schedule}\n\n")
            
            has_work = 'work' in orig_schedule
            
            if employment in ['Not in labor force', 'Unemployed'] and has_work:
                f.write("⚠ 异常: 就业状态显示未就业,但原始轨迹包含工作活动\n")
                f.write("  可能原因: 数据标注错误或兼职/临时工作未在就业状态中体现\n")
            elif employment == 'Employed' and not has_work:
                f.write("⚠ 异常: 就业状态显示已就业,但原始轨迹无工作活动\n")
                f.write("  可能原因: 休假日、在家办公、或数据采集当天未工作\n")
            else:
                f.write("✓ 就业状态与轨迹基本一致\n")
            
            # 分析为什么预测不准确
            f.write("\n预测失败原因分析:\n")
            f.write("-"*80 + "\n")
            
            if analysis['issues']:
                # 统计问题类型
                issue_types = defaultdict(int)
                for issue in analysis['issues']:
                    issue_types[issue['type']] += 1
                
                f.write("主要问题:\n")
                for issue_type, count in sorted(issue_types.items(), key=lambda x: -x[1]):
                    f.write(f"  - {issue_type} ({count}次)\n")
                
                # 给出改进建议
                f.write("\n改进建议:\n")
                
                if any('夜班' in issue['type'] for issue in analysis['issues']):
                    f.write("  1. 增强对非标准工作时间(夜班、轮班)的识别能力\n")
                    f.write("     - 在提示词中明确要求根据work_schedule字段判断工作时间\n")
                    f.write("     - 添加夜班工作的示例\n")
                
                if any('错误生成工作' in issue['type'] for issue in analysis['issues']):
                    f.write("  2. 强化就业状态的判断逻辑\n")
                    f.write("     - 明确告知LLM:如果employment_status为'Not in labor force'或'Unemployed',不应生成工作活动\n")
                
                if any('简化' in issue['type'] for issue in analysis['issues']):
                    f.write("  3. 提高活动多样性\n")
                    f.write("     - 增加对其他活动类型(购物、医疗、社交等)的生成概率\n")
                
                if any('接送' in issue['type'] for issue in analysis['issues']):
                    f.write("  4. 加强家庭信息的利用\n")
                    f.write("     - 如果家庭有孩子,提示可能有接送活动\n")
            else:
                f.write("未识别出明显的结构性问题,可能是时间细节上的差异\n")
            
            f.write("\n")
    
    print(f"✓ 可读报告已保存到: {report_file}")
    
    # 统计整体问题分布
    print("\n问题类型统计:")
    print("-"*80)
    
    all_issues = defaultdict(int)
    for analysis in detailed_analysis:
        for issue in analysis['issues']:
            all_issues[issue['type']] += 1
    
    for issue_type, count in sorted(all_issues.items(), key=lambda x: -x[1]):
        print(f"  {issue_type}: {count}次")
    
    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)


if __name__ == "__main__":
    main()
