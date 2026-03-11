import pandas as pd
import json
from datetime import datetime, timedelta
from collections import defaultdict

# 读取数据
print("正在读取数据...")
trips_df = pd.read_csv(r'E:\mayue\FinalTraj\Georgia\2017-tsdc-nhts-georgia-add-on-download\data\survey_trip.csv', low_memory=False)
household_df = pd.read_csv(r'E:\mayue\FinalTraj\Georgia\2017-tsdc-nhts-georgia-add-on-download\data\survey_household.csv', low_memory=False)
value_lookup_df = pd.read_excel(r'E:\mayue\FinalTraj\Georgia\2017-tsdc-nhts-georgia-add-on-download\data\Value_Lookup.xlsx')

print(f"读取了 {len(trips_df)} 条出行记录")
print(f"读取了 {len(household_df)} 条家庭记录")

# ============================================================
# 轨迹数据筛选机制
# ============================================================
print("\n" + "="*60)
print("开始进行轨迹数据筛选...")
print("="*60)

initial_trips = len(trips_df)
initial_persons = trips_df['sampno'].astype(str) + '_' + trips_df['perno'].astype(str)
initial_person_count = initial_persons.nunique()
initial_household_count = trips_df['sampno'].nunique()

print(f"\n初始出行记录数: {initial_trips}")
print(f"初始个人数: {initial_person_count}")
print(f"初始家庭数: {initial_household_count}")

invalid_sampno_set = set()

# 1. 筛选出行时长 >= 120 分钟的
print("\n步骤 1: 筛选出行时长 (trvlcmin >= 120)...")
long_trips = trips_df[trips_df['trvlcmin'] >= 120]
if len(long_trips) > 0:
    invalid_persons_from_duration = long_trips['sampno'].astype(str) + '_' + long_trips['perno'].astype(str)
    invalid_sampno_from_duration = long_trips['sampno'].unique()
    before_count = len(invalid_sampno_set)
    invalid_sampno_set.update(invalid_sampno_from_duration)
    after_count = len(invalid_sampno_set)
    print(f"   找到 {len(long_trips)} 条超长出行记录")
    print(f"   涉及 {invalid_persons_from_duration.nunique()} 个人")
    print(f"   涉及 {len(invalid_sampno_from_duration)} 个家庭")
    print(f"   新增移除: {after_count - before_count} 个家庭 (累计: {after_count})")

# # 2. 筛选起点和终点是否为家 (whyfrom=1 for tripno=1, whyto=1 for last trip)
# print("\n步骤 2: 筛选起点和终点为家...")
trips_df['person_id'] = trips_df['sampno'].astype(str) + '_' + trips_df['perno'].astype(str)

# invalid_persons_not_home = set()
# for person_id, group in trips_df.groupby('person_id'):
#     sorted_group = group.sort_values('tripno')
    
#     # 检查第一个 trip 的 whyfrom 是否为 1
#     first_trip = sorted_group.iloc[0]
#     if first_trip['whyfrom'] != 1:
#         invalid_persons_not_home.add(person_id)
#         continue
    
#     # 检查最后一个 trip 的 whyto 是否为 1
#     last_trip = sorted_group.iloc[-1]
#     if last_trip['whyto'] != 1:
#         invalid_persons_not_home.add(person_id)

# if len(invalid_persons_not_home) > 0:
#     invalid_sampno_from_not_home = trips_df[trips_df['person_id'].isin(invalid_persons_not_home)]['sampno'].unique()
#     before_count = len(invalid_sampno_set)
#     invalid_sampno_set.update(invalid_sampno_from_not_home)
#     after_count = len(invalid_sampno_set)
#     print(f"   找到 {len(invalid_persons_not_home)} 个人未从家开始或结束")
#     print(f"   涉及 {len(invalid_sampno_from_not_home)} 个家庭")
#     print(f"   新增移除: {after_count - before_count} 个家庭 (累计: {after_count})")

# 3. 筛选未知的出行目的
print("\n步骤 3: 筛选未知的出行目的...")
invalid_whyfrom_codes = [-9, -8, -7, -1, 97]
invalid_whyto_codes = [-9, -8, -7, -1, 97]

invalid_trips_purpose = trips_df[
    (trips_df['whyfrom'].isin(invalid_whyfrom_codes)) | 
    (trips_df['whyto'].isin(invalid_whyto_codes))
]

if len(invalid_trips_purpose) > 0:
    invalid_persons_from_purpose = invalid_trips_purpose['person_id'].unique()
    invalid_sampno_from_purpose = invalid_trips_purpose['sampno'].unique()
    before_count = len(invalid_sampno_set)
    invalid_sampno_set.update(invalid_sampno_from_purpose)
    after_count = len(invalid_sampno_set)
    print(f"   找到 {len(invalid_trips_purpose)} 条出行目的未知的记录")
    print(f"   涉及 {len(invalid_persons_from_purpose)} 个人")
    print(f"   涉及 {len(invalid_sampno_from_purpose)} 个家庭")
    print(f"   新增移除: {after_count - before_count} 个家庭 (累计: {after_count})")

# 4. 筛选 full-time 工作者必须有 work 活动
print("\n步骤 4: 筛选 full-time 工作者必须有 work 活动...")
# 读取 person 数据获取工作状态
person_df = pd.read_csv(r'E:\mayue\FinalTraj\Georgia\2017-tsdc-nhts-georgia-add-on-download\data\survey_person.csv', low_memory=False)
person_df['person_id'] = person_df['sampno'].astype(str) + '_' + person_df['perno'].astype(str)

# 获取 full-time 工作者 (wkftpt == 1 表示 full-time)
fulltime_workers = person_df[person_df['wkftpt'] == 1]['person_id'].unique()
print(f"   找到 {len(fulltime_workers)} 个 full-time 工作者")

# 检查这些工作者的轨迹中是否有 work 活动 (whyto 为 2, 3, 4, 5)
work_activity_codes = [2, 3, 4, 5]
invalid_fulltime_workers = []

for person_id in fulltime_workers:
    person_trips = trips_df[trips_df['person_id'] == person_id]
    if len(person_trips) > 0:
        has_work = person_trips['whyto'].isin(work_activity_codes).any()
        if not has_work:
            invalid_fulltime_workers.append(person_id)

if len(invalid_fulltime_workers) > 0:
    invalid_sampno_from_fulltime = trips_df[trips_df['person_id'].isin(invalid_fulltime_workers)]['sampno'].unique()
    before_count = len(invalid_sampno_set)
    invalid_sampno_set.update(invalid_sampno_from_fulltime)
    after_count = len(invalid_sampno_set)
    print(f"   找到 {len(invalid_fulltime_workers)} 个 full-time 工作者没有 work 活动")
    print(f"   涉及 {len(invalid_sampno_from_fulltime)} 个家庭")
    print(f"   新增移除: {after_count - before_count} 个家庭 (累计: {after_count})")
else:
    print(f"   所有 full-time 工作者都有 work 活动")

# # 5. 筛选一家人轨迹完全相同的情况
# print("\n步骤 5: 筛选一家人轨迹完全相同的情况...")
# identical_trajectory_households = []

# for sampno, household_trips in trips_df.groupby('sampno'):
#     # 获取该家庭的所有成员
#     persons_in_household = household_trips['person_id'].unique()
    
#     # 如果家庭只有1个人，跳过
#     if len(persons_in_household) <= 1:
#         continue
    
#     # 为每个人创建轨迹字符串（按时间排序的 whyto 序列）
#     person_trajectories = {}
#     for person_id in persons_in_household:
#         person_trips_sorted = household_trips[household_trips['person_id'] == person_id].sort_values('tripno')
#         trajectory_str = '_'.join(person_trips_sorted['whyto'].astype(str).tolist())
#         person_trajectories[person_id] = trajectory_str
    
#     # 检查是否所有人的轨迹都相同
#     unique_trajectories = set(person_trajectories.values())
#     if len(unique_trajectories) == 1:
#         identical_trajectory_households.append(sampno)

# if len(identical_trajectory_households) > 0:
#     before_count = len(invalid_sampno_set)
#     invalid_sampno_set.update(identical_trajectory_households)
#     after_count = len(invalid_sampno_set)
#     print(f"   找到 {len(identical_trajectory_households)} 个家庭所有成员轨迹完全相同")
#     print(f"   新增移除: {after_count - before_count} 个家庭 (累计: {after_count})")
# else:
#     print(f"   没有发现家庭成员轨迹完全相同的情况")

# 汇总所有需要移除的家庭
invalid_sampno_list = sorted([int(x) for x in invalid_sampno_set])
print("\n" + "="*60)
print(f"筛选汇总:")
print(f"需要移除的家庭总数: {len(invalid_sampno_list)}")
print("="*60)

# 保存需要移除的 sampno 列表
invalid_sampno_file = r'E:\mayue\FinalTraj\Georgia\processed_data\invalid_sampno_from_trajectory.json'
with open(invalid_sampno_file, 'w', encoding='utf-8') as f:
    json.dump(invalid_sampno_list, f, indent=2)
print(f"已保存需要移除的家庭ID列表到: {invalid_sampno_file}")

# 过滤掉这些家庭的出行记录
trips_df = trips_df[~trips_df['sampno'].isin(invalid_sampno_list)].copy()

final_trips = len(trips_df)
final_person_count = trips_df['person_id'].nunique()
final_household_count = trips_df['sampno'].nunique()

print(f"\n筛选后统计:")
print(f"最终出行记录数: {final_trips} (移除了 {initial_trips - final_trips} 条)")
print(f"最终个人数: {final_person_count} (移除了 {initial_person_count - final_person_count} 个)")
print(f"最终家庭数: {final_household_count} (移除了 {initial_household_count - final_household_count} 个)")
print(f"家庭保留比例: {final_household_count/initial_household_count*100:.2f}%")
print("="*60 + "\n")
# ============================================================

# 创建 sampno 到 tdaydat2 的映射
sampno_to_date = dict(zip(household_df['sampno'], household_df['tdaydat2']))
print(f"创建了 {len(sampno_to_date)} 个家庭的日期映射")

# 创建 WHYFROM 和 WHYTO 的查找字典
def create_lookup_dict(variable_name):
    """从 Value_Lookup 表中创建编码到描述的映射字典"""
    subset = value_lookup_df[value_lookup_df['NAME'] == variable_name]
    if len(subset) == 0:
        return {}
    lookup = {}
    for _, row in subset.iterrows():
        try:
            code = row['VALUE']
            label = row['LABEL']
            if pd.notna(code) and pd.notna(label):
                # 尝试将 code 转换为整数
                try:
                    code = int(float(code))
                except:
                    pass
                lookup[code] = str(label).strip()
        except:
            continue
    return lookup

whyfrom_lookup = create_lookup_dict('WHYFROM')
whyto_lookup = create_lookup_dict('WHYTO')

print(f"WHYFROM 映射数量: {len(whyfrom_lookup)}")
print(f"WHYTO 映射数量: {len(whyto_lookup)}")

# 活动映射（将原始代码映射到新的活动类别）
# 格式: {原始代码: (活动名称, 活动代码)}
activity_mapping = {
    -9: ('unknown', 0),
    -8: ('unknown', 0),
    -7: ('unknown', 0),
    -1: ('unknown', 0),
    97: ('unknown', 0),
    1: ('home', 1),
    2: ('work', 2),
    3: ('work', 2),
    4: ('work', 2),
    5: ('work', 2),
    6: ('dropoff_pickup', 10),
    7: ('transportation_change', -1),  # 特殊处理：使用前一个活动
    8: ('education', 3),
    9: ('medical', 6),
    10: ('medical', 6),
    11: ('shopping', 4),
    12: ('service', 5),
    13: ('dine_out', 7),
    14: ('service', 5),
    15: ('socialize', 8),
    16: ('exercise', 9),
    17: ('socialize', 8),
    18: ('medical', 6),
    19: ('socialize', 8)
}

# home_based 映射（基于 trippurp）
def get_home_based(trippurp):
    """获取 home_based 类别"""
    if pd.isna(trippurp):
        return 'NHB'
    
    trippurp_str = str(trippurp).strip().upper()
    
    home_based_mapping = {
        'HBW': 'HBW',           # Home-based work
        'HBSHOP': 'HBSHOP',     # Home-based shopping
        'HBO': 'HBO',           # Home-based other
        'HBSOCREC': 'HBO',      # Home-based social/recreational
        'HBSCHOOL': 'HBSCHOOL', # Home-based school
        'NHB': 'NHB',           # Non-home-based
    }
    
    return home_based_mapping.get(trippurp_str, 'NHB')

def time_to_minutes(time_str):
    """将HHMM格式转换为分钟数"""
    if pd.isna(time_str):
        return None
    time_str = str(int(time_str)).zfill(4)
    hours = int(time_str[:2])
    minutes = int(time_str[2:])
    return hours * 60 + minutes

def minutes_to_time_str(minutes):
    """将分钟数转换为 HH:MM:SS 格式"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}:00"

def get_activity_label(activity_code, lookup_dict):
    """获取活动的描述标签"""
    return lookup_dict.get(activity_code, f"Unknown (code: {activity_code})")

def get_activity_mapped(original_code):
    """获取映射后的活动类别和代码
    返回: (activity_name, activity_code)
    """
    return activity_mapping.get(original_code, ('unknown', 0))

def process_trajectories():
    """处理轨迹数据 - 将断开的轨迹补成连续的"""
    
    # 创建 person_id
    trips_df['person_id'] = trips_df['sampno'].astype(str) + '_' + trips_df['perno'].astype(str)
    
    # 按人分组
    grouped = trips_df.groupby('person_id')
    
    all_users_data = []
    
    for person_id, group in grouped:
        # 按开始时间排序
        group = group.sort_values('strttime').reset_index(drop=True)
        
        # 获取该用户的 sampno 和对应的日期
        sampno = group.iloc[0]['sampno']
        travel_date = sampno_to_date.get(sampno, None)
        
        # 如果没有日期信息，跳过该用户
        if travel_date is None or pd.isna(travel_date):
            continue
        
        # 提取所有的原始出行片段
        trip_segments = []
        for idx, row in group.iterrows():
            start_time = time_to_minutes(row['strttime'])
            end_time = time_to_minutes(row['endtime'])
            whyfrom = int(row['whyfrom']) if not pd.isna(row['whyfrom']) else 1
            whyto = int(row['whyto']) if not pd.isna(row['whyto']) else 1
            trippurp = row['trippurp'] if 'trippurp' in row.index else 'NHB'
            
            if start_time is None or end_time is None:
                continue
            
            trip_segments.append({
                'start_minutes': start_time,
                'end_minutes': end_time,
                'whyfrom': whyfrom,
                'whyto': whyto,
                'trippurp': trippurp
            })
        
        if not trip_segments:
            continue
        
        # 构建连续的轨迹
        continuous_trajectories = []
        
        # 第一段：从 0:00 到第一个轨迹的 endtime
        first_segment = trip_segments[0]
        first_original_code = first_segment['whyfrom']
        first_activity_source = get_activity_label(first_original_code, whyfrom_lookup)
        first_activity_mapped, first_activity_code = get_activity_mapped(first_original_code)
        first_home_based = get_home_based(first_segment['trippurp'])
        
        # 处理第一段的 transportation_change 情况（使用 home 作为默认）
        if first_original_code == 7:
            first_activity_mapped, first_activity_code = ('home', 1)
        
        continuous_trajectories.append({
            'day_id': 1,
            'sequence_id': 1,
            'date': str(travel_date),
            'arrival_time': '00:00:00',
            'departure_time': minutes_to_time_str(first_segment['end_minutes']),
            'activity_code': first_activity_code,
            'activity_source': first_activity_source,
            'activity': first_activity_mapped,
            'home_based': first_home_based,
            'duration': first_segment['end_minutes'],
            'hour': 0
        })
        
        # 中间段：从上一个轨迹的 endtime 到下一个轨迹的 endtime
        for i in range(len(trip_segments) - 1):
            current_segment = trip_segments[i]
            next_segment = trip_segments[i + 1]
            
            # 使用当前段的 whyto 作为这段时间的 activity
            original_code = current_segment['whyto']
            activity_source = get_activity_label(original_code, whyto_lookup)
            activity_mapped, activity_code = get_activity_mapped(original_code)
            home_based = get_home_based(current_segment['trippurp'])
            
            # 特殊处理：如果是 transportation_change (07)，使用前一个轨迹的活动
            if original_code == 7:
                prev_traj = continuous_trajectories[-1]
                activity_mapped = prev_traj['activity']
                activity_code = prev_traj['activity_code']
            
            start_minutes = current_segment['end_minutes']
            end_minutes = next_segment['end_minutes']
            duration = end_minutes - start_minutes
            
            continuous_trajectories.append({
                'day_id': 1,
                'sequence_id': len(continuous_trajectories) + 1,
                'date': str(travel_date),
                'arrival_time': minutes_to_time_str(start_minutes),
                'departure_time': minutes_to_time_str(end_minutes),
                'activity_code': activity_code,
                'activity_source': activity_source,
                'activity': activity_mapped,
                'home_based': home_based,
                'duration': duration,
                'hour': start_minutes // 60
            })
        
        # 最后一段：从最后一个轨迹的 endtime 到 24:00
        last_segment = trip_segments[-1]
        last_original_code = last_segment['whyto']
        last_activity_source = get_activity_label(last_original_code, whyto_lookup)
        last_activity_mapped, last_activity_code = get_activity_mapped(last_original_code)
        last_home_based = get_home_based(last_segment['trippurp'])
        
        # 特殊处理：如果最后一段是 transportation_change (07)，使用前一个轨迹的活动
        if last_original_code == 7 and continuous_trajectories:
            prev_traj = continuous_trajectories[-1]
            last_activity_mapped = prev_traj['activity']
            last_activity_code = prev_traj['activity_code']
        
        # 只有当最后一段没有到达 24:00 时才添加
        if last_segment['end_minutes'] < 24 * 60:
            continuous_trajectories.append({
                'day_id': 1,
                'sequence_id': len(continuous_trajectories) + 1,
                'date': str(travel_date),
                'arrival_time': minutes_to_time_str(last_segment['end_minutes']),
                'departure_time': '24:00:00',
                'activity_code': last_activity_code,
                'activity_source': last_activity_source,
                'activity': last_activity_mapped,
                'home_based': last_home_based,
                'duration': 24 * 60 - last_segment['end_minutes'],
                'hour': last_segment['end_minutes'] // 60
            })
        
        user_data = {
            'user_id': person_id,
            'trajectories': continuous_trajectories
        }
        
        all_users_data.append(user_data)
    
    return all_users_data

# 处理数据
print("正在处理轨迹数据...")
result = process_trajectories()

# 保存结果
output_file = r'E:\mayue\FinalTraj\Georgia\processed_data\georgia_trajectories_processed.json'
print(f"正在保存到 {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"处理完成！共处理了 {len(result)} 个用户的轨迹数据")
print(f"结果已保存到: {output_file}")

# 显示前几个用户的统计信息
print("\n前5个用户的统计信息:")
for i, user in enumerate(result[:5]):
    print(f"\n用户 {user['user_id']}:")
    print(f"  轨迹数量: {len(user['trajectories'])}")
    if user['trajectories']:
        first = user['trajectories'][0]
        last = user['trajectories'][-1]
        print(f"  第一个活动: {first['activity']} ({first['activity_source']}) at {first['arrival_time']}")
        print(f"  最后一个活动: {last['activity']} ({last['activity_source']}) at {last['departure_time']}")

# 显示一个完整示例
if result:
    print("\n" + "="*130)
    print("完整示例 - 用户", result[1]['user_id'])
    print("="*130)
    if result[1]['trajectories']:
        print(f"日期: {result[1]['trajectories'][0]['date']}")
    print(f"{'序号':<5} {'到达':<12} {'离开':<12} {'代码':<5} {'映射前(activity_source)':<60} {'映射后':<20} {'home_based':<10} {'时长':<8}")
    print("-"*130)
    for traj in result[1]['trajectories']:
        source = traj['activity_source'][:58] if len(traj['activity_source']) > 58 else traj['activity_source']
        print(f"{traj['sequence_id']:<5} {traj['arrival_time']:<12} {traj['departure_time']:<12} {traj['activity_code']:<5} {source:<60} {traj['activity']:<20} {traj['home_based']:<10} {traj['duration']:<8.0f}")
    print("="*130)
