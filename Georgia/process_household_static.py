import pandas as pd
import json
import holidays

# 读取数据
print("正在读取数据...")
household_df = pd.read_csv(r'E:\mayue\FinalTraj\Georgia\2017-tsdc-nhts-georgia-add-on-download\data\survey_household.csv', low_memory=False)
person_df = pd.read_csv(r'E:\mayue\FinalTraj\Georgia\2017-tsdc-nhts-georgia-add-on-download\data\survey_person.csv', low_memory=False)
trip_df = pd.read_csv(r'E:\mayue\FinalTraj\Georgia\2017-tsdc-nhts-georgia-add-on-download\data\survey_trip.csv', low_memory=False)
value_lookup_df = pd.read_excel(r'E:\mayue\FinalTraj\Georgia\2017-tsdc-nhts-georgia-add-on-download\data\Value_Lookup.xlsx')

print(f"读取了 {len(household_df)} 条家庭记录")
print(f"读取了 {len(person_df)} 条个人记录")
print(f"读取了 {len(trip_df)} 条行程记录")

# ============================================================
# 数据筛选机制
# ============================================================
print("\n" + "="*60)
print("开始进行数据筛选...")
print("="*60)

initial_count = len(household_df)
print(f"\n初始家庭数量: {initial_count}")

# 1. 计算每个家庭实际被调查的人数（三层一致性检查）
print("\n步骤 1: 三层一致性检查 (person表、trip表、household表的人数)...")

# 从 person 表中统计每个家庭的人数
person_count = person_df.groupby('sampno').size().reset_index(name='person_count')

# 从 trip 表中统计每个家庭的人数（基于 perno 字段）
trip_count = trip_df.groupby('sampno')['perno'].nunique().reset_index()
trip_count.columns = ['sampno', 'trip_person_count']

# 合并三个数据源的人数信息
household_df = household_df.merge(person_count, on='sampno', how='left')
household_df = household_df.merge(trip_count, on='sampno', how='left')
household_df['person_count'] = household_df['person_count'].fillna(0).astype(int)
household_df['trip_person_count'] = household_df['trip_person_count'].fillna(0).astype(int)

# 筛选: person表人数 == trip表人数 == household表hhsize（三层一致）
before_filter = len(household_df)
household_df = household_df[
    (household_df['person_count'] == household_df['hhsize']) &
    (household_df['trip_person_count'] == household_df['hhsize'])
]
after_filter = len(household_df)
print(f"   筛选前: {before_filter} 个家庭")
print(f"   筛选后: {after_filter} 个家庭 (移除了 {before_filter - after_filter} 个家庭)")
print(f"   一致性检查: person_count == trip_person_count == hhsize")

# # 2. 筛选社会经济数据完整的家庭
# print("\n步骤 2: 筛选社会经济数据完整的家庭 (移除 hhfaminc 为 -9, -8, -7 的家庭)...")
# before_filter = len(household_df)
# household_df = household_df[~household_df['hhfaminc'].isin([-9, -8, -7, -9.0, -8.0, -7.0])]
# after_filter = len(household_df)
# print(f"   筛选前: {before_filter} 个家庭")
# print(f"   筛选后: {after_filter} 个家庭 (移除了 {before_filter - after_filter} 个家庭)")

# 3. 只保留家庭规模 > 1 的家庭
print("\n步骤 3: 只保留家庭规模 > 1 的家庭...")
before_filter = len(household_df)
household_df = household_df[household_df['hhsize'] > 1]
after_filter = len(household_df)
print(f"   筛选前: {before_filter} 个家庭")
print(f"   筛选后: {after_filter} 个家庭 (移除了 {before_filter - after_filter} 个家庭)")

# 4. 只保留工作日出行的家庭 (travday 在 1-5)
print("\n步骤 4: 只保留工作日出行的家庭 (travday 在 1-5)...")
before_filter = len(household_df)
household_df = household_df[household_df['travday'].isin([1, 2, 3, 4, 5, 1.0, 2.0, 3.0, 4.0, 5.0])]
after_filter = len(household_df)
print(f"   筛选前: {before_filter} 个家庭")
print(f"   筛选后: {after_filter} 个家庭 (移除了 {before_filter - after_filter} 个家庭)")

# 5. 筛选节假日：移除出行日期在节假日的家庭
print("\n步骤 5: 筛选节假日 (移除出行日期在节假日的家庭)...")

# 使用 holidays 包自动获取美国联邦节假日（2016-2017年）
us_holidays = holidays.US(years=[2016, 2017])

# 转换 tdaydat2 为 datetime
household_df['tdaydat2'] = pd.to_datetime(household_df['tdaydat2'])

# 将 holidays 对象转换为 datetime 列表
us_holidays_dates = [pd.Timestamp(date) for date in us_holidays.keys()]

# 筛选：移除在节假日出行的家庭
before_filter = len(household_df)
household_df = household_df[~household_df['tdaydat2'].isin(us_holidays_dates)]
after_filter = len(household_df)
print(f"   筛选前: {before_filter} 个家庭")
print(f"   筛选后: {after_filter} 个家庭 (移除了 {before_filter - after_filter} 个家庭)")
print(f"   包含的节假日: {len(us_holidays)} 个美国联邦节假日")

# 6. 基于个人数据的筛选：移除包含无效个人数据的家庭
print("\n步骤 6: 基于个人数据筛选家庭 (移除成员有无效社会经济数据的家庭)...")
invalid_sampno_file = r'E:\mayue\FinalTraj\Georgia\processed_data\invalid_sampno_from_person.json'
try:
    # 尝试读取从 person 处理脚本生成的无效家庭列表
    if pd.io.common.file_exists(invalid_sampno_file):
        with open(invalid_sampno_file, 'r', encoding='utf-8') as f:
            invalid_sampno_list = json.load(f)
        
        before_filter = len(household_df)
        household_df = household_df[~household_df['sampno'].isin(invalid_sampno_list)]
        after_filter = len(household_df)
        print(f"   从 person 数据中识别出 {len(invalid_sampno_list)} 个需要移除的家庭")
        print(f"   筛选前: {before_filter} 个家庭")
        print(f"   筛选后: {after_filter} 个家庭 (移除了 {before_filter - after_filter} 个家庭)")
    else:
        print(f"   警告: 未找到个人筛选结果文件，请先运行 process_person_static.py")
        print(f"   跳过此筛选步骤...")
except Exception as e:
    print(f"   警告: 读取个人筛选结果时出错: {e}")
    print(f"   跳过此筛选步骤...")

# 7. 基于轨迹数据的筛选：移除有无效轨迹的家庭
print("\n步骤 7: 基于轨迹数据筛选家庭 (移除轨迹不符合要求的家庭)...")
invalid_sampno_trajectory_file = r'E:\mayue\FinalTraj\Georgia\processed_data\invalid_sampno_from_trajectory.json'
try:
    # 尝试读取从 trajectory 处理脚本生成的无效家庭列表
    if pd.io.common.file_exists(invalid_sampno_trajectory_file):
        with open(invalid_sampno_trajectory_file, 'r', encoding='utf-8') as f:
            invalid_sampno_trajectory_list = json.load(f)
        
        before_filter = len(household_df)
        household_df = household_df[~household_df['sampno'].isin(invalid_sampno_trajectory_list)]
        after_filter = len(household_df)
        print(f"   从 trajectory 数据中识别出 {len(invalid_sampno_trajectory_list)} 个需要移除的家庭")
        print(f"   筛选前: {before_filter} 个家庭")
        print(f"   筛选后: {after_filter} 个家庭 (移除了 {before_filter - after_filter} 个家庭)")
    else:
        print(f"   警告: 未找到轨迹筛选结果文件，请先运行 process_trajectories.py")
        print(f"   跳过此筛选步骤...")
except Exception as e:
    print(f"   警告: 读取轨迹筛选结果时出错: {e}")
    print(f"   跳过此筛选步骤...")

print("\n" + "="*60)
print(f"筛选完成！")
print(f"初始家庭数量: {initial_count}")
print(f"最终家庭数量: {len(household_df)}")
print(f"总共移除: {initial_count - len(household_df)} 个家庭")
print(f"保留比例: {len(household_df)/initial_count*100:.2f}%")
print("="*60 + "\n")
# ============================================================

# 创建查找字典，从 Value_Lookup 表中获取编码含义
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

# 创建各个字段的查找字典
print("正在创建查找字典...")
homeown_lookup = create_lookup_dict('HOMEOWN')
hhfaminc_lookup = create_lookup_dict('HHFAMINC')
msasize_lookup = create_lookup_dict('MSASIZE')
urban_lookup = create_lookup_dict('URBAN')
hh_race_lookup = create_lookup_dict('HH_RACE')
hh_hisp_lookup = create_lookup_dict('HH_HISP')
hhstfips_lookup = create_lookup_dict('HHSTFIPS')

print(f"住房拥有情况映射: {homeown_lookup}")
print(f"家庭收入映射示例: {list(hhfaminc_lookup.items())[:5]}")
print(f"城市区域映射: {urban_lookup}")

def safe_get_value(row, column_name, default=''):
    """安全获取列值"""
    try:
        if column_name.lower() not in row.index:
            return default
        value = row[column_name.lower()]
        if pd.isna(value):
            return default
        return value
    except:
        return default

def get_lookup_value(value, lookup_dict, default='Not specified'):
    """从查找字典中获取描述"""
    if pd.isna(value):
        return default
    
    # 尝试转换为整数
    try:
        value = int(float(value))
    except:
        pass
    
    return lookup_dict.get(value, default)

def safe_int(value):
    """安全转换为整数"""
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except:
        return None

def process_household_static():
    """处理家庭静态信息"""
    
    all_households = []
    
    for idx, row in household_df.iterrows():
        # 获取家庭ID
        sampno = safe_get_value(row, 'sampno', None)
        
        if sampno is None or pd.isna(sampno):
            continue
        
        household_id = str(int(sampno))
        
        # 提取各个字段
        homeown = safe_get_value(row, 'homeown', None)
        hhsize = safe_get_value(row, 'hhsize', None)
        hhvehcnt = safe_get_value(row, 'hhvehcnt', None)
        hhfaminc = safe_get_value(row, 'hhfaminc', None)
        drvrcnt = safe_get_value(row, 'drvrcnt', None)
        numadlt = safe_get_value(row, 'numadlt', None)
        youngchild = safe_get_value(row, 'youngchild', None)
        msasize = safe_get_value(row, 'msasize', None)
        urban = safe_get_value(row, 'urban', None)
        hh_race = safe_get_value(row, 'hh_race', None)
        hh_hisp = safe_get_value(row, 'hh_hisp', None)
        hhstfips = safe_get_value(row, 'hhstfips', None)
        
        # 创建家庭静态信息字典
        household_static = {
            'household_id': household_id,
            'home_ownership': get_lookup_value(homeown, homeown_lookup),
            'household_size': safe_int(hhsize),
            'vehicle_count': safe_int(hhvehcnt),
            'household_income': get_lookup_value(hhfaminc, hhfaminc_lookup),
            'driver_count': safe_int(drvrcnt),
            'adult_count': safe_int(numadlt),
            'young_children_count': safe_int(youngchild),
            'msa_size': get_lookup_value(msasize, msasize_lookup),
            'urban_area': get_lookup_value(urban, urban_lookup),
            'household_race': get_lookup_value(hh_race, hh_race_lookup),
            'household_hispanic': get_lookup_value(hh_hisp, hh_hisp_lookup),
            'state': get_lookup_value(hhstfips, hhstfips_lookup)
        }
        
        all_households.append(household_static)
        
        # 进度提示
        if (idx + 1) % 5000 == 0:
            print(f"已处理 {idx + 1} 条记录...")
    
    return all_households

# 处理数据
print("\n正在处理家庭静态信息...")
result = process_household_static()

# 保存结果
output_file = r'E:\mayue\FinalTraj\Georgia\processed_data\georgia_household_static.json'
print(f"\n正在保存到 {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n处理完成！共处理了 {len(result)} 个家庭的静态信息")
print(f"结果已保存到: {output_file}")

# 显示前几个示例
print("\n前3个家庭的静态信息示例:")
for i, household in enumerate(result[:3]):
    print(f"\n示例 {i+1}:")
    print(json.dumps(household, indent=2, ensure_ascii=False))
