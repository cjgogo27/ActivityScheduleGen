import pandas as pd
import json

# 读取数据
print("正在读取数据...")
person_df = pd.read_csv(r'E:\mayue\FinalTraj\Georgia\2017-tsdc-nhts-georgia-add-on-download\data\survey_person.csv', low_memory=False)
value_lookup_df = pd.read_excel(r'E:\mayue\FinalTraj\Georgia\2017-tsdc-nhts-georgia-add-on-download\data\Value_Lookup.xlsx')

print(f"读取了 {len(person_df)} 条个人记录")

# ============================================================
# 个人数据筛选机制
# ============================================================
print("\n" + "="*60)
print("开始进行个人数据筛选...")
print("="*60)

initial_person_count = len(person_df)
initial_household_count = person_df['sampno'].nunique()
print(f"\n初始个人数量: {initial_person_count}")
print(f"初始家庭数量: {initial_household_count}")

# 定义需要过滤的字段和无效值
print("\n筛选社会经济数据不完整的个人...")
invalid_conditions = []

# R_RELAT: -8, -7
if 'r_relat' in person_df.columns:
    mask_r_relat = person_df['r_relat'].isin([-8, -7, -8.0, -7.0])
    invalid_conditions.append(mask_r_relat)
    print(f"  R_RELAT 无效值 (-8, -7): {mask_r_relat.sum()} 人")

# R_SEX: -9, -8, -7
if 'r_sex' in person_df.columns:
    mask_r_sex = person_df['r_sex'].isin([-9, -8, -7, -9.0, -8.0, -7.0])
    invalid_conditions.append(mask_r_sex)
    print(f"  R_SEX 无效值 (-9, -8, -7): {mask_r_sex.sum()} 人")

# OUTCNTRY: 01 (Yes - 表示出国，排除)
if 'outcntry' in person_df.columns:
    mask_outcntry = person_df['outcntry'].isin([1, 1.0, '01'])
    invalid_conditions.append(mask_outcntry)
    print(f"  OUTCNTRY = 1 (出国): {mask_outcntry.sum()} 人")

# PRMACT: -9, -8, -7, -1
if 'prmact' in person_df.columns:
    mask_prmact = person_df['prmact'].isin([-9, -8, -7, -1, -9.0, -8.0, -7.0, -1.0])
    invalid_conditions.append(mask_prmact)
    print(f"  PRMACT 无效值 (-9, -8, -7, -1): {mask_prmact.sum()} 人")

# OCCAT: -9, -8, -7 (移除了 -1)了 -1)
if 'occat' in person_df.columns:
    mask_occat = person_df['occat'].isin([-9, -8, -7, -9.0, -8.0, -7.0])
    invalid_conditions.append(mask_occat)
    print(f"  OCCAT 无效值 (-9, -8, -7): {mask_occat.sum()} 人")

# WORKER: -9 (移除了 -1)了 -1)
if 'worker' in person_df.columns:
    mask_worker = person_df['worker'].isin([-9, -9.0])
    invalid_conditions.append(mask_worker)
    print(f"  WORKER 无效值 (-9): {mask_worker.sum()} 人")

# # DISTTOWK17: -9
# if 'disttowk17' in person_df.columns:
#     mask_disttowk17 = person_df['disttowk17'].isin([-9, -9.0])
#     invalid_conditions.append(mask_disttowk17)
#     print(f"  DISTTOWK17 无效值 (-9): {mask_disttowk17.sum()} 人")

# 合并所有无效条件 (OR逻辑)
if invalid_conditions:
    combined_mask = invalid_conditions[0]
    for mask in invalid_conditions[1:]:
        combined_mask |= mask
    
    # 找出需要移除的个人和家庭
    invalid_persons = person_df[combined_mask]
    invalid_sampno_list = invalid_persons['sampno'].unique().tolist()
    
    print(f"\n需要移除的个人总数: {combined_mask.sum()}")
    print(f"受影响的家庭数量: {len(invalid_sampno_list)}")
    
    # 保存需要移除的 sampno 列表到文件，供 household 处理脚本使用
    invalid_sampno_file = r'E:\mayue\FinalTraj\Georgia\processed_data\invalid_sampno_from_person.json'
    with open(invalid_sampno_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_sampno_list, f, indent=2)
    print(f"已保存需要移除的家庭ID列表到: {invalid_sampno_file}")
    
    # 筛选掉这些无效的个人
    person_df = person_df[~combined_mask].copy()

final_person_count = len(person_df)
final_household_count = person_df['sampno'].nunique()

print("\n" + "="*60)
print(f"个人数据筛选完成！")
print(f"初始个人数量: {initial_person_count}")
print(f"最终个人数量: {final_person_count}")
print(f"移除个人数量: {initial_person_count - final_person_count}")
print(f"初始家庭数量: {initial_household_count}")
print(f"最终家庭数量: {final_household_count}")
print(f"移除家庭数量: {initial_household_count - final_household_count}")
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
agerange_lookup = create_lookup_dict('AGERANGE')
r_hisp_lookup = create_lookup_dict('R_HISP')
r_relat_lookup = create_lookup_dict('R_RELAT')
r_sex_lookup = create_lookup_dict('R_SEX')
r_race_lookup = create_lookup_dict('R_RACE')
educ_lookup = create_lookup_dict('EDUC')
worker_lookup = create_lookup_dict('WORKER')
outcntry_lookup = create_lookup_dict('OUTCNTRY')
wkstfips_lookup = create_lookup_dict('WKSTFIPS')
tddriver_lookup = create_lookup_dict('TDDRIVER')
wrk_home_lookup = create_lookup_dict('WRK_HOME')
wkftpt_lookup = create_lookup_dict('WKFTPT')
occat_lookup = create_lookup_dict('OCCAT')
prmact_lookup = create_lookup_dict('PRMACT')

print(f"年龄范围映射示例: {list(agerange_lookup.items())[:5]}")
print(f"性别映射: {r_sex_lookup}")
print(f"工作者状态映射: {worker_lookup}")

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

def format_age_range(value, lookup_dict):
    """格式化年龄范围"""
    label = get_lookup_value(value, lookup_dict, '')
    if label:
        return label
    return 'Not specified'

def process_person_static():
    """处理个人静态信息"""
    
    all_persons = []
    
    for idx, row in person_df.iterrows():
        # 创建用户ID
        sampno = safe_get_value(row, 'sampno', '')
        perno = safe_get_value(row, 'perno', '')
        
        if not sampno or not perno:
            continue
        
        user_id = f"{int(sampno)}_{int(perno)}"
        
        # 提取各个字段
        agerange = safe_get_value(row, 'agerange', None)
        r_hisp = safe_get_value(row, 'r_hisp', None)
        r_relat = safe_get_value(row, 'r_relat', None)
        r_sex = safe_get_value(row, 'r_sex', None)
        r_race = safe_get_value(row, 'r_race', None)
        educ = safe_get_value(row, 'educ', None)
        worker = safe_get_value(row, 'worker', None)
        outcntry = safe_get_value(row, 'outcntry', None)
        disttowk17 = safe_get_value(row, 'disttowk17', None)
        wkstfips = safe_get_value(row, 'wkstfips', None)
        tddriver = safe_get_value(row, 'tddriver', None)
        wrk_home = safe_get_value(row, 'wrk_home', None)
        wkftpt = safe_get_value(row, 'wkftpt', None)
        occat = safe_get_value(row, 'occat', None)
        prmact = safe_get_value(row, 'prmact', None)
        
        # 创建个人静态信息字典
        person_static = {
            'user_id': user_id,
            'age_range': format_age_range(agerange, agerange_lookup),
            'hispanic': get_lookup_value(r_hisp, r_hisp_lookup),
            'relationship': get_lookup_value(r_relat, r_relat_lookup),
            'gender': get_lookup_value(r_sex, r_sex_lookup),
            'race': get_lookup_value(r_race, r_race_lookup),
            'education': get_lookup_value(educ, educ_lookup),
            'employment_status': get_lookup_value(worker, worker_lookup),
            'traveled_abroad': get_lookup_value(outcntry, outcntry_lookup),
            'distance_to_work_miles': float(disttowk17) if pd.notna(disttowk17) and disttowk17 != -1 else None,
            'work_state': get_lookup_value(wkstfips, wkstfips_lookup),
            'driver_on_travel_day': get_lookup_value(tddriver, tddriver_lookup),
            'work_from_home': get_lookup_value(wrk_home, wrk_home_lookup),
            'work_schedule': get_lookup_value(wkftpt, wkftpt_lookup),
            'occupation': get_lookup_value(occat, occat_lookup),
            'primary_activity': get_lookup_value(prmact, prmact_lookup)
        }
        
        all_persons.append(person_static)
        
        # 进度提示
        if (idx + 1) % 5000 == 0:
            print(f"已处理 {idx + 1} 条记录...")
    
    return all_persons

# 处理数据
print("\n正在处理个人静态信息...")
result = process_person_static()

# 保存结果
output_file = r'E:\mayue\FinalTraj\Georgia\processed_data\georgia_person_static.json'
print(f"\n正在保存到 {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n处理完成！共处理了 {len(result)} 个人的静态信息")
print(f"结果已保存到: {output_file}")

# 显示前几个示例
print("\n前3个人的静态信息示例:")
for i, person in enumerate(result[:3]):
    print(f"\n示例 {i+1}:")
    print(json.dumps(person, indent=2, ensure_ascii=False))
