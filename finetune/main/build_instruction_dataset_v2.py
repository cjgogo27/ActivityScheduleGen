"""
构建用于轨迹生成的指令微调数据集

核心逻辑:
1. 从household文件提取家庭ID (只处理有家庭的用户)
2. 根据household_id匹配person文件中的user_id (格式: household_id_person_num)
3. 从all_user_schedules.json获取真实轨迹作为输出
4. 使用与generate_trajectories_multiagent_negotiation.py一致的prompt模板

数据流程:
household (family info) -> person (individual info) -> schedule (trajectory output)

为什么这样做:
- 只处理有家庭的个人: 因为我们的任务是家庭轨迹生成,不属于家庭的个人不在研究范围内
- 从household出发: 保证数据的完整性和一致性
- 使用真实轨迹作为监督信号: 让模型学习真实的人类行为模式
"""

import json
import random
from typing import Dict, List, Any
from pathlib import Path
from collections import defaultdict

# 数据路径配置
PERSON_FILE = "/data/mayue/cjy/Other_method/FinalTraj/California/processed_data/california_person_static.json"
HOUSEHOLD_FILE = "/data/mayue/cjy/Other_method/FinalTraj/California/processed_data/california_household_static.json"
SCHEDULE_FILE = "/data/mayue/cjy/Other_method/FinalTraj/California/processed_data/all_user_schedules.json"
OUTPUT_FILE = "/data/mayue/cjy/Other_method/FinalTraj/finetune/trajectory_instruction_dataset.json"

# 活动类型 - 与原始脚本保持一致
ALLOWED_ACTIVITIES = {
    "home", "work", "education", "shopping", "service", 
    "medical", "dine_out", "socialize", "exercise", "dropoff_pickup"
}


def load_json_data(file_path: str) -> List[Dict]:
    """加载JSON数据"""
    print(f"正在加载: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  ✓ 加载了 {len(data)} 条记录")
    return data


def extract_household_id_from_user_id(user_id: str) -> str:
    """
    从user_id提取household_id
    格式: household_id_person_num (例如: 30150869_1 -> 30150869)
    """
    parts = user_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return None


def build_household_to_persons_mapping(person_data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    构建 household_id -> [person_info, ...] 的映射
    只保留属于家庭的个人 (user_id格式为 household_id_X)
    
    为什么这样做:
    - 过滤掉不属于家庭的个人 (user_id不是household_id_X格式的)
    - 这些人可能是数据中的异常值或不在家庭研究范围内
    """
    household_to_persons = defaultdict(list)
    skipped_no_household = 0
    
    for person in person_data:
        user_id = person.get('user_id', '')
        household_id = extract_household_id_from_user_id(user_id)
        
        if household_id:
            household_to_persons[household_id].append(person)
        else:
            skipped_no_household += 1
    
    print(f"  ✓ 找到 {len(household_to_persons)} 个家庭")
    print(f"  ✓ 跳过 {skipped_no_household} 个不属于家庭的个人")
    return dict(household_to_persons)


def build_schedule_mapping(schedule_data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    构建 user_id -> schedule 的映射
    """
    schedule_map = {}
    for item in schedule_data:
        user_id = item.get('user_id', '')
        schedule = item.get('schedule', [])
        if user_id and schedule:
            schedule_map[user_id] = schedule
    
    print(f"  ✓ 找到 {len(schedule_map)} 个用户的轨迹数据")
    return schedule_map


def validate_schedule(schedule: List[Dict]) -> bool:
    """
    验证轨迹是否有效:
    1. 必须包含至少一个活动
    2. 活动类型必须在允许列表中
    3. 必须有时间信息
    """
    if not schedule or len(schedule) == 0:
        return False
    
    for activity in schedule:
        if activity.get('activity') not in ALLOWED_ACTIVITIES:
            return False
        if not activity.get('start_time') or not activity.get('end_time'):
            return False
    
    return True


def create_trajectory_generation_instruction(
    person_info: Dict, 
    household_info: Dict, 
    schedule: List[Dict],
    other_members: List[Dict] = None
) -> Dict:
    """
    创建完整的轨迹生成指令
    
    参考generate_trajectories_multiagent_negotiation.py的核心prompt:
    - 输入: 个人信息 + 家庭信息 (作为context)
    - 输出: 完整的24小时轨迹schedule
    
    这是核心任务: 让模型学习根据个人和家庭特征生成真实轨迹
    
    为什么包含家庭信息:
    - 家庭特征会影响个人行为 (有小孩的家庭需要接送,高收入家庭可能有更多社交活动等)
    - 这与原始脚本的多智能体协商机制一致
    """
    # 构建输入信息
    user_id = person_info.get('user_id', 'Unknown')
    
    # 个人基本信息
    person_profile = f"""PERSON PROFILE (User ID: {user_id}):
- Relationship: {person_info.get('relationship', 'Unknown')}
- Age: {person_info.get('age_range', 'Unknown')}
- Gender: {person_info.get('gender', 'Unknown')}
- Race: {person_info.get('race', 'Unknown')}
- Hispanic: {person_info.get('hispanic', 'Unknown')}
- Education: {person_info.get('education', 'Unknown')}

EMPLOYMENT & WORK:
- Employment status: {person_info.get('employment_status', 'Unknown')}
- Work schedule: {person_info.get('work_schedule', 'Unknown')}
- Work from home: {person_info.get('work_from_home', 'Unknown')}
- Occupation: {person_info.get('occupation', 'Unknown')}
- Primary activity: {person_info.get('primary_activity', 'Unknown')}
- Distance to work: {person_info.get('distance_to_work_miles', 0)} miles
- Work state: {person_info.get('work_state', 'Unknown')}

MOBILITY:
- Driver on travel day: {person_info.get('driver_on_travel_day', 'Unknown')}
- Traveled abroad: {person_info.get('traveled_abroad', 'Unknown')}"""
    
    # 家庭信息
    household_profile = f"""HOUSEHOLD INFORMATION:
- Household ID: {household_info.get('household_id', 'Unknown')}
- Household size: {household_info.get('household_size', 0)} people
- Home ownership: {household_info.get('home_ownership', 'Unknown')}
- Household income: {household_info.get('household_income', 'Unknown')}
- Vehicle count: {household_info.get('vehicle_count', 0)}
- Driver count: {household_info.get('driver_count', 0)}
- Adult count: {household_info.get('adult_count', 0)}
- Young children: {household_info.get('young_children_count', 0)}
- Urban area: {household_info.get('urban_area', 'Unknown')}
- MSA size: {household_info.get('msa_size', 'Unknown')}
- Household race: {household_info.get('household_race', 'Unknown')}
- State: {household_info.get('state', 'Unknown')}"""
    
    # 家庭成员信息 (如果有)
    members_info = ""
    if other_members:
        members_info = "\nOTHER HOUSEHOLD MEMBERS:\n"
        for idx, member in enumerate(other_members, 1):
            members_info += f"  Member {idx}: {member.get('relationship', 'Unknown')}, "
            members_info += f"{member.get('age_range', 'Unknown')}, "
            members_info += f"Employment: {member.get('employment_status', 'Unknown')}\n"
    
    # 核心指令 - 参考原始脚本的prompt风格
    instruction = f"""Create a REALISTIC and COMPLETE 24-hour daily schedule for this person.

{person_profile}

{household_profile}{members_info}

TASK: Generate a realistic 24-hour trajectory that:
1. Covers 00:00-24:00 continuously with NO gaps
2. Reflects the person's employment status and work schedule
3. Considers household characteristics (children, income, location)
4. Includes realistic activity durations and transitions
5. Uses realistic times with minutes (e.g., 07:30, 16:45)

CRITICAL RULES FOR REALISM:
- Full-time workers: Typically 8-9 hours of work
- Part-time workers: 3-4 hours MAX
- Vary work start times (don't always use 07:00 or 08:00)
- Keep schedule SIMPLE (3-5 activities total is realistic)
- Home activities should be continuous blocks

ALLOWED ACTIVITIES: home, work, education, shopping, service, medical, dine_out, socialize, exercise, dropoff_pickup

OUTPUT FORMAT (JSON only, no explanation):
```json
{{
  "schedule": [
    {{"activity": "home", "start_time": "00:00", "end_time": "07:30"}},
    {{"activity": "work", "start_time": "07:30", "end_time": "16:45"}},
    {{"activity": "shopping", "start_time": "16:45", "end_time": "17:30"}},
    {{"activity": "home", "start_time": "17:30", "end_time": "24:00"}}
  ]
}}
```"""
    
    # 输出 - 真实的轨迹数据
    output_schedule = []
    for activity in schedule:
        output_schedule.append({
            "activity": activity["activity"],
            "start_time": activity["start_time"],
            "end_time": activity["end_time"]
        })
    
    output = json.dumps({"schedule": output_schedule}, ensure_ascii=False, indent=2)
    
    return {
        "instruction": instruction,
        "input": "",  # 所有信息都在instruction中
        "output": output
    }


def build_instruction_dataset() -> List[Dict]:
    """
    构建完整的指令数据集
    
    流程:
    1. 加载household数据 (作为主数据源)
    2. 对每个household,找到所有属于该家庭的person
    3. 对每个person,获取其真实轨迹
    4. 生成instruction-input-output格式的训练样本
    
    为什么从household出发:
    - 保证只处理有家庭的用户
    - 可以获取完整的家庭context信息
    - 与实际应用场景一致 (家庭单位的轨迹生成)
    """
    print("=" * 60)
    print("开始构建轨迹生成指令数据集")
    print("=" * 60)
    
    # 1. 加载所有数据
    print("\n[步骤1] 加载数据文件...")
    household_data = load_json_data(HOUSEHOLD_FILE)
    person_data = load_json_data(PERSON_FILE)
    schedule_data = load_json_data(SCHEDULE_FILE)
    
    # 2. 构建映射关系
    print("\n[步骤2] 构建数据映射关系...")
    household_to_persons = build_household_to_persons_mapping(person_data)
    schedule_map = build_schedule_mapping(schedule_data)
    
    # 3. 生成训练样本
    print("\n[步骤3] 生成训练样本...")
    instruction_dataset = []
    valid_households = 0
    valid_persons = 0
    skipped_no_schedule = 0
    skipped_invalid_schedule = 0
    
    for household in household_data:
        household_id = household.get('household_id', '')
        
        # 获取该家庭的所有成员
        persons_in_household = household_to_persons.get(household_id, [])
        
        if not persons_in_household:
            # 这个家庭在person文件中没有对应的成员 (可能是数据不一致)
            continue
        
        valid_households += 1
        
        # 为每个家庭成员生成训练样本
        for person in persons_in_household:
            user_id = person.get('user_id', '')
            
            # 获取该用户的轨迹
            schedule = schedule_map.get(user_id)
            
            if not schedule:
                skipped_no_schedule += 1
                continue
            
            # 验证轨迹有效性
            if not validate_schedule(schedule):
                skipped_invalid_schedule += 1
                continue
            
            # 获取其他家庭成员信息 (用于context)
            other_members = [p for p in persons_in_household if p['user_id'] != user_id]
            
            # 生成训练样本
            instruction_sample = create_trajectory_generation_instruction(
                person_info=person,
                household_info=household,
                schedule=schedule,
                other_members=other_members[:3]  # 最多包含3个其他成员信息
            )
            
            instruction_dataset.append(instruction_sample)
            valid_persons += 1
        
        # 进度显示
        if valid_households % 100 == 0:
            print(f"  处理进度: {valid_households} 个家庭, {valid_persons} 个有效样本")
    
    # 4. 统计信息
    print("\n" + "=" * 60)
    print("数据集构建完成!")
    print("=" * 60)
    print(f"总家庭数: {len(household_data)}")
    print(f"有成员的家庭数: {valid_households}")
    print(f"生成的训练样本数: {len(instruction_dataset)}")
    print(f"跳过(无轨迹): {skipped_no_schedule}")
    print(f"跳过(无效轨迹): {skipped_invalid_schedule}")
    print("=" * 60)
    
    return instruction_dataset


def save_dataset(dataset: List[Dict], output_file: str):
    """保存数据集到文件"""
    print(f"\n保存数据集到: {output_file}")
    
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 已保存 {len(dataset)} 条训练样本")
    
    # 显示一个示例
    if dataset:
        print("\n" + "=" * 60)
        print("示例数据 (第1条):")
        print("=" * 60)
        print(f"Instruction (前500字符):\n{dataset[0]['instruction'][:500]}...")
        print(f"\nOutput (前300字符):\n{dataset[0]['output'][:300]}...")
        print("=" * 60)


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("加州轨迹数据指令集构建工具")
    print("用于Llama模型微调")
    print("=" * 60)
    
    # 构建数据集
    dataset = build_instruction_dataset()
    
    if not dataset:
        print("\n❌ 错误: 没有生成任何训练样本!")
        return
    
    # 保存数据集
    save_dataset(dataset, OUTPUT_FILE)
    
    print("\n✓ 全部完成!")
    print(f"训练数据已保存到: {OUTPUT_FILE}")
    print(f"可以使用该数据集进行Llama微调")


if __name__ == "__main__":
    main()
