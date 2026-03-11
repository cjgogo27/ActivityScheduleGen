import json
import openai
import time
import os
from datetime import datetime

# ==================== 配置区域 ====================
# 运行模式选择：
# "num_users" - 按指定数量生成（原模式）
# "user_file" - 从user_id.json文件读取指定用户ID
GENERATION_MODE = "user_file"  # 可选: "num_users" 或 "user_file"

# num_users模式配置（仅当GENERATION_MODE="num_users"时有效）
NUM_USERS = 2

# user_file模式配置（仅当GENERATION_MODE="user_file"时有效）
USER_ID_FILE = "E:/mayue/FinalTraj/Trajectory_Generation_multi_agent/user_id/user_id_20251211_163143.json"
# ==================================================

# 设置OpenAI客户端
def create_openai_client():
    """创建OpenAI客户端"""
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),  # set OPENAI_API_KEY environment variable
        # base_url="https://api.siliconflow.cn/v1",
        timeout=30.0
    )
    return client

# 读取user_id文件
def read_user_ids(file_path):
    """从JSON文件读取user_id列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            user_ids = json.load(f)
        if not isinstance(user_ids, list):
            print(f"错误: user_id文件格式不正确，应该是数组")
            return []
        print(f"成功读取 {len(user_ids)} 个用户ID")
        return user_ids
    except Exception as e:
        print(f"读取user_id文件失败: {e}")
        return []

# 读取California person static信息
def read_california_static_info(file_path):
    """读取California个人静态信息JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功读取 {len(data)} 条个人静态信息")
        return data
    except Exception as e:
        print(f"读取文件失败: {e}")
        return []

# 处理用户信息，适配California数据格式
def extract_user_info(person_data):
    """从California person static数据中提取所有关键信息"""
    user_info = {
        'user_id': person_data.get('user_id', 'Unknown'),
        'age_range': person_data.get('age_range', 'Unknown'),
        'hispanic': person_data.get('hispanic', 'Unknown'),
        'relationship': person_data.get('relationship', 'Unknown'),
        'gender': person_data.get('gender', 'Unknown'),
        'race': person_data.get('race', 'Unknown'),
        'education': person_data.get('education', 'Unknown'),
        'employment_status': person_data.get('employment_status', 'Unknown'),
        'traveled_abroad': person_data.get('traveled_abroad', 'Unknown'),
        'distance_to_work_miles': person_data.get('distance_to_work_miles', 0),
        'work_state': person_data.get('work_state', 'Unknown'),
        'driver_on_travel_day': person_data.get('driver_on_travel_day', 'Unknown'),
        'work_from_home': person_data.get('work_from_home', 'Unknown'),
        'work_schedule': person_data.get('work_schedule', 'Unknown'),
        'occupation': person_data.get('occupation', 'Unknown'),
        'primary_activity': person_data.get('primary_activity', 'Unknown')
    }
    return user_info

# 生成用户轨迹（带重试机制）
def generate_user_trajectory_with_retry(client, user_info, max_retries=3):
    """使用OpenAI API生成24小时活动轨迹"""
    retries = 0
    
    while retries < max_retries:
        try:
            # 提取用户所有静态信息（16个维度）
            user_id = user_info['user_id']
            age_range = user_info['age_range']
            hispanic = user_info['hispanic']
            relationship = user_info['relationship']
            gender = user_info['gender']
            race = user_info['race']
            education = user_info['education']
            employment = user_info['employment_status']
            traveled_abroad = user_info['traveled_abroad']
            distance_to_work = user_info['distance_to_work_miles']
            work_state = user_info['work_state']
            is_driver = user_info['driver_on_travel_day']
            work_from_home = user_info['work_from_home']
            work_schedule = user_info['work_schedule']
            occupation = user_info['occupation']
            primary_activity = user_info['primary_activity']
            
            # 构建系统提示
            system_prompt = """You are a human behavior analysis expert specializing in generating realistic daily activity patterns. 
Your task is to generate a COMPLETE 24-hour activity trajectory with ABSOLUTELY NO MISSING TIME PERIODS.
Times do NOT need to be on the hour - use realistic times like 7:23, 8:47, etc."""
            
            # 构建用户提示 - 使用所有16个静态信息维度
            user_prompt = f"""
User Information (California NHTS 2017 Data - Complete Profile):
- User ID: {user_id}
- Age Range: {age_range}
- Hispanic: {hispanic}
- Relationship: {relationship}
- Gender: {gender}
- Race: {race}
- Education: {education}
- Employment Status: {employment}
- Traveled Abroad: {traveled_abroad}
- Distance to Work: {distance_to_work} miles
- Work State: {work_state}
- Driver on Travel Day: {is_driver}
- Work From Home: {work_from_home}
- Work Schedule: {work_schedule}
- Occupation: {occupation}
- Primary Activity: {primary_activity}

CRITICAL REQUIREMENTS:
1. Generate a COMPLETE 24-hour activity trajectory (00:00 to 24:00)
2. NO MISSING TIME PERIODS - every minute must be accounted for
3. Use ONLY these 10 activity types (NO OTHER ACTIVITIES ALLOWED):
   • home - at home (sleeping, resting, meals at home, leisure at home)
   • work - at workplace (includes commuting time - use home before and after)
   • education - at school/university
   • shopping - shopping activities
   • service - personal services (bank, government, etc.)
   • medical - medical appointments, healthcare
   • dine_out - eating out at restaurants
   • socialize - social activities, visiting friends/family
   • exercise - physical exercise, gym, sports
   • dropoff_pickup - dropping off or picking up people
   
   ⚠️ DO NOT USE: commute, lunch, breakfast, dinner, sleep, rest, leisure, etc.
   ⚠️ For commuting: use "home" before work and after work
   ⚠️ For meals at home: use "home"
   ⚠️ For meals outside: use "dine_out"

4. Consider ALL 16 user characteristics when generating the trajectory:
   - Age, Hispanic status, relationship, gender, race
   - Education level, employment status, work schedule (full-time/part-time)
   - Travel abroad experience, distance to work, work location state
   - Driver status, work from home status, occupation, primary activity

5. Times do NOT need to be on the hour - use realistic times like 7:15, 8:45, 12:30, etc.


Output Format:
First, provide a brief reasoning (max 100 words) about why this trajectory fits the user profile.
Then on a new line, output ONLY the trajectory in this EXACT format:
0:00-7:23 home, 7:23-8:15 exercise, 8:15-17:30 work, 17:30-18:45 shopping, 18:45-24:00 home

Important Rules:
- Start time MUST be 0:00 (or 00:00)
- End time MUST be 24:00
- Time intervals must NOT overlap
- Time intervals must be continuous with NO GAPS
- Activity names must be EXACTLY as listed (lowercase, no extra words)
- Times can be realistic (not just whole hours): 7:15, 8:47, 12:30, etc.
"""
            
            # 调用OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=500,
                temperature=1
            )
            
            result = response.choices[0].message.content.strip()
            
            # 返回成功结果
            return True, result
            
        except Exception as e:
            error_msg = str(e)
            
            # 检查是否是配额不足错误
            if "insufficient_user_quota" in error_msg.lower() or "quota" in error_msg.lower():
                return False, f"API配额不足: {error_msg}\n建议: 1) 减少批处理数量 2) 等待配额重置 3) 使用其他API密钥"
            
            # 检查是否是网络错误
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                retries += 1
                if retries < max_retries:
                    print(f"  网络错误，正在重试 ({retries}/{max_retries})...")
                    time.sleep(5)
                    continue
            
            # 其他错误
            retries += 1
            if retries < max_retries:
                print(f"  生成出错，正在重试 ({retries}/{max_retries}): {error_msg}")
                time.sleep(3)
            else:
                return False, f"生成失败 (已重试{max_retries}次): {error_msg}"
    
    return False, "达到最大重试次数"

# 解析生成的轨迹
def parse_trajectory_result(result):
    """解析API返回的轨迹结果，分离推理和轨迹部分"""
    lines = result.split('\n')
    
    reasoning_lines = []
    trajectory_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 更精确的轨迹识别：包含时间模式 (数字:数字-数字:数字)
        # 例如: "0:00-7:23 home" 或 "00:00-08:00 home"
        import re
        time_pattern = r'\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}'
        
        if re.search(time_pattern, line):
            # 这是轨迹行
            trajectory_lines.append(line)
        else:
            # 这是推理行
            reasoning_lines.append(line)
    
    reasoning = ' '.join(reasoning_lines).strip()
    trajectory = ', '.join(trajectory_lines).strip() if trajectory_lines else ''
    
    # 如果没有找到明确的轨迹，尝试从整个结果中提取
    if not trajectory:
        # 尝试找到包含活动模式的行
        import re
        # 匹配格式: 数字:数字-数字:数字 活动名
        pattern = r'\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s+\w+'
        matches = re.findall(pattern, result)
        if matches:
            trajectory = ', '.join(matches)
    
    return reasoning, trajectory

# 将轨迹转换为schedule格式
def convert_trajectory_to_schedule(trajectory):
    """将轨迹字符串转换为schedule格式的列表"""
    import re
    
    if not trajectory:
        return []
    
    schedule = []
    
    # 分割轨迹段
    segments = trajectory.split(',')
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # 提取时间和活动
        match = re.match(r'(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s+(.+)', segment)
        if match:
            start_hour = match.group(1).zfill(2)
            start_min = match.group(2)
            end_hour = match.group(3).zfill(2)
            end_min = match.group(4)
            activity = match.group(5).strip()
            
            schedule.append({
                "activity": activity,
                "start_time": f"{start_hour}:{start_min}",
                "end_time": f"{end_hour}:{end_min}"
            })
    
    return schedule

# 保存生成的轨迹
def save_trajectories(trajectories, output_dir):
    """保存轨迹到JSON文件（完整版和Schedule格式版）"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 保存完整版本（包含所有信息）
    output_file = os.path.join(output_dir, f"california_generated_trajectories_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)
    print(f"\n完整版轨迹已保存至: {output_file}")
    
    # 创建schedule格式版本（新格式，类似multi_agent的输出）
    schedule_trajectories = []
    for item in trajectories:
        schedule = convert_trajectory_to_schedule(item["trajectory"])
        if schedule:  # 只添加成功转换的轨迹
            schedule_trajectories.append({
                "user_id": item["user_id"],
                "schedule": schedule
            })
    
    # 保存schedule格式版本
    schedule_file = os.path.join(output_dir, f"all_trajectories_{timestamp}.json")
    with open(schedule_file, 'w', encoding='utf-8') as f:
        json.dump(schedule_trajectories, f, ensure_ascii=False, indent=2)
    print(f"Schedule格式轨迹已保存至: {schedule_file}")
    
    return output_file, schedule_file

# 保存统计报告
def save_statistics_report(statistics, output_dir):
    """保存统计报告"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"generation_report_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("California Trajectory Generation Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Statistics:\n")
        f.write("-"*80 + "\n")
        for key, value in statistics.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"统计报告已保存至: {report_file}")
    return report_file

# 主函数
def main():
    print(f"\n运行模式: {GENERATION_MODE}")
    if GENERATION_MODE == "user_file":
        print(f"User ID文件: {USER_ID_FILE}")
    elif GENERATION_MODE == "num_users":
        print(f"生成用户数: {NUM_USERS}")
    print("="*80 + "\n")
    
    # 文件路径配置
    static_file_path = "E:\mayue\FinalTraj\Oklahoma\processed_data\oklahoma_person_static.json"
    output_dir = "E:/mayue/FinalTraj/Trajectory_Generation/output"
    
    # 读取静态信息
    print(f"正在读取静态信息文件: {static_file_path}")
    static_info = read_california_static_info(static_file_path)
    
    if not static_info:
        print("错误: 没有找到有效的静态信息，程序退出。")
        return
    
    # 创建user_id到数据的映射
    user_dict = {item['user_id']: item for item in static_info}
    
    # 根据模式选择用户
    total_users = len(static_info)
    print(f"\n可用用户总数: {total_users}")
    print(f"当前运行模式: {GENERATION_MODE}")
    
    selected_users = []
    
    if GENERATION_MODE == "user_file":
        # 模式1: 从user_id文件读取
        print(f"\n正在从文件读取user_id: {USER_ID_FILE}")
        target_user_ids = read_user_ids(USER_ID_FILE)
        
        if not target_user_ids:
            print("错误: 未能读取有效的user_id列表，程序退出。")
            return
        
        print(f"目标用户ID数量: {len(target_user_ids)}")
        print(f"目标用户ID: {target_user_ids}")
        
        # 筛选出这些用户的数据
        for user_id in target_user_ids:
            if user_id in user_dict:
                selected_users.append(user_dict[user_id])
            else:
                print(f"  警告: 用户ID '{user_id}' 在数据集中未找到，跳过")
        
        print(f"\n成功找到 {len(selected_users)} 个用户数据")
        
        if not selected_users:
            print("错误: 没有找到任何有效的用户数据")
            return
        
    elif GENERATION_MODE == "num_users":
        # 模式2: 按数量生成（原模式）
        print("\n建议生成数量:")
        print("  - 测试: 5-10 个用户")
        print("  - 小批量: 50-100 个用户")
        print("  - 中等批量: 500-1000 个用户")
        print("  - 警告: 大批量生成可能消耗大量API配额\n")
        
        num_users = NUM_USERS
        print(f"当前设置: 生成前 {num_users} 个用户的轨迹")
        print("(如需修改，请编辑脚本顶部的 NUM_USERS 变量)\n")
        
        # 选择用户
        selected_users = static_info[:num_users]
    
    else:
        print(f"错误: 未知的运行模式 '{GENERATION_MODE}'")
        print("请设置 GENERATION_MODE 为 'num_users' 或 'user_file'")
        return
    
    num_users = len(selected_users)
    
    # 创建OpenAI客户端
    print("正在初始化OpenAI客户端...")
    client = create_openai_client()
    print("客户端初始化成功\n")
    
    # 生成轨迹
    all_trajectories = []
    success_count = 0
    failure_count = 0
    
    print("="*80)
    print("开始生成轨迹")
    print("="*80 + "\n")
    
    for i, person_data in enumerate(selected_users):
        user_info = extract_user_info(person_data)
        user_id = user_info['user_id']
        
        print(f"\n[{i+1}/{num_users}] 正在处理用户: {user_id}")
        print(f"  - 年龄: {user_info['age_range']}, 性别: {user_info['gender']}, 种族: {user_info['race']}, 关系: {user_info['relationship']}")
        print(f"  - 教育: {user_info['education']}, 是否拉美裔: {user_info['hispanic']}")
        print(f"  - 就业: {user_info['employment_status']}, 工作时间: {user_info['work_schedule']}, 职业: {user_info['occupation']}")
        print(f"  - 主要活动: {user_info['primary_activity']}, 在家工作: {user_info['work_from_home']}")
        print(f"  - 驾驶: {user_info['driver_on_travel_day']}, 到工作距离: {user_info['distance_to_work_miles']}英里, 工作州: {user_info['work_state']}")
        
        # 生成轨迹
        success, result = generate_user_trajectory_with_retry(client, user_info)
        
        if success:
            print(f"  ✓ 轨迹生成成功")
            success_count += 1
            
            # 解析结果
            reasoning, trajectory = parse_trajectory_result(result)
            
            # 存储结果
            all_trajectories.append({
                "user_id": user_id,
                "user_info": user_info,
                "reasoning": reasoning,
                "trajectory": trajectory,
                "raw_result": result,
                "generation_time": datetime.now().isoformat()
            })
            
            print(f"  推理: {reasoning[:100]}..." if len(reasoning) > 100 else f"  推理: {reasoning}")
            print(f"  轨迹: {trajectory[:100]}..." if len(trajectory) > 100 else f"  轨迹: {trajectory}")
        else:
            print(f"  ✗ 轨迹生成失败: {result}")
            failure_count += 1
            
            # 如果是配额错误，提前终止
            if "quota" in result.lower():
                print("\n警告: API配额不足，停止生成。")
                break
        
        # 添加延迟，避免API请求过于频繁
        if i < num_users - 1:
            delay = 1
            print(f"  等待 {delay} 秒...")
            time.sleep(delay)
    
    # 保存结果
    print("\n" + "="*80)
    print("生成完成，正在保存结果")
    print("="*80 + "\n")
    
    if all_trajectories:
        output_file, schedule_file = save_trajectories(all_trajectories, output_dir)
        
        # 统计信息
        statistics = {
            "运行模式": GENERATION_MODE,
            "总用户数": total_users,
            "选择生成数": num_users,
            "成功生成": success_count,
            "失败数量": failure_count,
            "成功率": f"{success_count/num_users*100:.2f}%",
            "完整版输出文件": output_file,
            "Schedule格式输出文件": schedule_file
        }
        
        report_file = save_statistics_report(statistics, output_dir)
        
        # 打印统计信息
        print("\n" + "="*80)
        print("生成统计")
        print("="*80)
        for key, value in statistics.items():
            print(f"{key}: {value}")
        print("="*80 + "\n")
        
    else:
        print("\n错误: 没有成功生成任何轨迹。")

if __name__ == "__main__":
    main()
