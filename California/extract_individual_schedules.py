"""
提取每个人的轨迹并保存为独立的JSON文件
从 california_trajectories_processed.json 中提取每个用户的活动时间表
"""

import json
import os
from pathlib import Path


def load_trajectories(input_file):
    """加载轨迹数据"""
    print(f"正在加载轨迹数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查数据类型
    if isinstance(data, dict):
        print(f"成功加载 {len(data)} 个用户的轨迹 (字典格式)")
    elif isinstance(data, list):
        print(f"成功加载 {len(data)} 个用户的轨迹 (列表格式)")
    else:
        print(f"警告: 未知的数据格式类型: {type(data)}")
    
    return data


def extract_schedule(user_data):
    """从用户数据中提取时间表"""
    schedule = []
    
    # 检查是否有轨迹数据
    if 'trajectories' not in user_data:
        return None
    
    trajectories = user_data['trajectories']
    
    if not trajectories or len(trajectories) == 0:
        return None
    
    for traj in trajectories:
        # 提取时间,去掉秒数部分
        start_time = traj.get('arrival_time', '00:00:00')[:5]  # 取 HH:MM
        end_time = traj.get('departure_time', '00:00:00')[:5]  # 取 HH:MM
        
        schedule_item = {
            'activity': traj.get('activity', 'other'),
            'start_time': start_time,
            'end_time': end_time
        }
        schedule.append(schedule_item)
    
    return schedule


def save_all_schedules(all_schedules, output_file):
    """保存所有用户的时间表到一个JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_schedules, f, indent=2, ensure_ascii=False)


def main():
    # 设置路径
    base_dir = Path(r"E:\mayue\FinalTraj\California\processed_data")
    input_file = base_dir / "california_trajectories_processed.json"
    output_dir = base_dir
    output_file = output_dir / "all_user_schedules.json"
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录: {output_dir}")
    print(f"输出文件: {output_file}")
    
    # 加载轨迹数据
    trajectories_data = load_trajectories(input_file)
    
    # 统计信息
    total_users = len(trajectories_data)
    successful = 0
    failed = 0
    
    # 存储所有用户的时间表
    all_schedules = []
    
    print(f"\n开始提取个人时间表...")
    
    # 遍历每个用户
    # 数据可能是列表或字典格式
    if isinstance(trajectories_data, dict):
        user_items = trajectories_data.items()
    else:
        # 如果是列表,每个元素应该包含user_id
        user_items = [(item.get('user_id', f'user_{i}'), item) for i, item in enumerate(trajectories_data)]
    
    for user_id, user_data in user_items:
        try:
            # 提取时间表
            schedule = extract_schedule(user_data)
            
            if schedule is None or len(schedule) == 0:
                failed += 1
                continue
            
            # 添加到总列表中
            user_schedule = {
                'user_id': user_id,
                'schedule': schedule
            }
            all_schedules.append(user_schedule)
            successful += 1
            
            # 每处理1000个用户显示进度
            if successful % 1000 == 0:
                print(f"已处理: {successful} 个用户...")
                
        except Exception as e:
            print(f"处理用户 {user_id} 时出错: {str(e)}")
            failed += 1
    
    # 保存所有时间表到一个文件
    if successful > 0:
        print(f"\n正在保存所有用户的时间表到文件...")
        save_all_schedules(all_schedules, output_file)
    
    # 输出统计信息
    print(f"\n" + "="*60)
    print(f"处理完成!")
    print(f"总用户数: {total_users}")
    print(f"成功提取: {successful}")
    print(f"失败/跳过: {failed}")
    print(f"输出文件: {output_file}")
    print(f"="*60)
    
    # 显示前3个用户的示例
    if successful > 0:
        print(f"\n前3个用户的示例:")
        for i, user_schedule in enumerate(all_schedules[:3]):
            print(f"\n--- 用户 {i+1}: {user_schedule['user_id']} ---")
            print(json.dumps(user_schedule, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
