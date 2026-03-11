#!/usr/bin/env python3
"""
从评估结果txt文件中恢复CSV记录
"""

import os
import re
import pandas as pd
from datetime import datetime

# 需要处理的文件列表
EVALUATION_FILES = [
    "evaluation_20251117_122456.txt",
    "evaluation_20251117_123459.txt",
    "evaluation_20251117_123525.txt",
    "evaluation_20251117_150350.txt",
    "evaluation_20251117_155339.txt",
    "evaluation_20251117_163034.txt",
    "evaluation_20251124_121652.txt",
    "evaluation_20251124_122003.txt",
    "evaluation_20251124_123420.txt",
    "evaluation_20251124_132710.txt",
]

EVAL_DIR = r"E:\FrankYcj\FinalTraj\evaluation\evaluation_results"
CSV_PATH = r"E:\FrankYcj\FinalTraj\evaluation\evaluation_results\experiment_results.csv"


def extract_info_from_file(file_path):
    """从评估文件中提取信息"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取时间戳
    timestamp_match = re.search(r'Time: (.+)', content)
    timestamp = timestamp_match.group(1) if timestamp_match else ""
    
    # 提取生成文件名
    gen_file_match = re.search(r'Generated File: (.+)', content)
    gen_file_full = gen_file_match.group(1) if gen_file_match else ""
    gen_filename = os.path.basename(gen_file_full)
    
    # 提取原始文件名
    orig_file_match = re.search(r'Original File: (.+)', content)
    orig_file_full = orig_file_match.group(1) if orig_file_match else ""
    orig_filename = os.path.basename(orig_file_full)
    
    # 提取用户数
    users_match = re.search(r'Users: (\d+)', content)
    num_users = int(users_match.group(1)) if users_match else 0
    
    # 判断轨迹类型
    if 'Trajectory_Generation_multi_agent' in gen_file_full or 'multi_agent' in gen_file_full.lower():
        trajectory_type = 'Multi_agent'
    elif 'Trajectory_Generation_Household' in gen_file_full or 'household' in gen_filename.lower():
        trajectory_type = 'Household'
    else:
        trajectory_type = 'Personal'
    
    # 提取所有指标
    metrics = {}
    metrics_section = re.search(r'Metrics:\n-+\n(.*?)\n\n', content, re.DOTALL)
    if metrics_section:
        metrics_text = metrics_section.group(1)
        # 提取每个指标
        for line in metrics_text.split('\n'):
            if ':' in line:
                parts = line.split(':')
                metric_name = parts[0].strip()
                metric_value = parts[1].strip()
                try:
                    metrics[metric_name] = float(metric_value)
                except ValueError:
                    metrics[metric_name] = None
    
    # 构建记录
    record = {
        'type': trajectory_type,
        'timestamp': timestamp,
        'generated_file': gen_filename,
        'original_file': orig_filename,
        'num_users': num_users,
        'accuracy': metrics.get('accuracy'),
        'f1_score': metrics.get('f1-score'),
        'edit_dist': metrics.get('edit_dist'),
        'bleu_score': metrics.get('bleu_score'),
        'data_jsd': metrics.get('data_jsd'),
        'macro_int': metrics.get('macro_int'),
        'micro_int': metrics.get('micro_int'),
        'act_type': metrics.get('act_type'),
        'uni_act_type': metrics.get('uni_act_type'),
        'traj_len': metrics.get('traj_len'),
        'macro_hour': metrics.get('macro_hour'),
        'micro_hour': metrics.get('micro_hour')
    }
    
    return record


def main():
    """主函数"""
    print("="*70)
    print("从评估文件恢复CSV记录")
    print("="*70)
    
    records = []
    
    for filename in EVALUATION_FILES:
        file_path = os.path.join(EVAL_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"⚠ 文件不存在: {filename}")
            continue
        
        try:
            record = extract_info_from_file(file_path)
            records.append(record)
            print(f"✓ 处理完成: {filename}")
            print(f"  类型: {record['type']}, 用户数: {record['num_users']}, 准确率: {record['accuracy']:.4f}")
        except Exception as e:
            print(f"✗ 处理失败: {filename}")
            print(f"  错误: {e}")
    
    if not records:
        print("\n没有成功提取任何记录!")
        return
    
    # 创建DataFrame
    df = pd.DataFrame(records)
    
    # 确保列的顺序
    columns_order = ['type', 'timestamp', 'generated_file', 'original_file', 'num_users',
                     'accuracy', 'f1_score', 'edit_dist', 'bleu_score', 'data_jsd',
                     'macro_int', 'micro_int', 'act_type', 'uni_act_type', 'traj_len',
                     'macro_hour', 'micro_hour']
    
    # 只保留存在的列
    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns]
    
    # 按时间排序
    df = df.sort_values('timestamp')
    
    # 保存到CSV
    df.to_csv(CSV_PATH, index=False, encoding='utf-8')
    
    print(f"\n{'='*70}")
    print(f"✓ CSV文件已保存: {CSV_PATH}")
    print(f"  总记录数: {len(df)}")
    print(f"\n记录概览:")
    print(f"  Multi_agent: {len(df[df['type'] == 'Multi_agent'])} 条")
    print(f"  Household: {len(df[df['type'] == 'Household'])} 条")
    print(f"  Personal: {len(df[df['type'] == 'Personal'])} 条")
    print(f"\n平均准确率:")
    print(f"  Multi_agent: {df[df['type'] == 'Multi_agent']['accuracy'].mean():.4f}")
    print(f"  Household: {df[df['type'] == 'Household']['accuracy'].mean():.4f}")
    print(f"  Personal: {df[df['type'] == 'Personal']['accuracy'].mean():.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
