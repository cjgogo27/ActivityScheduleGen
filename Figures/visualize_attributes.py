"""
可视化个人和家庭属性分布
从generate_trajectories_with_household.py使用的数据文件中提取属性并绘制分布图
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import Counter
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 配置
PERSON_FILE = r"E:\mayue\FinalTraj\Oklahoma\processed_data_past\Oklahoma_person_static.json"
HOUSEHOLD_FILE = r"E:\mayue\FinalTraj\Oklahoma\processed_data_past\Oklahoma_household_static.json"
OUTPUT_DIR = r"E:\mayue\FinalTraj\Figures"

# 个人属性列表（16个维度）
PERSON_ATTRIBUTES = [
    'age_range', 'hispanic', 'relationship', 'gender', 'race', 
    'education', 'employment_status', 'traveled_abroad',
    'distance_to_work_miles', 'work_state', 'driver_on_travel_day',
    'work_from_home', 'work_schedule', 'occupation', 'primary_activity'
]

# 家庭属性列表（13个维度）
HOUSEHOLD_ATTRIBUTES = [
    'home_ownership', 'household_size', 'vehicle_count', 
    'household_income', 'driver_count', 'adult_count', 
    'young_children_count', 'msa_size', 'urban_area', 
    'household_race', 'household_hispanic', 'state'
]

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ 成功加载: {file_path} ({len(data)} 条记录)")
        return data
    except Exception as e:
        print(f"✗ 加载失败 {file_path}: {e}")
        return []

def plot_categorical_attribute(data, attribute_name, title, ax, top_n=15):
    """绘制分类属性的柱状图"""
    # 提取属性值
    values = [item.get(attribute_name, 'Unknown') for item in data]
    
    # 统计频次
    counter = Counter(values)
    
    # 取前N个最常见的值
    if len(counter) > top_n:
        most_common = counter.most_common(top_n)
        labels, counts = zip(*most_common)
    else:
        labels, counts = zip(*counter.most_common())
    
    # 绘制柱状图
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars = ax.bar(range(len(labels)), counts, color=colors, edgecolor='black', linewidth=0.5)
    
    # 设置标签
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('数量', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱子上显示数值
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=7)
    
    return counter

def plot_numerical_attribute(data, attribute_name, title, ax):
    """绘制数值属性的直方图"""
    # 提取数值
    values = []
    for item in data:
        val = item.get(attribute_name, 0)
        if isinstance(val, (int, float)) and val >= 0:
            values.append(val)
    
    if not values:
        ax.text(0.5, 0.5, '无有效数据', ha='center', va='center', fontsize=12)
        ax.set_title(title, fontsize=11, fontweight='bold')
        return None
    
    # 绘制直方图
    n, bins, patches = ax.hist(values, bins=20, color='skyblue', 
                                edgecolor='black', linewidth=0.5, alpha=0.7)
    
    # 设置标签
    ax.set_xlabel('值', fontsize=10)
    ax.set_ylabel('频次', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加统计信息
    mean_val = np.mean(values)
    median_val = np.median(values)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'均值: {mean_val:.1f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, label=f'中位数: {median_val:.1f}')
    ax.legend(fontsize=8)
    
    return {'mean': mean_val, 'median': median_val, 'min': min(values), 'max': max(values)}

def visualize_person_attributes(person_data):
    """可视化所有个人属性（16个维度）"""
    print("\n" + "="*70)
    print("开始绘制个人属性分布图...")
    print("="*70)
    
    # 创建大图：4行4列
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('个人属性分布 (Person Attributes)', fontsize=16, fontweight='bold', y=0.995)
    
    # 分类属性
    categorical_attrs = [
        ('age_range', '年龄段'),
        ('gender', '性别'),
        ('relationship', '家庭关系'),
        ('race', '种族'),
        ('hispanic', '西班牙裔'),
        ('education', '教育程度'),
        ('employment_status', '就业状态'),
        ('work_schedule', '工作时间表'),
        ('occupation', '职业'),
        ('primary_activity', '主要活动'),
        ('work_from_home', '在家工作'),
        ('driver_on_travel_day', '出行日驾驶'),
        ('traveled_abroad', '出国旅行'),
        ('work_state', '工作州')
    ]
    
    # 数值属性
    numerical_attrs = [
        ('distance_to_work_miles', '通勤距离 (英里)')
    ]
    
    # 绘制分类属性
    for idx, (attr, title) in enumerate(categorical_attrs, 1):
        ax = fig.add_subplot(4, 4, idx)
        counter = plot_categorical_attribute(person_data, attr, title, ax, top_n=10)
        print(f"  {idx}. {title}: {len(counter)} 个不同值")
    
    # 绘制数值属性
    for idx, (attr, title) in enumerate(numerical_attrs, len(categorical_attrs) + 1):
        ax = fig.add_subplot(4, 4, idx)
        stats = plot_numerical_attribute(person_data, attr, title, ax)
        if stats:
            print(f"  {idx}. {title}: 均值={stats['mean']:.1f}, 中位数={stats['median']:.1f}")
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'person_attributes_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 个人属性图已保存: {output_path}")
    plt.close()

def visualize_household_attributes(household_data):
    """可视化所有家庭属性（13个维度）"""
    print("\n" + "="*70)
    print("开始绘制家庭属性分布图...")
    print("="*70)
    
    # 创建大图：4行4列
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('家庭属性分布 (Household Attributes)', fontsize=16, fontweight='bold', y=0.995)
    
    # 分类属性
    categorical_attrs = [
        ('home_ownership', '房屋所有权'),
        ('household_income', '家庭收入'),
        ('msa_size', 'MSA大小'),
        ('urban_area', '城市区域'),
        ('household_race', '家庭种族'),
        ('household_hispanic', '家庭西班牙裔'),
        ('state', '州')
    ]
    
    # 数值属性
    numerical_attrs = [
        ('household_size', '家庭规模'),
        ('vehicle_count', '车辆数'),
        ('driver_count', '驾驶员数'),
        ('adult_count', '成人数'),
        ('young_children_count', '幼儿数')
    ]
    
    # 绘制分类属性
    for idx, (attr, title) in enumerate(categorical_attrs, 1):
        ax = fig.add_subplot(4, 4, idx)
        counter = plot_categorical_attribute(household_data, attr, title, ax, top_n=10)
        print(f"  {idx}. {title}: {len(counter)} 个不同值")
    
    # 绘制数值属性
    for idx, (attr, title) in enumerate(numerical_attrs, len(categorical_attrs) + 1):
        ax = fig.add_subplot(4, 4, idx)
        stats = plot_numerical_attribute(household_data, attr, title, ax)
        if stats:
            print(f"  {idx}. {title}: 均值={stats['mean']:.1f}, 中位数={stats['median']:.1f}")
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'household_attributes_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 家庭属性图已保存: {output_path}")
    plt.close()

def create_summary_statistics(person_data, household_data):
    """创建统计摘要并保存为文本文件"""
    output_path = os.path.join(OUTPUT_DIR, 'attributes_statistics_summary.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("属性统计摘要 (Attributes Statistics Summary)\n")
        f.write("="*70 + "\n\n")
        
        # 个人属性统计
        f.write("【个人属性】\n")
        f.write(f"总人数: {len(person_data)}\n\n")
        
        for attr in ['age_range', 'gender', 'employment_status', 'education']:
            values = [item.get(attr, 'Unknown') for item in person_data]
            counter = Counter(values)
            f.write(f"\n{attr}:\n")
            for val, count in counter.most_common(10):
                percentage = (count / len(values)) * 100
                f.write(f"  {val}: {count} ({percentage:.1f}%)\n")
        
        f.write("\n" + "="*70 + "\n\n")
        
        # 家庭属性统计
        f.write("【家庭属性】\n")
        f.write(f"总家庭数: {len(household_data)}\n\n")
        
        for attr in ['household_size', 'vehicle_count', 'household_income', 'home_ownership']:
            values = [item.get(attr, 'Unknown') for item in household_data]
            if attr in ['household_size', 'vehicle_count']:
                numerical_vals = [v for v in values if isinstance(v, (int, float))]
                if numerical_vals:
                    f.write(f"\n{attr}:\n")
                    f.write(f"  均值: {np.mean(numerical_vals):.2f}\n")
                    f.write(f"  中位数: {np.median(numerical_vals):.2f}\n")
                    f.write(f"  最小值: {min(numerical_vals)}\n")
                    f.write(f"  最大值: {max(numerical_vals)}\n")
            else:
                counter = Counter(values)
                f.write(f"\n{attr}:\n")
                for val, count in counter.most_common(10):
                    percentage = (count / len(values)) * 100
                    f.write(f"  {val}: {count} ({percentage:.1f}%)\n")
    
    print(f"\n✓ 统计摘要已保存: {output_path}")

def main():
    """主函数"""
    print("="*70)
    print("可视化个人和家庭属性分布")
    print("="*70)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}\n")
    
    # 加载数据
    person_data = load_json(PERSON_FILE)
    household_data = load_json(HOUSEHOLD_FILE)
    
    if not person_data:
        print("✗ 无个人数据，退出程序")
        return
    
    if not household_data:
        print("✗ 无家庭数据，退出程序")
        return
    
    # 可视化个人属性
    visualize_person_attributes(person_data)
    
    # 可视化家庭属性
    visualize_household_attributes(household_data)
    
    # 创建统计摘要
    create_summary_statistics(person_data, household_data)
    
    print("\n" + "="*70)
    print("✓ 所有可视化完成！")
    print("="*70)
    print(f"\n生成的文件:")
    print(f"  1. {os.path.join(OUTPUT_DIR, 'person_attributes_distribution.png')}")
    print(f"  2. {os.path.join(OUTPUT_DIR, 'household_attributes_distribution.png')}")
    print(f"  3. {os.path.join(OUTPUT_DIR, 'attributes_statistics_summary.txt')}")

if __name__ == "__main__":
    main()
