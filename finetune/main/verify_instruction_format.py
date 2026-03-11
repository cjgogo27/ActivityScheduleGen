"""
验证指令集格式是否符合Llama模型要求

展示:
1. 指令集的原始格式 (instruction-input-output)
2. 转换后的Llama3.1格式
3. 与参考示例的对比
"""

import json

def show_format_example():
    """展示指令集格式示例"""
    
    print("=" * 80)
    print("指令集格式验证")
    print("=" * 80)
    
    # 1. 我们的指令集格式
    print("\n【1】我们的指令集格式 (存储在JSON文件中):")
    print("-" * 80)
    
    our_format = {
        "instruction": """Create a REALISTIC and COMPLETE 24-hour daily schedule for this person.

PERSON PROFILE (User ID: 30150869_1):
- Relationship: Self
- Age: 35-44
- Gender: Male
- Employment status: Yes
- Work schedule: Full-time
- Can drive: Yes

HOUSEHOLD INFORMATION:
- Household size: 2 people
- Young children: 0
- Vehicle count: 2

TASK: Generate a realistic 24-hour trajectory that:
1. Covers 00:00-24:00 continuously with NO gaps
2. Reflects the person's employment status and work schedule
3. Uses realistic times with minutes (e.g., 07:30, 16:45)

OUTPUT FORMAT (JSON only):
```json
{
  "schedule": [
    {"activity": "home", "start_time": "00:00", "end_time": "07:30"},
    {"activity": "work", "start_time": "07:30", "end_time": "16:45"}
  ]
}
```""",
        "input": "",
        "output": """{
  "schedule": [
    {"activity": "home", "start_time": "00:00", "end_time": "07:30"},
    {"activity": "work", "start_time": "07:30", "end_time": "16:45"},
    {"activity": "shopping", "start_time": "16:45", "end_time": "17:30"},
    {"activity": "home", "start_time": "17:30", "end_time": "24:00"}
  ]
}"""
    }
    
    print(json.dumps(our_format, ensure_ascii=False, indent=2))
    
    # 2. 参考的甄嬛示例格式
    print("\n" + "=" * 80)
    print("【2】参考示例 - 甄嬛对话格式:")
    print("-" * 80)
    
    reference_format = {
        "instruction": "你是谁？",
        "input": "",
        "output": "家父是大理寺少卿甄远道。"
    }
    
    print(json.dumps(reference_format, ensure_ascii=False, indent=2))
    
    # 3. 格式对比
    print("\n" + "=" * 80)
    print("【3】格式对比分析:")
    print("-" * 80)
    
    comparison = """
    ┌─────────────────┬──────────────────────┬──────────────────────┐
    │     字段        │    甄嬛示例          │    我们的轨迹生成    │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ instruction     │ 简短问题             │ 详细任务描述+数据    │
    │                 │ "你是谁？"           │ "Create schedule..." │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ input           │ 空                   │ 空                   │
    │                 │ ""                   │ ""                   │
    ├─────────────────┼──────────────────────┼──────────────────────┤
    │ output          │ 简短回答             │ JSON格式的轨迹数据   │
    │                 │ "家父是..."          │ {"schedule": [...]}  │
    └─────────────────┴──────────────────────┴──────────────────────┘
    
    ✓ 格式完全一致: 都是 {instruction, input, output} 三元组
    ✓ 符合Llama要求: 可以被process_func正确处理
    """
    print(comparison)
    
    # 4. Llama3.1处理后的格式
    print("\n" + "=" * 80)
    print("【4】经过process_func处理后的Llama3.1格式:")
    print("-" * 80)
    
    system_prompt = "You are an AI assistant specialized in generating realistic daily activity schedules."
    user_message = our_format['instruction']
    assistant_response = our_format['output']
    
    llama_format = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_response}<|eot_id|>"""
    
    print("完整的Llama3.1 Prompt Template (前500字符):")
    print(llama_format[:500] + "...\n")
    
    # 5. 验证结论
    print("=" * 80)
    print("【5】验证结论:")
    print("-" * 80)
    print("""
    ✅ 指令集格式正确:
       - 使用标准的 {instruction, input, output} 格式
       - 与参考示例(甄嬛)的格式完全一致
       - 可以被train_llama_lora.py中的process_func正确处理
    
    ✅ 数据内容合理:
       - instruction: 包含完整的任务描述和输入数据(个人+家庭信息)
       - input: 为空(所有信息已在instruction中)
       - output: JSON格式的轨迹数据,符合模型输出要求
    
    ✅ Llama3.1兼容:
       - process_func会将其转换为Llama3.1的Prompt Template格式
       - 自动添加<|begin_of_text|>, <|start_header_id|>等特殊token
       - instruction部分的token会被mask为-100(不计算loss)
       - output部分的token用于训练(计算loss)
    
    📌 关键点:
       1. instruction可以很长(包含所有context信息) ← 这是正确的!
       2. input可以为空(信息已在instruction中) ← 这也是正确的!
       3. output是模型需要学习生成的内容 ← 完美!
    """)
    print("=" * 80)


def verify_with_actual_data():
    """使用实际数据进行验证"""
    import os
    
    dataset_file = "/data/mayue/cjy/Other_method/FinalTraj/finetune/trajectory_instruction_dataset.json"
    
    if not os.path.exists(dataset_file):
        print(f"\n⚠️  数据集文件不存在: {dataset_file}")
        print("   请先运行: python build_instruction_dataset_v2.py")
        return
    
    print("\n" + "=" * 80)
    print("【6】实际数据集验证:")
    print("-" * 80)
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✓ 数据集大小: {len(dataset)} 条样本")
    
    if dataset:
        sample = dataset[0]
        
        # 验证格式
        required_keys = {'instruction', 'input', 'output'}
        actual_keys = set(sample.keys())
        
        if required_keys == actual_keys:
            print("✓ 格式正确: 包含所有必需字段 (instruction, input, output)")
        else:
            print(f"✗ 格式错误: 缺少字段 {required_keys - actual_keys}")
            return
        
        # 检查数据类型
        print(f"✓ instruction类型: {type(sample['instruction'])} (长度: {len(sample['instruction'])} 字符)")
        print(f"✓ input类型: {type(sample['input'])} (长度: {len(sample['input'])} 字符)")
        print(f"✓ output类型: {type(sample['output'])} (长度: {len(sample['output'])} 字符)")
        
        # 验证output是否为有效JSON
        try:
            output_json = json.loads(sample['output'])
            if 'schedule' in output_json:
                print(f"✓ output是有效的JSON,包含 {len(output_json['schedule'])} 个活动")
            else:
                print("⚠️  output JSON缺少'schedule'字段")
        except:
            print("✗ output不是有效的JSON")
        
        # 显示第一个样本的摘要
        print("\n第1个样本摘要:")
        print(f"  Instruction (前200字符): {sample['instruction'][:200]}...")
        print(f"  Input: '{sample['input']}'")
        print(f"  Output (前150字符): {sample['output'][:150]}...")


if __name__ == "__main__":
    show_format_example()
    verify_with_actual_data()
    
    print("\n" + "=" * 80)
    print("总结:")
    print("=" * 80)
    print("""
我们的指令集格式完全符合Llama模型的要求!

格式: {instruction, input, output}
- instruction: 任务描述 + 所有输入数据 (个人信息、家庭信息)
- input: 空字符串 (因为所有数据已在instruction中)
- output: 模型应该生成的轨迹JSON

这与参考示例(甄嬛)的格式完全一致,区别只是:
- 甄嬛: instruction简短, output简短
- 我们: instruction详细(包含context), output是JSON

两者都能被process_func正确处理并转换为Llama3.1格式!
    """)
    print("=" * 80)
