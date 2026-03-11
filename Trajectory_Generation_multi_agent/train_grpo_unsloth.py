"""
Stage 3: Rule-Guided GRPO Training for Editor Agent (Unsloth Version)

按照 https://github.com/datawhalechina/self-llm Qwen3 GRPO教程实现
使用 Unsloth + vLLM 框架进行高效训练
"""

import os
import json
import re
import torch
import numpy as np
from datasets import Dataset
from transformers import TextStreamer
from vllm import SamplingParams

# ==================== 配置参数 ====================

# 模型路径
BASE_MODEL_PATH = "/data/alice/cjtest/FinalTraj_KDD/finetune/models/Qwen3-8B/Qwen/Qwen3-8B"
SFT_MODEL_PATH = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage2_sft_output_epoch10/final_model"

# 数据路径
TRAIN_DATA_FILE = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage1_training_data_3/metadata.jsonl"

# 输出路径
OUTPUT_DIR = "/data/alice/cjtest/FinalTraj_KDD/Trajectory_Generation_multi_agent/stage3_grpo_output"

# Unsloth 参数
max_seq_length = 4096  # 最大序列长度
lora_rank = 32         # LoRA秩，教程推荐32

# GRPO 训练参数
LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 3
MAX_STEPS = 200  # 先用较小值测试
TEMPERATURE = 1.0
NUM_GENERATIONS = 4  # 每个prompt生成4个候选


# ==================== CoT 思考模板 ====================

# 定义标记（对应教程中的reasoning和solution标记）
reasoning_start = "[THOUGHT]"
reasoning_end = "[/THOUGHT]"
solution_start = "[JSON]"
solution_end = "[/JSON]"

# 系统提示词
system_prompt = f"""You are a Critic & Editor Agent for daily schedule generation.

Your task:
1. Analyze the user profile and generate a constraint-compliant daily schedule.
2. Output format: {reasoning_start}reasoning process{reasoning_end}{solution_start}schedule JSON{solution_end}

Constraints to satisfy:
- Physical: No time overlaps, total duration = 24 hours
- Logical: Proper start/end times
- Commonsense: Activities match user attributes
- Socioeconomic: Reasonable for user's occupation
- Temporal: Activities at appropriate times
- Internal: Consistent with user profile"""

# 构建 chat_template（参考教程）
chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

chat_template = chat_template\
    .replace("'{system_prompt}'", f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")


# ==================== 奖励模型 ====================

class ScheduleRewardModel:
    """程序化奖励模型"""
    
    def __init__(self):
        pass
    
    def compute_reward(self, user_profile: dict, generated_schedule: list) -> tuple:
        """
        计算奖励分数
        Returns: (total_reward, breakdown)
        """
        breakdown = {}
        
        # 硬约束检查
        r_physical, phys_details = self._check_physical(generated_schedule)
        r_logical, logic_details = self._check_logical(generated_schedule)
        
        breakdown['physical'] = {'score': r_physical, 'details': phys_details}
        breakdown['logical'] = {'score': r_logical, 'details': logic_details}
        
        # 违反硬约束直接惩罚
        if r_physical < 0 or r_logical < 0:
            total = r_physical + r_logical
            breakdown['total'] = total
            breakdown['violated_hard'] = True
            return total, breakdown
        
        # 软约束检查
        r_commonsense = self._check_commonsense(user_profile, generated_schedule)
        r_socioeconomic = self._check_socioeconomic(user_profile, generated_schedule)
        r_temporal = self._check_temporal(user_profile, generated_schedule)
        r_internal = self._check_internal(user_profile, generated_schedule)
        
        breakdown['commonsense'] = r_commonsense
        breakdown['socioeconomic'] = r_socioeconomic
        breakdown['temporal'] = r_temporal
        breakdown['internal'] = r_internal
        
        # 总分
        r_hard = r_physical + r_logical
        r_soft = r_commonsense + r_socioeconomic + r_temporal + r_internal
        total = r_hard * 1.0 + r_soft * 0.3
        
        breakdown['total'] = total
        breakdown['violated_hard'] = False
        return total, breakdown
    
    def _check_physical(self, schedule):
        """检查物理约束"""
        if not schedule:
            return -100, ['Empty schedule']
        
        issues = []
        total_duration = 0
        prev_end = None
        
        try:
            for i, act in enumerate(schedule):
                start = self._time_to_minutes(act.get('start_time', '0:00'))
                end = self._time_to_minutes(act.get('end_time', '0:00'))
                duration = act.get('duration', 0)
                
                # 检查时间顺序
                if end <= start and end != 0:
                    if not (start > 1200):  # 不是跨天情况
                        issues.append(f'Act {i}: invalid time range')
                
                # 检查重叠
                if prev_end is not None and start < prev_end:
                    issues.append(f'Act {i}: time overlap')
                
                prev_end = end if end > start else 1440
                total_duration += duration if duration > 0 else (end - start) / 60.0
            
            # 检查总时长
            if abs(total_duration - 24.0) > 0.5:
                issues.append(f'Total {total_duration:.1f}h != 24h')
            
            return (1, ['OK']) if not issues else (-100, issues)
        except Exception as e:
            return -100, [f'Parse error: {str(e)}']
    
    def _check_logical(self, schedule):
        """检查逻辑约束"""
        if not schedule:
            return -100, ['Empty schedule']
        
        issues = []
        first_act = schedule[0].get('activity', '').lower()
        last_act = schedule[-1].get('activity', '').lower()
        
        # 简化检查：只要求有合理的开始和结束
        if schedule[0].get('start_time') not in ['00:00', '0:00']:
            if 'home' not in first_act and 'sleep' not in first_act:
                issues.append('First activity should start at 00:00 or be home')
        
        if schedule[-1].get('end_time') not in ['24:00', '23:59', '00:00']:
            if 'home' not in last_act and 'sleep' not in last_act:
                issues.append('Last activity should end at 24:00 or be home')
        
        return (1, ['OK']) if not issues else (-50, issues)
    
    def _check_commonsense(self, profile, schedule):
        """常识检查"""
        score = 0
        occupation = profile.get('occupation', '').lower()
        activities = [a.get('activity', '').lower() for a in schedule]
        
        if 'student' in occupation:
            score += 2 if not any('work' in a for a in activities) else -5
        
        if 'employed' in profile.get('employment_status', '').lower():
            score += 5 if any('work' in a for a in activities) else -3
        
        return score
    
    def _check_socioeconomic(self, profile, schedule):
        """社会经济一致性"""
        score = 0
        occupation = profile.get('occupation', '').lower()
        
        # 简单检查：工人类少餐厅，专业人士可有休闲
        if any(w in occupation for w in ['worker', 'driver']):
            dining = sum(1 for a in schedule if 'dining' in a.get('activity', '').lower())
            score += 2 if dining <= 1 else -3
        
        return score
    
    def _check_temporal(self, profile, schedule):
        """时间节律"""
        score = 0
        for act in schedule:
            start = self._time_to_minutes(act.get('start_time', '0:00'))
            act_name = act.get('activity', '').lower()
            
            # 工作应在白天
            if 'work' in act_name:
                score += 5 if 480 <= start <= 1080 else -3
            
            # 夜间应在家
            if start >= 1320 or start <= 360:
                score += 2 if 'home' in act_name or 'sleep' in act_name else -2
        
        return score
    
    def _check_internal(self, profile, schedule):
        """内部一致性"""
        score = 0
        wfh = profile.get('work_from_home', '').lower()
        
        if wfh == 'yes':
            # 在家工作不应有通勤
            for i, act in enumerate(schedule):
                if 'work' in act.get('activity', '').lower():
                    if i > 0 and 'commut' in schedule[i-1].get('activity', '').lower():
                        score -= 10
                    if i < len(schedule)-1 and 'commut' in schedule[i+1].get('activity', '').lower():
                        score -= 10
            if score == 0:
                score += 5
        
        return score
    
    def _time_to_minutes(self, time_str):
        """时间转分钟"""
        try:
            if ':' in str(time_str):
                h, m = map(int, str(time_str).split(':'))
            else:
                h, m = int(time_str), 0
            return h * 60 + m
        except:
            return 0


# ==================== 格式匹配 ====================

# 提取JSON的正则
solution_end_regex = r"\[/JSON\][\s]{0,}"
match_format = re.compile(
    rf"{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end_regex}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL
)

def extract_schedule_from_output(output_text):
    """从输出提取schedule JSON"""
    try:
        json_match = re.search(r'\[JSON\](.*?)\[/JSON\]', output_text, re.DOTALL)
        if not json_match:
            return []
        json_str = json_match.group(1).strip()
        schedule = json.loads(json_str)
        return schedule if isinstance(schedule, list) else []
    except:
        return []


# ==================== 奖励函数（教程方式）====================

# 全局打印控制
PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5

def match_format_exactly(completions, **kwargs):
    """精确格式匹配奖励"""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        if match_format.search(response) is not None:
            score += 3.0
        scores.append(score)
    return scores

def match_format_approximately(completions, **kwargs):
    """近似格式匹配"""
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        score += 0.5 if response.count(reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(solution_start) == 1 else -1.0
        score += 0.5 if response.count(solution_end) == 1 else -1.0
        scores.append(score)
    return scores

def check_schedule_quality(prompts, completions, user_profile, **kwargs):
    """检查日程质量（主要奖励函数）"""
    global PRINTED_TIMES, PRINT_EVERY_STEPS
    
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    
    reward_model = ScheduleRewardModel()
    scores = []
    
    # 打印调试信息
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print('*'*50)
        print(f"Prompt (first 200 chars): {question[:200]}...")
        print(f"Response (first 300 chars): {responses[0][:300]}...")
    PRINTED_TIMES += 1
    
    for response, profile in zip(responses, user_profile):
        schedule = extract_schedule_from_output(response)
        
        if not schedule:
            scores.append(-5.0)
            continue
        
        # 计算奖励
        reward, breakdown = reward_model.compute_reward(profile, schedule)
        scores.append(float(reward))
    
    return scores


# ==================== 数据准备 ====================

def load_and_format_dataset():
    """加载并格式化数据集"""
    print("📂 Loading training data...")
    
    data_list = []
    with open(TRAIN_DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            user_profile = item.get('user_profile', {})
            
            # 构造prompt（messages格式）
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_profile, ensure_ascii=False)}
            ]
            
            data_list.append({
                "prompt": prompt,
                "user_profile": user_profile
            })
    
    dataset = Dataset.from_list(data_list)
    print(f"  ✓ Loaded {len(dataset)} samples")
    
    return dataset


# ==================== 主训练函数 ====================

def train():
    """主训练流程"""
    print("="*80)
    print(" STAGE 3: GRPO TRAINING (Unsloth Version)")
    print("="*80)
    print()
    
    # Step 1: 加载模型（使用Unsloth）
    print("🔧 Loading model with Unsloth...")
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_PATH,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,
    )
    print("  ✓ Model loaded")
    
    # Step 2: 添加LoRA
    print("\n🔧 Adding LoRA configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("  ✓ LoRA configured")
    
    # Step 3: 设置tokenizer的chat_template
    tokenizer.chat_template = chat_template
    print("  ✓ Chat template set")
    
    # Step 4: 加载数据集
    dataset = load_and_format_dataset()
    
    # Step 5: 先做短暂SFT（教程建议）
    print("\n🏋️ Running brief SFT warmup...")
    from trl import SFTTrainer, SFTConfig
    
    # 格式化数据用于SFT
    def format_for_sft(example):
        profile_str = json.dumps(example['user_profile'], ensure_ascii=False)
        # 简单生成一个示例输出（实际应该用ground truth）
        sample_output = f"{reasoning_start}Analyzing user profile...{reasoning_end}{solution_start}[]{solution_end}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": profile_str},
            {"role": "assistant", "content": sample_output}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    
    sft_dataset = dataset.map(format_for_sft)
    
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=5,
            num_train_epochs=1,  # 只做1个epoch的warmup
            learning_rate=2e-4,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            output_dir=os.path.join(OUTPUT_DIR, "sft_warmup"),
        ),
    )
    
    print("  Starting SFT warmup...")
    sft_trainer.train()
    print("  ✓ SFT warmup completed")
    
    # 清理内存
    del sft_dataset, sft_trainer
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # Step 6: GRPO训练
    print("\n🎯 Starting GRPO training...")
    
    # 计算最大长度
    tokenized = dataset.map(lambda x: {
        "tokens": tokenizer.apply_chat_template(
            x["prompt"],
            add_generation_prompt=True,
            tokenize=True
        )
    }, batched=True)
    
    tokenized = tokenized.map(lambda x: {"L": len(x["tokens"])})
    maximum_length = int(np.quantile(tokenized["L"], 0.9))
    print(f"  Max prompt length (90th percentile): {maximum_length}")
    
    # 过滤长样本
    dataset = dataset.select(
        np.where(np.array(tokenized["L"]) <= maximum_length)[0]
    )
    del tokenized
    
    max_prompt_length = maximum_length + 1
    max_completion_length = max_seq_length - max_prompt_length
    
    # vLLM采样参数
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    
    # GRPO配置
    from trl import GRPOConfig, GRPOTrainer
    
    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=TEMPERATURE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=MAX_STEPS,
        save_steps=50,
        report_to="none",
        output_dir=OUTPUT_DIR,
    )
    
    # 创建GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            check_schedule_quality,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    
    print("  ✓ GRPO Trainer initialized")
    print(f"  Training samples: {len(dataset)}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Num generations per prompt: {NUM_GENERATIONS}")
    print()
    
    # 开始训练
    trainer.train()
    
    print("\n✅ GRPO Training completed!")
    
    # 保存模型
    final_model_path = os.path.join(OUTPUT_DIR, "final_model")
    model.save_lora(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n💾 Model saved to: {final_model_path}")


if __name__ == "__main__":
    train()
