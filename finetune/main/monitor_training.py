#!/usr/bin/env python3
"""
实时监控训练进度和显存使用
使用方法: python monitor_training.py
"""

import os
import time
import subprocess
from pathlib import Path
import json

OUTPUT_DIR = "/data/mayue/cjy/Other_method/FinalTraj/finetune/output/llama3_1_trajectory_lora"

def get_gpu_memory():
    """获取GPU显存使用情况"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip().split('\n')
    except:
        return []

def get_latest_checkpoint():
    """获取最新的检查点"""
    if not os.path.exists(OUTPUT_DIR):
        return None
    
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')]
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    return checkpoints[-1]

def parse_trainer_state():
    """解析trainer_state.json获取训练进度"""
    checkpoint = get_latest_checkpoint()
    if not checkpoint:
        return None
    
    state_file = os.path.join(OUTPUT_DIR, checkpoint, 'trainer_state.json')
    if not os.path.exists(state_file):
        return None
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    return state

def main():
    """主监控循环"""
    print("\n" + "=" * 80)
    print("Llama 3.1 训练监控")
    print("=" * 80)
    print("按 Ctrl+C 停止监控\n")
    
    try:
        while True:
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("=" * 80)
            print(f"训练监控 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            # GPU状态
            print("\n【GPU 显存使用】")
            gpu_info = get_gpu_memory()
            if gpu_info:
                for i, line in enumerate(gpu_info):
                    parts = line.split(',')
                    if len(parts) >= 4:
                        gpu_id, mem_used, mem_total, util = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
                        mem_percent = (float(mem_used) / float(mem_total)) * 100
                        print(f"  GPU {gpu_id}: {mem_used}MB / {mem_total}MB ({mem_percent:.1f}%)  |  利用率: {util}%")
            else:
                print("  无法获取GPU信息")
            
            # 训练进度
            print("\n【训练进度】")
            state = parse_trainer_state()
            if state:
                current_step = state.get('global_step', 0)
                max_steps = state.get('max_steps', 0)
                epoch = state.get('epoch', 0)
                
                # 最新的loss
                log_history = state.get('log_history', [])
                latest_loss = None
                latest_lr = None
                
                for log in reversed(log_history):
                    if 'loss' in log and latest_loss is None:
                        latest_loss = log['loss']
                    if 'learning_rate' in log and latest_lr is None:
                        latest_lr = log['learning_rate']
                    if latest_loss and latest_lr:
                        break
                
                progress = (current_step / max_steps * 100) if max_steps > 0 else 0
                
                print(f"  Epoch: {epoch:.2f}")
                print(f"  Steps: {current_step} / {max_steps} ({progress:.1f}%)")
                if latest_loss:
                    print(f"  Loss: {latest_loss:.4f}")
                if latest_lr:
                    print(f"  Learning Rate: {latest_lr:.2e}")
                
                # 进度条
                bar_length = 50
                filled = int(bar_length * progress / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\n  [{bar}] {progress:.1f}%")
                
            else:
                print("  训练尚未开始或无法读取状态")
                checkpoint = get_latest_checkpoint()
                if checkpoint:
                    print(f"  最新检查点: {checkpoint}")
                else:
                    print("  暂无检查点")
            
            # 检查点列表
            print("\n【保存的检查点】")
            if os.path.exists(OUTPUT_DIR):
                checkpoints = sorted([d for d in os.listdir(OUTPUT_DIR) 
                                     if d.startswith('checkpoint-')],
                                    key=lambda x: int(x.split('-')[1]))
                if checkpoints:
                    for cp in checkpoints[-5:]:  # 显示最近5个
                        print(f"  ✓ {cp}")
                else:
                    print("  暂无检查点")
            
            print("\n" + "=" * 80)
            print("刷新间隔: 10秒 | TensorBoard: http://localhost:6006")
            print("=" * 80)
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    main()
