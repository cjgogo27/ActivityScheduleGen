"""
下载 Meta-Llama-3.1-8B-Instruct 到本地
目标路径: /data/alice/cjtest/FinalTraj_ar/finetune/models/Llama-3.1-8B-Instruct/
"""
from modelscope import snapshot_download

model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3.1-8B-Instruct',
    cache_dir='/data/alice/cjtest/FinalTraj_arr/finetune/models/Llama-3.1-8B-Instruct',
    revision='master',
    ignore_file_pattern=['original/consolidated.00.pth'],  # 跳过 15GB 的 PyTorch 原始格式，训练只需 safetensors
)
print(f"模型已下载到: {model_dir}")
