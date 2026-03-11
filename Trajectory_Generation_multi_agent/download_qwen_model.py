"""
Download Qwen3-8B model from ModelScope
"""

import os
from modelscope import snapshot_download

MODEL_NAME = "Qwen/Qwen3-8B"
CACHE_DIR = "/data/alice/cjtest/FinalTraj_KDD/finetune/models/Qwen3-8B"

print(f"Downloading {MODEL_NAME}...")
print(f"Cache directory: {CACHE_DIR}")

model_dir = snapshot_download(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    revision='master'
)

print(f"\n✅ Model downloaded successfully!")
print(f"Model path: {model_dir}")
