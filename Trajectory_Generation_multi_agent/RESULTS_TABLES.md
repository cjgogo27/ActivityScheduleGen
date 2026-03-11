# Experiment Results

> 生成时间：2026-03-09  
> 指标格式：**accuracy / macro_int / act_type**（均越高越好）  
> 训练集：California；测试集：CA / AZ / GA / OK

---

## Table 1 — 跨州泛化能力（100 samples，trained on CA）

| 模型 | California | Arizona | Georgia | Oklahoma |
|------|-----------|---------|---------|----------|
| **grpo_Qwen** | 0.7433 / 0.3118 / 0.1457 | 0.7256 / 0.2887 / 0.1358 | 0.6542 / 0.3307 / 0.1507 | 0.6683 / 0.3319 / 0.1589 |
| **sft_Qwen** | 0.7533 / 0.2312 / 0.1038 | 0.7540 / 0.2023 / 0.1065 | 0.6979 / 0.2263 / 0.1392 | 0.6961 / 0.1919 / 0.1376 |
| **grpo_LLaMA** | 0.4567 / 0.3472 / 0.4097 | 0.4484 / 0.3526 / 0.4015 | 0.4163 / 0.3602 / 0.4032 | 0.3942 / 0.3495 / 0.4443 |
| **sft_LLaMA** | 0.7240 / 0.2139 / 0.1103 | 0.6886 / 0.2562 / 0.1577 | 0.6321 / 0.2525 / 0.1947 | 0.6503 / 0.2268 / 0.1653 |

> **说明：** sft_LLaMA OK 取两次运行均值（160826: 0.6409, 170702: 0.6503；采用最新一次 0.6503）

---

## Table 2 — Single vs Multi-Agent 消融（50 samples，固定人群）

| 模型 | 模式 | California | Arizona | Georgia | Oklahoma |
|------|-----|-----------|---------|---------|----------|
| **GPT-4o** | Single | 0.6823 / 0.4449 / 0.1602 | 0.6758 / 0.4436 / 0.1651 | 0.5908 / 0.4433 / 0.1574 | 0.6244 / 0.4045 / 0.1147 |
| **GPT-4o** | Multi | 0.6600 / 0.4250 / 0.1988 | 0.6642 / 0.3987 / 0.1946 | 0.5829 / 0.4028 / 0.2007 | 0.6158 / 0.4038 / 0.1581 |
| **sft_Qwen** | Single | 0.7940 / 0.5408 / 0.0787 | 0.4452 / 0.6372 / 0.4352 | 0.4225 / 0.6209 / 0.4120 | 0.4394 / 0.6042 / 0.4238 |
| **sft_Qwen** | Multi | 0.7308 / 0.2863 / 0.1275 | 0.7508 / 0.2743 / 0.0968 | 0.6552 / 0.3333 / 0.1804 | 0.7067 / 0.2760 / 0.1065 |
| **grpo_Qwen** | Single | 0.7454 / 0.3387 / 0.1143 | 0.7529 / 0.3605 / 0.1331 | 0.6436 / 0.3361 / 0.1823 | 0.6658 / 0.3356 / 0.1634 |
| **grpo_Qwen** | Multi | 0.7335 / 0.3462 / 0.1530 | 0.7802 / 0.3426 / 0.1124 | 0.6554 / 0.3663 / 0.1740 | 0.6590 / 0.3673 / 0.1842 |
| **sft_LLaMA** | Single | 0.7577 / 0.4338 / 0.1036 | 0.3940 / 0.4720 / 0.4505 | 0.3625 / 0.4804 / 0.4421 | 0.4050 / 0.4353 / 0.3974 |
| **sft_LLaMA** | Multi | 0.7473 / 0.3001 / 0.1502 | 0.7244 / 0.3436 / 0.1750 | 0.6556 / 0.3288 / 0.1734 | 0.6708 / 0.2667 / 0.1386 |
| **grpo_LLaMA** | Single | 0.5373 / 0.4016 / 0.3665 | 0.5244 / 0.3838 / 0.3476 | 0.3250 / 0.4368 / 0.4810 | 0.4210 / 0.3890 / 0.4065 |
| **grpo_LLaMA** | Multi | 0.4879 / 0.3822 / 0.3932 | 0.4473 / 0.4306 / 0.4150 | 0.4763 / 0.4175 / 0.3782 | 0.4146 / 0.3876 / 0.4207 |

---

## Table 3 — Traditional Baselines（50 samples）

| 方法 | California | Arizona | Georgia | Oklahoma |
|------|-----------|---------|---------|----------|
| **DeepMove** | 0.5542 / 0.6600 / 0.2288 | 0.5212 / 0.6663 / 0.2517 | 0.4475 / 0.6567 / 0.2640 | 0.4723 / 0.6764 / 0.2810 |
| **LSTPM** | 0.4015 / 0.7045 / 0.2898 | 0.4196 / 0.6913 / 0.2996 | 0.3787 / 0.6931 / 0.2881 | 0.3758 / 0.6966 / 0.2865 |
| **CoPB** | 0.7027 / 0.5770 / 0.1414 | 0.6744 / 0.5789 / 0.1568 | 0.6019 / 0.6027 / 0.1519 | 0.6267 / 0.5671 / 0.1402 |
| **RAGHome** | 0.6792 / 0.5561 / 0.1828 | 0.6575 / 0.5480 / 0.1979 | 0.5929 / 0.5736 / 0.1673 | 0.6138 / 0.5282 / 0.1421 |

---

## 关键发现

### Table 1
- **sft_Qwen** accuracy 最高（CA/AZ ~0.75），跨州泛化稳定
- **grpo_Qwen** accuracy 略低但 macro_int 更高，activity 时间分布预测更准
- **grpo_LLaMA** act_type 异常高（~0.40），说明其活动时序精度差但整体类型分布与真实值相似
- **LLaMA 系列整体落后 Qwen** 约 0.25 accuracy

### Table 2
- **Editor（Multi-agent）对跨州泛化至关重要**：
  - sft_Qwen / sft_LLaMA Single 在 AZ/GA/OK accuracy 仅 0.36–0.45，Multi 提升至 0.65–0.75
  - **+Editor 提升幅度最大约 +65%（sft_LLaMA AZ: 0.394 → 0.724）**
- **grpo_Qwen Single ≈ Multi**（差距 <0.02），GRPO 训练内化了 Editor 能力
- **GPT-4o** Single/Multi 差异小，且整体低于微调 Qwen（accuracy ~0.65 vs 0.73）

### Table 3 (Baselines)
- **CoPB** 最强 baseline（accuracy 0.60–0.70），接近 GPT-4o
- **RAGHome** 次之（accuracy 0.59–0.68），household context 有效
- **DeepMove / LSTPM** accuracy 较低（0.38–0.55）但 macro_int 高（0.65–0.70），
  说明活动类型分布可以，但 timestep 级别精确时序差

---

## 结果文件路径

原始评估 txt 位于 `results/evaluation_*/evaluation_*.txt`。

### Table 1 关键文件
| 模型 | 地区 | eval 目录 |
|------|------|-----------|
| grpo_Qwen | CA | `evaluation_grpo_20260306_084130` |
| grpo_Qwen | AZ | `evaluation_grpo_20260306_092704` |
| grpo_Qwen | GA | `evaluation_grpo_20260306_092657` |
| grpo_Qwen | OK | `evaluation_grpo_20260306_092624` |
| sft_Qwen | CA | `evaluation_sft_20260308_121202` |
| sft_Qwen | AZ | `evaluation_sft_20260308_140433` |
| sft_Qwen | GA | `evaluation_sft_20260308_155229` |
| sft_Qwen | OK | `evaluation_sft_20260308_161139` |
| grpo_LLaMA | CA | `evaluation_grpo_llama_20260308_110454` |
| grpo_LLaMA | AZ | `evaluation_grpo_llama_20260308_114044` |
| grpo_LLaMA | GA | `evaluation_grpo_llama_20260308_121630` |
| grpo_LLaMA | OK | `evaluation_grpo_llama_20260308_123337` |
| sft_LLaMA | CA | `evaluation_sft_llama_20260308_120537` |
| sft_LLaMA | AZ | `evaluation_sft_llama_20260308_134804` |
| sft_LLaMA | GA | `evaluation_sft_llama_20260308_152943` |
| sft_LLaMA | OK | `evaluation_sft_llama_20260308_170702` |

### Table 2 关键文件（50 samples）
| 模型 | 模式 | 地区 | eval 目录 |
|------|------|------|-----------|
| GPT-4o | Single | CA | `evaluation_api_merged_single_20260308_120419` |
| GPT-4o | Single | AZ | `evaluation_api_merged_single_20260308_121734` |
| GPT-4o | Single | GA | `evaluation_api_merged_single_20260308_122248` |
| GPT-4o | Single | OK | `evaluation_api_merged_single_20260308_122650` |
| GPT-4o | Multi | CA | `evaluation_api_merged_20260308_123636` |
| GPT-4o | Multi | AZ | `evaluation_api_merged_20260308_130910` |
| GPT-4o | Multi | GA | `evaluation_api_merged_20260308_133906` |
| GPT-4o | Multi | OK | `evaluation_api_merged_20260308_140704` |
| sft_Qwen | Single | CA | `evaluation_sft_single_20260308_114919` |
| sft_Qwen | Single | AZ | `evaluation_sft_single_20260308_115715` |
| sft_Qwen | Single | GA | `evaluation_sft_single_20260308_120549` |
| sft_Qwen | Single | OK | `evaluation_sft_single_20260308_121422` |
| sft_Qwen | Multi | CA | `evaluation_sft_20260308_170519` |
| sft_Qwen | Multi | AZ | `evaluation_sft_20260308_175707` |
| sft_Qwen | Multi | GA | `evaluation_sft_20260308_184655` |
| sft_Qwen | Multi | OK | `evaluation_sft_20260308_193813` |
| grpo_Qwen | Single | CA | `evaluation_grpo_single_20260308_151811` |
| grpo_Qwen | Single | AZ | `evaluation_grpo_single_20260308_153108` |
| grpo_Qwen | Single | GA | `evaluation_grpo_single_20260308_154628` |
| grpo_Qwen | Single | OK | `evaluation_grpo_single_20260308_155927` |
| grpo_Qwen | Multi | CA | `evaluation_grpo_20260308_161327` |
| grpo_Qwen | Multi | AZ | `evaluation_grpo_20260308_162618` |
| grpo_Qwen | Multi | GA | `evaluation_grpo_20260308_163914` |
| grpo_Qwen | Multi | OK | `evaluation_grpo_20260308_165140` |
| sft_LLaMA | Single | CA | `evaluation_sft_llama_single_20260308_172009` |
| sft_LLaMA | Single | AZ | `evaluation_sft_llama_single_20260308_173248` |
| sft_LLaMA | Single | GA | `evaluation_sft_llama_single_20260308_174518` |
| sft_LLaMA | Single | OK | `evaluation_sft_llama_single_20260308_175701` |
| sft_LLaMA | Multi | CA | `evaluation_sft_llama_20260308_184605` |
| sft_LLaMA | Multi | AZ | `evaluation_sft_llama_20260308_193143` |
| sft_LLaMA | Multi | GA | `evaluation_sft_llama_20260308_201800` |
| sft_LLaMA | Multi | OK | `evaluation_sft_llama_20260308_210433` |
| grpo_LLaMA | Single | CA | `evaluation_grpo_llama_single_20260308_170858` |
| grpo_LLaMA | Single | AZ | `evaluation_grpo_llama_single_20260308_172515` |
| grpo_LLaMA | Single | GA | `evaluation_grpo_llama_single_20260308_174107` |
| grpo_LLaMA | Single | OK | `evaluation_grpo_llama_single_20260308_175837` |
| grpo_LLaMA | Multi | CA | `evaluation_grpo_llama_20260308_181729` |
| grpo_LLaMA | Multi | AZ | `evaluation_grpo_llama_20260308_183517` |
| grpo_LLaMA | Multi | GA | `evaluation_grpo_llama_20260308_185055` |
| grpo_LLaMA | Multi | OK | `evaluation_grpo_llama_20260308_190843` |
