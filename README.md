# llmprediction

基于轻量级 LLM 的时间序列预测实验项目：SMETimes（Small but Mighty）论文复现（Windows / RTX 5070 Ti 适配）及三个扩展实验。

## 目录结构

```
llmprediction/
├── train.py / test.py       # 训练 / 测试入口（原 run.py，test.py 等价 --is_training 0）
├── config/                  # 各数据集的运行配置（.sh / .ps1，含超参数组合）
├── data/                    # 数据集（ETT-small、weather；CSV 在此，.pt 嵌入由预处理生成）
├── models/                  # 模型代码 + models/llm/ 预训练权重（11G+，不入库）
├── utils/ layers/ data_provider/ exp/   # SMETimes 主体代码
├── scripts/                 # 预处理与模型下载脚本（preprocess*.py、download_model.py）
├── results/                 # 训练产物：checkpoints/、test_results/、result_long_term_forecast.txt
├── experiments/
│   ├── 3b/                  # Llama-3.2-3B 训练流水线（见 3B_完整操作说明.txt）
│   ├── effectiveness/       # 小模型有效性因果实验（预训练 vs 随机权重，见其 README）
│   └── rul_semantic/        # MA-RDG-Mamba 语义骨干重构（RUL 任务，见其 README）
├── references/              # 原始上游仓库 SMETimes-main（只读归档）
├── archives/                # 历史压缩包
└── docs/                    # 论文、论文中文翻译、Windows 运行说明
```

## 快速开始（Windows）

环境：`D:\ANACONDA\envs\pytorch`（torch 2.12.1+cu132、transformers 5.15.0）。

```powershell
# 1. 准备预训练权重（models/llm/ 下，可用 ModelScope 代替）
python scripts\download_model.py --repo_id TinyLlama/TinyLlama-1.1B-Chat-v1.0
# 或按 docs\Windows下载和运行说明_本机适配.md 用魔搭下载 llama-3.2-1b/3b-instruct

# 2. 生成时间嵌入（换 LLM 后必须重新预处理）
python scripts\preprocess.py --dataset ETTh1 --llm_ckp_dir .\models\llm\llama-3.2-1b-instruct

# 3. 训练（也可直接运行 config\SMETimes_ETTh1_windows.ps1，支持 -Smoke 冒烟）
python train.py --task_name long_term_forecast --is_training 1 `
  --root_path .\data\ETT-small\ --data_path ETTh1.csv --data ETTh1 `
  --model_id ETTh1_672_96 --model SMETimes_Llama `
  --seq_len 672 --label_len 576 --token_len 96 `
  --test_seq_len 672 --test_label_len 576 --test_pred_len 96 `
  --llm_ckp_dir .\models\llm\llama-3.2-1b-instruct --num_workers 0

# 4. 测试
python test.py --task_name long_term_forecast --model_id ETTh1_672_96 `
  --root_path .\data\ETT-small\ --data_path ETTh1.csv --data ETTh1 `
  --seq_len 672 --label_len 576 --token_len 96 `
  --test_seq_len 672 --test_label_len 576 --test_pred_len 96 `
  --llm_ckp_dir .\models\llm\llama-3.2-1b-instruct --num_workers 0
```

Linux 用户可参考 `config/*.sh`（从仓库根运行，如 `bash config/SMETimes_ETTh1.sh`）。

## 三个实验

| 实验 | 问题 | 入口 |
|---|---|---|
| `experiments/3b` | Llama-3.2-3B 能否在 16GB 显卡上稳定训练（Hybrid-Adam 等本机防护） | `3B_完整操作说明.txt`、`train_all_3b.ps1` |
| `experiments/effectiveness` | 冻结预训练权重是否带来统计显著且实际意义的预测增益 | `run_sweep.py`（先 `--dry-run`），详见其 README |
| `experiments/rul_semantic` | RUL 预测中语义骨干的替换性（TransE / SLM / 无） | 依赖外部 `MA_RDG_Mamba_RUL_complete_v1`，见其 README |

## 其他

- 数据集来源：ETT（ETT-small）、Weather；`config/` 中 ECL/Traffic/Solar 的配置保留但数据未下载。
- `references/SMETimes-main` 为上游原始仓库（只读），主代码已按本仓库布局平铺到根目录。
- 论文与中文翻译见 `docs/`。
