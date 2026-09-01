# SMETimes 小模型有效性验证实验

这个目录用来回答一个严格、可证伪的问题：

> 在 ETTh1 的 `672→96` 预测任务中，冻结的 1B Llama **预训练权重本身**，是否比同架构、同参数量、同样冻结的随机权重带来稳定且具有实际意义的预测增益？

它不修改仓库根的主代码（原 `SMETimes-main-WINDOWS`）、不覆盖原有 1B/3B 结果，并把每个变体和 seed 保存到独立目录。

## 1. 能证明什么，不能证明什么

当前 ETTh1 prompt 只含时间范围和 OT 通道的均值、标准差、变化量，没有外部故障知识或领域知识。因此：

- `pretrained_no_context` 优于 `random_no_context`：支持“预训练权重有贡献”；
- `pretrained_full` 优于 `pretrained_no_context`：只支持“额外的时间/OT 统计上下文支路有贡献”；
- `pretrained_full` 优于 `pretrained_shuffled_context`：支持“上下文与样本对齐有贡献”；
- 上述结果都不能单独写成“小模型的领域知识有效”或“知识图谱有效”。

若论文要主张“语言表示/知识增强”，第二阶段还必须加入 `P-RawMeta`：把完全相同的时间和统计量直接交给小型 MLP，再与文本嵌入比较。当前版本没有伪装成已经完成这一层归因。

另一个实现边界是：当前默认 `num_experts=1`，softmax gate 恒为 1，所谓 MoE 在这套配置下实际上退化为一个线性变换；本实验不把 MoE 当作已验证创新点。

## 2. 已实现的五个变体

| 变体 | 上下文 | 冻结骨干 | 回答的问题 |
|---|---|---|---|
| `pretrained_full` | 对齐的现有 embedding | 预训练 1B | 完整 SMETimes 参考组 |
| `pretrained_no_context` | 无 | 预训练 1B | 主实验的处理组 |
| `random_no_context` | 无 | 同架构随机 1B | 主实验的严格随机权重对照 |
| `identity_no_context` | 无 | Identity | 冻结 Llama 变换相对 Identity 路径的差异 |
| `pretrained_shuffled_context` | split 内固定置乱 | 预训练 1B | 上下文对齐的探索性对照 |

主比较固定为：

```text
pretrained_no_context  vs  random_no_context
```

二者的外部可训练层使用相同 seed 初始化；随机 Llama 的初始化使用独立 RNG，不改变 adapter、dropout 或 DataLoader 的随机序列。

## 3. 为什么这个运行器比原脚本更适合做证据

- seed 显式写入命令、目录、配置和结果；
- DataLoader 使用 split 独立、显式播种的 generator；
- 训练阶段只创建 train/validation，不查看 test loss；
- checkpoint 用验证集最后 96 步 MSE 选择，与 test 主终点一致；训练目标仍保持上游完整 next-patch 序列；
- test 阶段必须通过训练协议哈希和预先锁定的分析计划，并且完整测试只允许写一次；
- checkpoint 只保存可训练参数，避免把冻结 1B 权重重复保存几百 MB；
- 保存 `predictions.npy`、`targets.npy` 和逐样本误差，结果可复算；
- 记录数据、embedding、模型架构配置、完整权重、源码与环境的 SHA256，以及参数量、时长和峰值显存；
- 禁止 `pred_len>96`，避免上游滚动预测复用未来统计 embedding；
- 统计以“配对训练 seed”为独立单位，不把 19495 个重叠窗口冒充独立样本。
- 确认性运行遇到不支持确定性实现的算子会直接失败，不以警告方式继续。

checkpoint 的 SHA256 用于发现文件在锁定后是否被改动；PyTorch 序列化容器本身可能含非规范化元数据，因此它不被用作“不同运行生成了逐字节相同模型”的证明。模型初始化的一致性由规范化张量审计哈希另行验证。

## 4. 推荐执行顺序（Anaconda Prompt / CMD）

先校验数据、embedding、1B 配置及实验约束；这一步不加载 1B 权重：

```cmd
cd /d D:\VSCODEPROJECT\llmprediction\experiments\effectiveness
D:\ANACONDA\envs\pytorch\python.exe run_one.py --stage validate --output-root outputs\validation
```

先查看 smoke 矩阵，不运行：

```cmd
cd /d D:\VSCODEPROJECT\llmprediction\experiments\effectiveness
D:\ANACONDA\envs\pytorch\python.exe run_sweep.py --preset smoke --stage train --dry-run
```

非论文证据的 smoke（五组 × 1 seed × 每个 split 仅 2 batches）可以一次训练并测试。输出会明确标记 `diagnostic_truncated=true`，汇总器不会把它纳入正式判据：

```cmd
cd /d D:\VSCODEPROJECT\llmprediction\experiments\effectiveness
D:\ANACONDA\envs\pytorch\python.exe run_sweep.py --preset smoke --stage both
```

Pilot 只训练并查看验证集，测试集继续封存：

```cmd
cd /d D:\VSCODEPROJECT\llmprediction\experiments\effectiveness
D:\ANACONDA\envs\pytorch\python.exe run_sweep.py --preset pilot --stage train
```

如果 pilot 方向稳定，先确定超参数和 1% 最小实际效应，再用一个全新的输出根目录训练 10 个配对 seed 的主对照：

```cmd
cd /d D:\VSCODEPROJECT\llmprediction\experiments\effectiveness
D:\ANACONDA\envs\pytorch\python.exe run_sweep.py --preset confirmatory --stage train --output-root outputs\confirmatory_v1
```

训练完成、且尚不存在任何 test 指标时，生成一次性分析计划锁。脚本会核验 10 对运行的共同协议、参数量、adapter 初始哈希、checkpoint 哈希，以及预训练/随机骨干哈希确实不同；锁文件已存在时拒绝覆盖：

```cmd
cd /d D:\VSCODEPROJECT\llmprediction\experiments\effectiveness
D:\ANACONDA\envs\pytorch\python.exe lock_analysis_plan.py --output-root outputs\confirmatory_v1 --expected-seeds 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 --minimum-relative-improvement 0.01 --mae-noninferiority-margin 0.01 --minimum-win-fraction 0.80
```

锁定后才执行一次确认性测试；没有有效锁文件时，完整测试会直接拒绝运行：

```cmd
cd /d D:\VSCODEPROJECT\llmprediction\experiments\effectiveness
D:\ANACONDA\envs\pytorch\python.exe run_sweep.py --preset confirmatory --stage test --output-root outputs\confirmatory_v1
```

重新汇总（不会重新训练或测试）：

```cmd
cd /d D:\VSCODEPROJECT\llmprediction\experiments\effectiveness
D:\ANACONDA\envs\pytorch\python.exe summarize_results.py --output-root outputs\confirmatory_v1
```

## 5. 预先锁定的主判据

默认的 1% 只是一个待作者确认的建议，必须由 `analysis_plan.lock.json` 在打开确认性 test 之前锁定。汇总命令不能事后重新定义它。当前自动判据要求：

1. 共有至少 10 个相同 seed 的有效配对；
2. `MSE(pretrained) / MSE(random)` 的配对 95% CI 上界 `< 0.99`；
3. 至少锁定比例（默认 80%，10 对时为 8 对）的 seed 中 pretrained 的 MSE 更低；
4. MAE 的 95% CI 排除超过 1% 的恶化。

同时报告绝对差、几何均值比、相对改善、Student-t 95% CI、seed bootstrap 95% CI 和配对 sign-flip p 值。p 值不是单独的成功条件。sign-flip 作用于配对 log-ratio，并假定 seed 级差值可交换/近似对称；t 区间假定 seed 级 log-ratio 近似独立且正态。MSE 和 MAE 的探索性比较分别构成各自的 Holm 校正家族。

主区间是 seed 级配对 log-ratio 的 Student-t 区间；seed bootstrap 仅作为敏感性分析。由于 ETTh1 只提供一个固定测试时期，这些区间描述的是“给定该测试时期时，训练 seed 带来的不确定性”，不能替代跨时期或跨数据集外推。若论文要讨论对其他运行周期的泛化，应再按预测起点做 moving-block bootstrap，或增加独立时期/独立数据集复现。

如果 CI 跨过 `1` 和 `0.99`，结论是“证据不足”，不是“已经证明无效”；若 90% CI 完全落入 `[0.99, 1.01]`，才有资格进一步讨论实际等价。

## 6. 输出结构

```text
outputs/<阶段>/
├── analysis_plan.lock.json
├── runs/<variant>/seed_<seed>/
│   ├── config.json
│   ├── environment.json
│   ├── model_audit.json
│   ├── train_summary.json
│   ├── checkpoint/trainable_state.pth
│   ├── TEST_CLAIM.lock
│   ├── TEST_COMPLETE.json
│   ├── metrics.json
│   ├── predictions.npy
│   ├── targets.npy
│   └── errors_by_sample.npz
└── summary/
    ├── runs.csv
    ├── validation_variant_summary.csv
    ├── validation_paired_comparisons.csv
    ├── test_variant_summary.csv
    ├── test_paired_comparisons.csv
    └── decision.json
```

`TEST_CLAIM.lock` 在完整测试开始前以独占方式创建，成功后才写入 `TEST_COMPLETE.json`。如果进程在二者之间崩溃，该运行会“失败关闭”，不会自动重测或覆盖；应保留现场并换一个新的输出根目录重新进行整套确认性实验。训练 checkpoint 只保存可训练权重，不包含优化器与 RNG 状态，因此中断的正式训练同样不做中途续跑。

## 7. 后续论文级扩展

完成主对照后，再按顺序加入：

1. `P-RawMeta`，区分“语言表示”与“多给了有用数值”；
2. DLinear、PatchTST 或你的 Mamba，比较 MSE—训练时间—显存 Pareto 前沿；
3. CWRU 分类或 C-MAPSS RUL，但每个任务分别定义标签、指标与基线，不能把 ETTh1 结果直接外推；
4. 1B 通过后再验证 3B。1B/3B 的 hidden size 会改变外围可训练参数量，所以不能只凭 3B 更好就归因于“知识更多”。

现有 `experiments\3b` 的坏样本替换逻辑可能只替换输入而不同比替换标签；修正并审计 guard/skip 计数前，不应把当前 3B wrapper 用于正式因果结论。
