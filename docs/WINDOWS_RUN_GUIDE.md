# Windows 下载和运行说明

本文说明国内用户如何从魔搭社区下载 1B/3B 模型，并在本项目中运行。

## 1. 先安装 ModelScope

在项目根目录运行：

```powershell
cd D:\VSCODEPROJECT\llmprediction

& "D:\ANACONDA\envs\pytorch\python.exe" -m pip install modelscope
```

## 2. 下载 1B 模型

推荐先用 1B 跑通代码。1B 对 8GB 显存更友好。

魔搭模型：

```text
LLM-Research/Llama-3.2-1B-Instruct
```

下载命令：

```powershell
& "D:\ANACONDA\envs\pytorch\python.exe" -c "from modelscope import snapshot_download; snapshot_download('LLM-Research/Llama-3.2-1B-Instruct', local_dir=r'.\models\llm\llama-3.2-1b-instruct')"
```

下载后本地目录应为：

```text
D:\VSCODEPROJECT\llmprediction\models\llm\llama-3.2-1b-instruct
```

## 3. 下载 3B 模型

如果要更接近论文设置，可以下载 3B。3B 更吃显存，建议 12GB 显存起步，8GB 显存可能会 OOM。

魔搭模型：

```text
LLM-Research/Llama-3.2-3B-Instruct
```

下载命令：

```powershell
& "D:\ANACONDA\envs\pytorch\python.exe" -c "from modelscope import snapshot_download; snapshot_download('LLM-Research/Llama-3.2-3B-Instruct', local_dir=r'.\models\llm\llama-3.2-3b-instruct')"
```

下载后本地目录应为：

```text
D:\VSCODEPROJECT\llmprediction\models\llm\llama-3.2-3b-instruct
```

## 4. 检查模型是否下载成功

检查 1B：

```powershell
& "D:\ANACONDA\envs\pytorch\python.exe" -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained(r'.\models\llm\llama-3.2-1b-instruct'); print(c.model_type, c.hidden_size)"
```

检查 3B：

```powershell
& "D:\ANACONDA\envs\pytorch\python.exe" -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained(r'.\models\llm\llama-3.2-3b-instruct'); print(c.model_type, c.hidden_size)"
```

能输出 `llama 2048` 或 `llama 3072` 就说明模型目录可用。

## 5. 预处理

用 1B：

```powershell
& "D:\ANACONDA\envs\pytorch\python.exe" .\scripts\preprocess.py `
  --dataset ETTh1 `
  --llm_ckp_dir .\models\llm\llama-3.2-1b-instruct `
  --device auto `
  --batch_size 1 `
  --num_workers 0
```

用 3B：

```powershell
& "D:\ANACONDA\envs\pytorch\python.exe" .\scripts\preprocess.py `
  --dataset ETTh1 `
  --llm_ckp_dir .\models\llm\llama-3.2-3b-instruct `
  --device auto `
  --batch_size 1 `
  --num_workers 0
```

预处理会生成：

```text
data\ETT-small\ETTh1.pt
```

换模型后必须重新预处理。

## 6. 训练

用 1B：

```powershell
.\config\SMETimes_ETTh1_windows.ps1 `
  -LlmCkpDir .\models\llm\llama-3.2-1b-instruct `
  -BatchSize 4 `
  -TrainEpochs 2
```

用 3B：

```powershell
.\config\SMETimes_ETTh1_windows.ps1 `
  -LlmCkpDir .\models\llm\llama-3.2-3b-instruct `
  -BatchSize 1 `
  -TrainEpochs 2
```

如果显存不足，把 `-BatchSize` 改成 `1`。

## 7. 模型选择建议

- 只是想跑通代码：下载 1B。
- 8GB 显存：优先 1B。
- 想尽量复现论文：下载 3B。
- 3B 建议 12GB 或更高显存。

魔搭社区链接：

- 1B：https://modelscope.cn/models/LLM-Research/Llama-3.2-1B-Instruct
- 3B：https://modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct
