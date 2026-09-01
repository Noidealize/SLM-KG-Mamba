# SMETimes Windows Run Guide

This project can run on Windows after generating the timestamp embedding `.pt`
files and providing a Hugging Face language model.

## Python Environment

The expected environment for this workspace is:

```powershell
D:\ANACONDA\envs\pytorch\python.exe
```

Install missing dependencies with:

```powershell
& "D:\ANACONDA\envs\pytorch\python.exe" -m pip install -r requirements.txt
```

## Free Model Choices

Recommended Llama-compatible free model:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Other supported free choices:

```text
openai-community/gpt2-large
facebook/opt-1.3b
```

Download TinyLlama to a local folder:

```powershell
& "D:\ANACONDA\envs\pytorch\python.exe" .\scripts\download_model.py `
  --repo_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
  --local_dir .\models\llm\tinyllama-1.1b-chat
```

You can also pass the Hugging Face repo id directly to `--llm_ckp_dir`; Transformers
will download it to the Hugging Face cache automatically.

## Quick Smoke Test

This uses a tiny random Llama model to verify the Windows/Python pipeline quickly.
It is only for code validation, not for meaningful forecasting quality.

```powershell
.\config\SMETimes_ETTh1_windows.ps1 -Smoke
```

## Run ETTh1 With TinyLlama

Generate timestamp embeddings first. The `.pt` file is saved next to the CSV:

```powershell
& "D:\ANACONDA\envs\pytorch\python.exe" .\scripts\preprocess.py `
  --dataset ETTh1 `
  --llm_ckp_dir .\models\llm\tinyllama-1.1b-chat `
  --device auto
```

Train and test:

```powershell
& "D:\ANACONDA\envs\pytorch\python.exe" .\train.py `
  --task_name long_term_forecast `
  --is_training 1 `
  --root_path .\data\ETT-small\ `
  --data_path ETTh1.csv `
  --model_id ETTh1_672_96 `
  --model SMETimes_Llama `
  --data ETTh1 `
  --seq_len 672 `
  --label_len 576 `
  --token_len 96 `
  --test_seq_len 672 `
  --test_label_len 576 `
  --test_pred_len 96 `
  --batch_size 64 `
  --learning_rate 0.001 `
  --mlp_hidden_layers 0 `
  --train_epochs 2 `
  --gpu 0 `
  --device auto `
  --llm_ckp_dir .\models\llm\tinyllama-1.1b-chat `
  --num_workers 0 `
  --cosine `
  --tmax 10 `
  --mix_embeds
```

## Notes

- If you change `--llm_ckp_dir`, rerun the matching preprocess script so the
  `.pt` embedding dimension matches the training model.
- Use `preprocess_gpt.py` with `SMETimes_Gpt2`, and `preprocess_opt.py` with
  `SMETimes_Opt_1b`.
- On Windows, `--num_workers 0` is the safest DataLoader setting.
- The original bash scripts remain for Linux; use the `.ps1` script on Windows.
