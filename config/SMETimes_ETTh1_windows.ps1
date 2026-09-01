param(
    [string]$Python = "D:\ANACONDA\envs\pytorch\python.exe",
    [string]$LlmCkpDir = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    [int]$BatchSize = 256,
    [int]$TrainEpochs = 2,
    [int]$NumWorkers = 0,
    [switch]$Smoke
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if ($Smoke) {
    $LlmCkpDir = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    $BatchSize = 512
    $TrainEpochs = 1
}

& $Python .\scripts\preprocess.py `
  --dataset ETTh1 `
  --llm_ckp_dir $LlmCkpDir `
  --device auto `
  --batch_size $BatchSize `
  --num_workers $NumWorkers
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python .\train.py `
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
  --batch_size $BatchSize `
  --learning_rate 0.001 `
  --mlp_hidden_layers 0 `
  --train_epochs $TrainEpochs `
  --gpu 0 `
  --device auto `
  --llm_ckp_dir $LlmCkpDir `
  --num_workers $NumWorkers `
  --cosine `
  --tmax 10 `
  --mix_embeds
