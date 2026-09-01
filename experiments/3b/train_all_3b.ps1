# 串行训练论文基准数据集（共享 GPU，避免显存冲突）
# 注意：不能用 $ErrorActionPreference="Stop"——PS5.1 会把 python 的 stderr（tqdm 进度条）
# 包装成 NativeCommandError 并中断脚本。改为检查 $LASTEXITCODE。
$ErrorActionPreference = "Continue"
$expRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $expRoot)
$python = "D:\ANACONDA\envs\pytorch\python.exe"
$train = Join-Path $expRoot "train_smetimes.py"
$llm = Join-Path $projectRoot "models\llm\llama-3.2-3b-instruct"
$batchSize = 8

# stdout 走管道是块缓冲，进程异常退出会丢失缓冲内输出；-u + 无缓冲让日志实时落盘。
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONFAULTHANDLER = "1"

Set-Location $expRoot

# root_path / data_path / --data / model_id
$jobs = @(
    @("ETTh1_672_96_3b",  ".\dataset_3b\ETT-small\", "ETTh1.csv",  "ETTh1"),
    @("ETTh2_672_96_3b",  ".\dataset_3b\ETT-small\", "ETTh2.csv",  "ETTh2"),
    @("ETTm1_672_96_3b",  ".\dataset_3b\ETT-small\", "ETTm1.csv",  "ETTm1"),
    @("ETTm2_672_96_3b",  ".\dataset_3b\ETT-small\", "ETTm2.csv",  "ETTm2"),
    @("weather_672_96_3b", ".\dataset_3b\weather\",   "weather.csv", "weather")
)

foreach ($j in $jobs) {
    $id = $j[0]; $root = $j[1]; $csv = $j[2]; $dtype = $j[3]
    Write-Output "===== TRAIN $id ($csv) ====="
    & $python -u $train `
        --task_name long_term_forecast --is_training 1 `
        --model SMETimes_Llama --data $dtype `
        --root_path $root --data_path $csv `
        --model_id $id `
        --seq_len 672 --label_len 576 --token_len 96 `
        --test_seq_len 672 --test_label_len 576 --test_pred_len 96 `
        --batch_size $batchSize --learning_rate 0.001 `
        --mlp_hidden_layers 0 --train_epochs 10 `
        --gpu 0 --device auto `
        --llm_ckp_dir $llm --num_workers 0 `
        --cosine --tmax 10 --mix_embeds `
        --checkpoints .\outputs\checkpoints\ --local_files_only --des llama3_2_3b `
        --no_use_amp
    if ($LASTEXITCODE -ne 0) { throw "training failed for $id" }
}

Write-Output "ALL TRAINING DONE"
