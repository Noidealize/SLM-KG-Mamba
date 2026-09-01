$ErrorActionPreference = "Continue"
$expRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $expRoot)
$python = "D:\ANACONDA\envs\pytorch\python.exe"
$preprocess = Join-Path $expRoot "preprocess_smetimes.py"
$llm = Join-Path $projectRoot "models\llm\llama-3.2-3b-instruct"

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONFAULTHANDLER = "1"

Set-Location $expRoot

foreach ($dataset in @("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather")) {
    Write-Output "===== PREPROCESS 3B $dataset ====="
    & $python -u $preprocess --dataset $dataset --batch_size 8 --num_workers 0 `
        --device auto --llm_ckp_dir $llm --local_files_only
    if ($LASTEXITCODE -ne 0) { throw "3B preprocessing failed for $dataset" }
}

Write-Output "ALL 3B PREPROCESSING DONE"
