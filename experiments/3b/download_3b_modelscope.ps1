param(
    [int]$MaxRetries = 10,
    [int]$RetryDelaySeconds = 15
)

$ErrorActionPreference = "Continue"
# 关闭单个大文件的分段并发，强制使用支持 Range 的单流断点续传。
# 这与 CLI 的 --max-workers（控制多个文件并发）不是同一个设置。
$env:MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS = "1"
$modelscope = "D:\ANACONDA\envs\pytorch\Scripts\modelscope.exe"
$repo = "LLM-Research/Llama-3.2-3B-Instruct"
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$target = Join-Path $projectRoot "models\llm\llama-3.2-3b-instruct"

if (-not (Test-Path -LiteralPath $modelscope)) {
    throw "ModelScope command not found: $modelscope"
}
New-Item -ItemType Directory -Path $target -Force | Out-Null

# SMETimes/Transformers 所需文件。故意不下载 original/consolidated.00.pth，
# 因为它与 safetensors 权重重复，会额外占用约 5 GB。
$files = @(
    "config.json",
    "configuration.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "model.safetensors.index.json",
    "model-00002-of-00002.safetensors",
    "model-00001-of-00002.safetensors",
    "LICENSE.txt",
    "USE_POLICY.md",
    "README.md"
)

foreach ($file in $files) {
    $ok = $false
    for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
        Write-Output "===== DOWNLOAD $file (attempt $attempt/$MaxRetries) ====="
        & $modelscope download --model $repo $file --local-dir $target
        $downloaded = Join-Path $target $file
        if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $downloaded) -and
            (Get-Item -LiteralPath $downloaded).Length -gt 0) {
            $ok = $true
            break
        }
        if ($attempt -lt $MaxRetries) {
            Write-Output "Retrying $file after $RetryDelaySeconds seconds..."
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }
    if (-not $ok) {
        throw "Download failed after $MaxRetries attempts: $file"
    }
}

$requiredSizes = @{
    "config.json" = 100
    "tokenizer.json" = 1000000
    "model.safetensors.index.json" = 1000
    "model-00001-of-00002.safetensors" = 4000000000
    "model-00002-of-00002.safetensors" = 1000000000
}

foreach ($entry in $requiredSizes.GetEnumerator()) {
    $path = Join-Path $target $entry.Key
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Validation failed; missing file: $path"
    }
    $size = (Get-Item -LiteralPath $path).Length
    if ($size -lt $entry.Value) {
        throw "Validation failed; file is too small: $path ($size bytes)"
    }
}

$incomplete = Get-ChildItem -LiteralPath $target -Force -Recurse -File |
    Where-Object { $_.Name -like "*.incomplete" }
if ($incomplete) {
    throw "Validation failed; incomplete download files remain under $target"
}

& "D:\ANACONDA\envs\pytorch\python.exe" -c `
    "from transformers import AutoConfig, AutoTokenizer; p=r'$target'; c=AutoConfig.from_pretrained(p, local_files_only=True); t=AutoTokenizer.from_pretrained(p, local_files_only=True); print('model_type=', c.model_type); print('hidden_size=', c.hidden_size); print('vocab_size=', len(t))"
if ($LASTEXITCODE -ne 0) {
    throw "Transformers config/tokenizer validation failed"
}

Write-Output "DOWNLOAD AND VALIDATION COMPLETE: $target"
