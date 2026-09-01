$ErrorActionPreference = "Stop"

$expRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $expRoot)
$sourceRoot = Join-Path $projectRoot "data"
$targetRoot = Join-Path $expRoot "dataset_3b"

$files = @(
    @("ETT-small", "ETTh1.csv"),
    @("ETT-small", "ETTh2.csv"),
    @("ETT-small", "ETTm1.csv"),
    @("ETT-small", "ETTm2.csv"),
    @("weather", "weather.csv")
)

foreach ($item in $files) {
    $subdir = $item[0]
    $name = $item[1]
    $source = Join-Path (Join-Path $sourceRoot $subdir) $name
    $targetDir = Join-Path $targetRoot $subdir
    $target = Join-Path $targetDir $name
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Missing source dataset: $source"
    }
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    if (-not (Test-Path -LiteralPath $target)) {
        New-Item -ItemType HardLink -Path $target -Target $source | Out-Null
    }
    Write-Output "READY $target"
}

New-Item -ItemType Directory -Path (Join-Path $expRoot "outputs\checkpoints") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $expRoot "logs") -Force | Out-Null
$datasetAlias = Join-Path $expRoot "data"
if (-not (Test-Path -LiteralPath $datasetAlias)) {
    New-Item -ItemType Junction -Path $datasetAlias -Target $targetRoot | Out-Null
}
Write-Output "3B isolated data/output directories are ready."
