param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

. (Join-Path $PSScriptRoot "_common.ps1")
$python = Initialize-BreezeWindows
$modelRoot = Get-RequiredEnvironmentValue "BREEZE_MODEL_ROOT"
$trainManifest = Get-RequiredEnvironmentValue "BREEZE_TRAIN_MANIFEST"
$validationManifest = Get-RequiredEnvironmentValue "BREEZE_VALIDATION_MANIFEST"
$cacheRoot = Get-RequiredEnvironmentValue "BREEZE_CACHE_ROOT"

$command = @(
    "-m", "training.real_data",
    "--model-root", $modelRoot,
    "--train-manifest", $trainManifest,
    "--validation-manifest", $validationManifest,
    "--output-root", $cacheRoot
) + $RemainingArgs
Invoke-BreezePython -Python $python -Command $command
