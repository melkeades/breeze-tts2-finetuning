param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

. (Join-Path $PSScriptRoot "_common.ps1")
$python = Initialize-BreezeWindows
$modelRoot = Get-RequiredEnvironmentValue "BREEZE_MODEL_ROOT"
$cacheRoot = Get-RequiredEnvironmentValue "BREEZE_CACHE_ROOT"
$runRoot = Get-RequiredEnvironmentValue "BREEZE_RUN_ROOT"

$command = @(
    "-m", "training.real_lora",
    "--model-root", $modelRoot,
    "--cache-root", $cacheRoot,
    "--output-root", $runRoot
) + $RemainingArgs
Invoke-BreezePython -Python $python -Command $command
