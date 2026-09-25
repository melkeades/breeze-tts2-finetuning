param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

. (Join-Path $PSScriptRoot "_common.ps1")
$python = Initialize-BreezeWindows
$modelRoot = Get-RequiredEnvironmentValue "BREEZE_MODEL_ROOT"
$cacheRoot = Get-RequiredEnvironmentValue "BREEZE_CACHE_ROOT"
$sweepPlan = Get-RequiredEnvironmentValue "BREEZE_SWEEP_PLAN"
$runRoot = Get-RequiredEnvironmentValue "BREEZE_RUN_ROOT"

$command = @(
    "-m", "training.full_sft_sweep",
    "--plan", $sweepPlan,
    "--model-root", $modelRoot,
    "--cache-root", $cacheRoot,
    "--output-root", $runRoot
) + $RemainingArgs
Invoke-BreezePython -Python $python -Command $command
