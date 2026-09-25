param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

. (Join-Path $PSScriptRoot "_common.ps1")
$python = Initialize-BreezeWindows
$modelRoot = Get-RequiredEnvironmentValue "BREEZE_MODEL_ROOT"
$trainAudio = Get-RequiredEnvironmentValue "BREEZE_TRAIN_AUDIO"
$trainTranscript = Get-RequiredEnvironmentValue "BREEZE_TRAIN_TRANSCRIPT"
$runRoot = Get-RequiredEnvironmentValue "BREEZE_RUN_ROOT"
$device = if ([string]::IsNullOrWhiteSpace($env:BREEZE_DEVICE)) { "cuda:0" } else { $env:BREEZE_DEVICE }

$smokeCommand = @(
    "-m", "training.full_sft_smoke",
    "--model-root", $modelRoot,
    "--audio", $trainAudio,
    "--transcript", $trainTranscript,
    "--output-root", $runRoot,
    "--device", $device
) + $RemainingArgs
Invoke-BreezePython -Python $python -Command $smokeCommand
Invoke-BreezePython -Python $python -Command @(
    "-m", "training.reload_smoke",
    "--run-root", $runRoot,
    "--device", $device
)
