param(
    [Parameter(Mandatory = $true, Position = 0)][string]$ModelRoot,
    [Parameter(Mandatory = $true)][string]$Text,
    [string]$Output = "output.wav",
    [string]$Adapter,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$RemainingArgs
)

. (Join-Path $PSScriptRoot "_common.ps1")
$python = Initialize-BreezeWindows
$command = @("infer.py", $ModelRoot, "--text", $Text, "--output", $Output)
if (-not [string]::IsNullOrWhiteSpace($Adapter)) {
    $command += @("--adapter", $Adapter)
}
$command += $RemainingArgs
Invoke-BreezePython -Python $python -Command $command
