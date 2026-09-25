param(
    [Parameter(Mandatory = $true, Position = 0)][string]$ModelRoot,
    [string]$Adapter,
    [string]$AdaptersDir,
    [string]$ListenAddress = "127.0.0.1",
    [int]$Port = 7860,
    [switch]$NoFastAll,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$RemainingArgs
)

. (Join-Path $PSScriptRoot "_common.ps1")
$python = Initialize-BreezeWindows
$command = @(
    "-m", "breeze_infer.api", $ModelRoot,
    "--host", $ListenAddress,
    "--port", $Port.ToString()
)
if (-not [string]::IsNullOrWhiteSpace($Adapter)) {
    $command += @("--adapter", $Adapter)
}
if (-not [string]::IsNullOrWhiteSpace($AdaptersDir)) {
    $command += @("--adapters-dir", $AdaptersDir)
}
if ($NoFastAll) {
    $command += "--no-fast-all"
}
$command += $RemainingArgs
Invoke-BreezePython -Python $python -Command $command
