Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RequiredEnvironmentValue {
    param([Parameter(Mandatory = $true)][string]$Name)

    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Set $Name before running this script."
    }
    return $value
}

function Initialize-BreezeWindows {
    $repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
    $python = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Missing $python. Run 'uv sync --extra evaluation --group dev' from the repository root."
    }
    if ([string]::IsNullOrWhiteSpace($env:CUDA_VISIBLE_DEVICES)) {
        $env:CUDA_VISIBLE_DEVICES = "0"
    }
    Set-Location -LiteralPath $repoRoot
    return $python
}

function Invoke-BreezePython {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [string[]]$Command
    )

    $cleanCommand = @($Command | Where-Object { -not [string]::IsNullOrEmpty($_) })
    & $Python @cleanCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Python exited with code $LASTEXITCODE."
    }
}
