Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath ([System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..")))

foreach ($required in @("LICENSE", "NOTICE", "MODEL_LICENSE.md", "README.md")) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf) -or (Get-Item -LiteralPath $required).Length -eq 0) {
        throw "Missing required release file: $required"
    }
}

foreach ($pattern in @("*.wav", "*.mp3", "*.flac", "*.pt", "*.pth", "*.ckpt", "*.bin", "*.safetensors", "*.onnx", "*.npy", "*.npz")) {
    $tracked = @(& git ls-files $pattern)
    if ($LASTEXITCODE -ne 0) {
        throw "git ls-files failed for $pattern"
    }
    if ($tracked.Count -gt 0) {
        throw "Tracked model, audio, or tensor artifact matches $pattern"
    }
}

$privatePattern = '/Users/[^/]+/|/home/[^/]+/|/mnt/(work|ext4)|FEMALE_01|female01'
& git grep -I -n -E $privatePattern -- . ':!scripts/check_public_tree.ps1' ':!training/release_bundle.py'
if ($LASTEXITCODE -eq 0) {
    throw "Tracked source contains a private path or study identifier"
}
if ($LASTEXITCODE -gt 1) {
    throw "git grep failed with code $LASTEXITCODE"
}

Write-Output "public source contract passed"
$global:LASTEXITCODE = 0
