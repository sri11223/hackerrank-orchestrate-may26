param(
    [string]$RepoUrl = "https://github.com/sri11223/hackerrank-orchestrate-may26.git",
    [switch]$Local
)

$ErrorActionPreference = "Stop"

function Invoke-UserPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PythonArgs)

    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 @PythonArgs
        return
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python @PythonArgs
        return
    }

    throw "Python was not found. Install Python 3.10+ from https://www.python.org/downloads/ and rerun this installer."
}

function Add-ToUserPath {
    param([string]$Directory)

    $existingUserPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $userParts = @()
    if ($existingUserPath) {
        $userParts = $existingUserPath -split ";" | Where-Object { $_ }
    }

    if ($userParts -notcontains $Directory) {
        $newUserPath = (($userParts + $Directory) -join ";")
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
        Write-Host "Added to User PATH: $Directory" -ForegroundColor Green
    } else {
        Write-Host "User PATH already contains: $Directory" -ForegroundColor DarkGray
    }

    $processParts = $env:Path -split ";" | Where-Object { $_ }
    if ($processParts -notcontains $Directory) {
        $env:Path = (($processParts + $Directory) -join ";")
    }
}

Write-Host "Installing Orchestrate triage CLI..." -ForegroundColor Cyan

$packageTarget = if ($Local) { $PSScriptRoot } else { "git+$RepoUrl" }
Invoke-UserPython -m pip install --user --upgrade --no-warn-script-location $packageTarget

$scriptsDir = (
    Invoke-UserPython -c "import sysconfig; print(sysconfig.get_path('scripts', scheme='nt_user') or sysconfig.get_path('scripts'))" |
        Select-Object -Last 1
).Trim()
Add-ToUserPath -Directory $scriptsDir

$triageExe = Join-Path $scriptsDir "triage.exe"
if (Test-Path -LiteralPath $triageExe) {
    Write-Host ""
    Write-Host "Verifying triage command..." -ForegroundColor Cyan
    & $triageExe --help | Select-Object -First 14
} else {
    Write-Host "triage.exe was not found in $scriptsDir. You can still run: python -m triage --help" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Ready. Try: triage run" -ForegroundColor Green
Write-Host "If an already-open Command Prompt still cannot find triage, open a new terminal once." -ForegroundColor DarkGray
