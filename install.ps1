# Studio3D installer / updater for Windows 10/11 (x64, NVIDIA GPU)
#
# Run in PowerShell (no admin needed):
#   irm https://raw.githubusercontent.com/ahmad-ksa/3D/main/install.ps1 | iex
#
# Designed to work with Windows 11 Smart App Control (SAC) ON:
#  * Nothing is downloaded through a web browser, so no file gets the
#    "Mark of the Web" that SAC uses to block scripts/installers.
#  * Only the official, digitally signed python.exe (Python Software Foundation)
#    and Microsoft Edge are executed. Studio3D itself is plain .py source code.
#  * pip is always run as "python -m pip" (never the unsigned pip.exe launcher).
#  * Uses only basic cmdlets, so it also works in PowerShell Constrained Language Mode.
#
# Re-run the same command at any time to update.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Repo        = 'ahmad-ksa/3D'
$Branch      = 'main'
$PyVersion   = '3.11.9'
$HySha       = 'f8db63096c8282cb27354314d896feba5ba6ff8a'   # Tencent Hunyuan3D-2 (pinned)
$TorchIndex  = 'https://download.pytorch.org/whl/cu128'     # CUDA 12.8 => RTX 50xx support

$Root   = Join-Path $env:LOCALAPPDATA 'Studio3D'
$PyDir  = Join-Path $Root 'python'
$Py     = Join-Path $PyDir 'tools\python.exe'
$App    = Join-Path $Root 'app'
$Tmp    = Join-Path $env:TEMP 'studio3d-install'

function Step($n, $msg) { Write-Host ''; Write-Host "[$n/6] $msg" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }

Write-Host '==============================================' -ForegroundColor Green
Write-Host '   Studio3D - local AI 3D generator installer ' -ForegroundColor Green
Write-Host '==============================================' -ForegroundColor Green
Write-Host "Install folder: $Root"

New-Item -ItemType Directory -Force -Path $Root, $Tmp | Out-Null

# --- 0. GPU check -----------------------------------------------------------
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpu = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
    Write-Host "GPU: $gpu"
} else {
    Write-Host 'WARNING: NVIDIA driver not found. Install the latest NVIDIA driver (570 or newer for RTX 50xx).' -ForegroundColor Yellow
}

# --- 1. Python (official signed build, private copy) ------------------------
Step 1 "Python $PyVersion"
if (-not (Test-Path $Py)) {
    $pyZip = Join-Path $Tmp 'python.zip'
    Write-Host 'Downloading official Python from python.org (NuGet package)...'
    Invoke-WebRequest -UseBasicParsing -Uri "https://www.nuget.org/api/v2/package/python/$PyVersion" -OutFile $pyZip
    if (Test-Path $PyDir) { Remove-Item -Recurse -Force $PyDir }
    Expand-Archive -Path $pyZip -DestinationPath $PyDir -Force
}
& $Py -m ensurepip --upgrade | Out-Null
Check 'ensurepip'
& $Py -m pip install --upgrade --no-warn-script-location --disable-pip-version-check pip
Check 'pip upgrade'

# --- 2. Studio3D source -----------------------------------------------------
Step 2 'Studio3D program files'
if (Test-Path $App) { Remove-Item -Recurse -Force $App }
if ($env:STUDIO3D_SRC) {
    # local source folder (used by automated tests)
    Copy-Item -Recurse -Path $env:STUDIO3D_SRC -Destination $App
} else {
    $srcZip = Join-Path $Tmp 'src.zip'
    $srcDir = Join-Path $Tmp 'src'
    Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/$Repo/archive/refs/heads/$Branch.zip" -OutFile $srcZip
    if (Test-Path $srcDir) { Remove-Item -Recurse -Force $srcDir }
    Expand-Archive -Path $srcZip -DestinationPath $srcDir -Force
    $extracted = Get-ChildItem -Path $srcDir -Directory | Select-Object -First 1
    Move-Item -Path $extracted.FullName -Destination $App
}

# --- 3. PyTorch (CUDA 12.8) -------------------------------------------------
Step 3 'PyTorch with CUDA 12.8 (about 3 GB, please wait)'
& $Py -m pip install --no-warn-script-location --disable-pip-version-check `
    torch==2.7.1 torchvision==0.22.1 --index-url $TorchIndex
Check 'PyTorch install'

# --- 4. Other libraries -----------------------------------------------------
Step 4 'AI and 3D libraries'
& $Py -m pip install --no-warn-script-location --disable-pip-version-check -r (Join-Path $App 'requirements.txt')
Check 'requirements install'
& $Py -m pip install --no-warn-script-location --disable-pip-version-check --no-deps --force-reinstall `
    "https://github.com/Tencent-Hunyuan/Hunyuan3D-2/archive/$HySha.zip"
Check 'Hunyuan3D install'

# --- 5. Launcher + desktop shortcut -----------------------------------------
Step 5 'Launcher'
$launcher = Join-Path $Root 'Studio3D.cmd'
$cmd = @"
@echo off
chcp 65001 >nul
title Studio3D
cd /d "%~dp0app"
set PYTHONPATH=%~dp0app
"%~dp0python\tools\python.exe" -m studio3d %*
if errorlevel 1 pause
"@
Set-Content -Path $launcher -Value $cmd -Encoding ASCII

$desktop = Join-Path $env:USERPROFILE 'Desktop'
$oneDriveDesktop = Join-Path $env:USERPROFILE 'OneDrive\Desktop'
if (-not (Test-Path $desktop) -and (Test-Path $oneDriveDesktop)) { $desktop = $oneDriveDesktop }
$made = $false
try {
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut((Join-Path $desktop 'Studio3D.lnk'))
    $lnk.TargetPath = $launcher
    $lnk.WorkingDirectory = $Root
    $lnk.IconLocation = "$env:SystemRoot\System32\imageres.dll,186"
    $lnk.Description = 'Studio3D - AI 3D model generator'
    $lnk.Save()
    $made = $true
} catch { }
if (-not $made) { Copy-Item $launcher (Join-Path $desktop 'Studio3D.cmd') -Force }
Write-Host "Desktop shortcut created: $desktop"

# Connect to the Claude desktop app (MCP) so Claude can generate 3D models itself
$env:PYTHONPATH = $App
Push-Location $App
& $Py -m studio3d.claude_setup
Pop-Location
New-Item -ItemType Directory -Force -Path (Join-Path $env:USERPROFILE 'Pictures\Studio3D') | Out-Null
# Claude desktop extension (one click install: double-click the file, or Settings > Extensions)
$mcpb = Join-Path $App 'dist\studio3d.mcpb'
if (Test-Path $mcpb) {
    Copy-Item $mcpb (Join-Path $Root 'Studio3D-Claude.mcpb') -Force
    Copy-Item $mcpb (Join-Path $desktop 'Studio3D-Claude.mcpb') -Force
    Write-Host "Claude extension: double-click 'Studio3D-Claude.mcpb' on your desktop to add Studio3D to Claude."
}

# --- 6. Smart App Control check ---------------------------------------------
Step 6 'Smart App Control compatibility check'
$env:PYTHONPATH = $App
Push-Location $App
& $Py -m studio3d.sac_check
Pop-Location

Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue

Write-Host ''
if ($env:STUDIO3D_NONINTERACTIVE) { $ans = 'n' } else { $ans = Read-Host 'Download the AI models now (~15 GB)? Otherwise they download on first use. [Y/n]' }
if ($ans -eq '' -or $ans -match '^[Yy]') {
    Push-Location $App
    & $Py -m studio3d --download
    Pop-Location
}

Write-Host ''
Write-Host 'Done! Open "Studio3D" from your desktop.' -ForegroundColor Green
