$ErrorActionPreference = "Stop"

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $workspace ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Create .venv before packaging the app."
}

& $python -m pip install -e "${workspace}[package]"
& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name "RekordBot" `
    --collect-all pystray `
    --copy-metadata fastmcp `
    --collect-submodules mido.backends `
    --hidden-import rtmidi `
    --collect-submodules rekordbox_performer `
    (Join-Path $workspace "rekordbot_app.py")

Write-Host "Built: $workspace\dist\RekordBot\RekordBot.exe"
