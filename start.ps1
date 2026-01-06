# TKLocalAI Launcher
# Just run: .\start.ps1 or .\start.ps1 desktop or .\start.ps1 server

param(
    [Parameter(Position=0)]
    [string]$Command = "desktop"
)

$ErrorActionPreference = "Stop"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Activate virtual environment
$VenvPath = Join-Path $ScriptDir "venv\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    . $VenvPath
} else {
    Write-Host "Error: Virtual environment not found. Run setup first." -ForegroundColor Red
    exit 1
}

# Add CUDA libraries to PATH
$NvidiaPath = Join-Path $ScriptDir "venv\Lib\site-packages\nvidia"
if (Test-Path "$NvidiaPath\cublas\bin") { $env:PATH = "$NvidiaPath\cublas\bin;$env:PATH" }
if (Test-Path "$NvidiaPath\cuda_runtime\bin") { $env:PATH = "$NvidiaPath\cuda_runtime\bin;$env:PATH" }
if (Test-Path "$NvidiaPath\cudnn\bin") { $env:PATH = "$NvidiaPath\cudnn\bin;$env:PATH" }

# Run command
switch ($Command.ToLower()) {
    "desktop" {
        Write-Host "`n⚡ Starting TKLocalAI Desktop...`n" -ForegroundColor Cyan
        python main.py desktop
    }
    "server" {
        Write-Host "`n⚡ Starting TKLocalAI Server on http://127.0.0.1:8000`n" -ForegroundColor Cyan
        python main.py server
    }
    "finetune" {
        Write-Host "`n⚡ Starting QLoRA Fine-tuning...`n" -ForegroundColor Cyan
        python main.py finetune
    }
    default {
        Write-Host "`nTKLocalAI - Local AI Assistant`n" -ForegroundColor Cyan
        Write-Host "Usage: .\start.ps1 [command]`n"
        Write-Host "Commands:"
        Write-Host "  desktop   - Run desktop app (default)"
        Write-Host "  server    - Run web server"
        Write-Host "  finetune  - Run fine-tuning"
        Write-Host ""
    }
}
