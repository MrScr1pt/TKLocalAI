# TKLocalAI Setup Script for Windows
# ===================================
# This script sets up everything needed to run TKLocalAI

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TKLocalAI Setup" -ForegroundColor Cyan
Write-Host "   Local AI Assistant with LoRA and RAG" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ===========================================
# Step 1: Check for Python 3.11
# ===========================================

Write-Host "[1/7] Checking for Python 3.11..." -ForegroundColor Yellow

$pythonCmd = $null
$pythonPaths = @(
    "py -3.11",
    "C:\Python311\python.exe",
    "C:\Program Files\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)

# Try py launcher first
try {
    $version = & py -3.11 --version 2>&1
    if ($version -match "Python 3\.11") {
        $pythonCmd = "py -3.11"
        Write-Host "  Found Python 3.11 via py launcher" -ForegroundColor Green
    }
} catch {}

# Check common paths if py launcher failed
if (-not $pythonCmd) {
    foreach ($path in $pythonPaths[1..3]) {
        if (Test-Path $path) {
            $pythonCmd = $path
            Write-Host "  Found Python 3.11 at: $path" -ForegroundColor Green
            break
        }
    }
}

if (-not $pythonCmd) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Red
    Write-Host "  ERROR: Python 3.11 not found!" -ForegroundColor Red
    Write-Host "================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Python 3.11 is required for ML packages compatibility." -ForegroundColor White
    Write-Host ""
    Write-Host "Please download and install Python 3.11.9:" -ForegroundColor Yellow
    Write-Host "  https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "During installation, make sure to check:" -ForegroundColor Yellow
    Write-Host "  [x] Add Python to PATH" -ForegroundColor White
    Write-Host "  [x] Install for all users (recommended)" -ForegroundColor White
    Write-Host ""
    Write-Host "After installing, run this script again." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Show Python version
if ($pythonCmd -eq "py -3.11") {
    & py -3.11 --version
} else {
    & $pythonCmd --version
}

# ===========================================
# Step 2: Create Virtual Environment
# ===========================================

Write-Host ""
Write-Host "[2/7] Creating virtual environment..." -ForegroundColor Yellow

# Remove old venv if exists
if (Test-Path "venv") {
    Write-Host "  Removing old virtual environment..." -ForegroundColor Gray
    Remove-Item -Recurse -Force "venv"
    Start-Sleep -Seconds 2
}

# Create new venv
if ($pythonCmd -eq "py -3.11") {
    & py -3.11 -m venv venv
} else {
    & $pythonCmd -m venv venv
}

if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "  ERROR: Failed to create virtual environment!" -ForegroundColor Red
    exit 1
}

Write-Host "  Virtual environment created" -ForegroundColor Green

# Activate venv
& .\venv\Scripts\Activate.ps1

# ===========================================
# Step 3: Upgrade pip and install build tools
# ===========================================

Write-Host ""
Write-Host "[3/7] Upgrading pip and build tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip wheel setuptools --quiet
Write-Host "  Done" -ForegroundColor Green

# ===========================================
# Step 4: Install PyTorch with CUDA
# ===========================================

Write-Host ""
Write-Host "[4/7] Installing PyTorch with CUDA support..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray

$torchInstalled = $false

# Try CUDA 11.8 first (broadest compatibility)
try {
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
    $torchInstalled = $true
    Write-Host "  PyTorch with CUDA 11.8 installed" -ForegroundColor Green
} catch {
    Write-Host "  CUDA 11.8 failed, trying CUDA 12.1..." -ForegroundColor Yellow
}

# Try CUDA 12.1
if (-not $torchInstalled) {
    try {
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
        $torchInstalled = $true
        Write-Host "  PyTorch with CUDA 12.1 installed" -ForegroundColor Green
    } catch {
        Write-Host "  CUDA 12.1 failed, installing CPU version..." -ForegroundColor Yellow
    }
}

# Fall back to CPU
if (-not $torchInstalled) {
    pip install torch torchvision torchaudio --quiet
    Write-Host "  PyTorch CPU version installed (no GPU acceleration)" -ForegroundColor Yellow
}

# ===========================================
# Step 5: Install llama-cpp-python with CUDA
# ===========================================

Write-Host ""
Write-Host "[5/7] Installing llama-cpp-python with CUDA..." -ForegroundColor Yellow
Write-Host "  Trying pre-built CUDA wheels..." -ForegroundColor Gray

$llamaInstalled = $false
$cudaVersions = @("cu118", "cu121", "cu122", "cu123", "cu124")

foreach ($cuda in $cudaVersions) {
    if (-not $llamaInstalled) {
        try {
            $url = "https://abetlen.github.io/llama-cpp-python/whl/$cuda"
            pip install llama-cpp-python --prefer-binary --extra-index-url $url --quiet 2>$null
            $llamaInstalled = $true
            Write-Host "  llama-cpp-python with $cuda installed" -ForegroundColor Green
        } catch {
            # Continue to next version
        }
    }
}

# Fall back to CPU
if (-not $llamaInstalled) {
    Write-Host "  No CUDA wheels found, installing CPU version..." -ForegroundColor Yellow
    pip install llama-cpp-python --quiet
    Write-Host "  llama-cpp-python CPU version installed" -ForegroundColor Yellow
}

# ===========================================
# Step 6: Install all dependencies
# ===========================================

Write-Host ""
Write-Host "[6/7] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor Gray

# Install core dependencies that often cause issues first
$coreDeps = @(
    "numpy",
    "scipy",
    "scikit-learn",
    "huggingface-hub",
    "tokenizers",
    "safetensors"
)

foreach ($dep in $coreDeps) {
    Write-Host "  Installing $dep..." -ForegroundColor Gray
    pip install $dep --quiet 2>$null
}

# Install sentence-transformers (depends on scipy)
Write-Host "  Installing sentence-transformers..." -ForegroundColor Gray
pip install sentence-transformers --quiet

# Install the rest from requirements.txt
Write-Host "  Installing remaining packages..." -ForegroundColor Gray
pip install -r requirements.txt --quiet --ignore-installed llama-cpp-python torch torchvision torchaudio 2>$null

Write-Host "  All dependencies installed" -ForegroundColor Green

# ===========================================
# Step 7: Create directories and verify
# ===========================================

Write-Host ""
Write-Host "[7/7] Setting up project structure..." -ForegroundColor Yellow

$dirs = @("models", "adapters", "data\documents", "data\vectordb", "data\training", "logs", "cache")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "  Directories created" -ForegroundColor Green

# ===========================================
# Verify installations
# ===========================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Verifying Installation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check PyTorch
Write-Host "PyTorch: " -NoNewline
try {
    $pytorchInfo = python -c "import torch; print(f'{torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>&1
    if ($pytorchInfo -match "True") {
        Write-Host $pytorchInfo -ForegroundColor Green
    } else {
        Write-Host $pytorchInfo -ForegroundColor Yellow
    }
} catch {
    Write-Host "NOT INSTALLED" -ForegroundColor Red
}

# Check llama-cpp-python
Write-Host "llama-cpp-python: " -NoNewline
try {
    python -c "import llama_cpp; print('OK')" 2>$null
    Write-Host "OK" -ForegroundColor Green
} catch {
    Write-Host "NOT INSTALLED" -ForegroundColor Red
}

# Check sentence-transformers
Write-Host "sentence-transformers: " -NoNewline
try {
    python -c "import sentence_transformers; print('OK')" 2>$null
    Write-Host "OK" -ForegroundColor Green
} catch {
    Write-Host "NOT INSTALLED" -ForegroundColor Red
}

# Check FastAPI
Write-Host "FastAPI: " -NoNewline
try {
    python -c "import fastapi; print('OK')" 2>$null
    Write-Host "OK" -ForegroundColor Green
} catch {
    Write-Host "NOT INSTALLED" -ForegroundColor Red
}

# Check pywebview
Write-Host "pywebview: " -NoNewline
try {
    python -c "import webview; print('OK')" 2>$null
    Write-Host "OK" -ForegroundColor Green
} catch {
    Write-Host "NOT INSTALLED" -ForegroundColor Red
}

# ===========================================
# Done!
# ===========================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Download a model (if you haven't already):" -ForegroundColor White
Write-Host "   Visit: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" -ForegroundColor Cyan
Write-Host "   Download: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (~4.9GB)" -ForegroundColor Cyan
Write-Host "   Save to: models\" -ForegroundColor Cyan
Write-Host ""
Write-Host "2. Run the app:" -ForegroundColor White
Write-Host "   .\start.ps1 desktop   # Desktop app with UI" -ForegroundColor Cyan
Write-Host "   .\start.ps1 server    # API server only" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. (Optional) For fine-tuning, install training dependencies:" -ForegroundColor White
Write-Host "   pip install peft bitsandbytes trl" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"
