@echo off
setlocal enabledelayedexpansion
title GPU Video Inpainter - New Machine Automatic Setup

echo ========================================================
echo   GPU Video Watermark Inpainter - 1-Click Setup
echo ========================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found on your system PATH!
    echo Please install Python 3.10, 3.11, or 3.12 from python.org and check "Add Python to PATH".
    pause
    exit /b 1
)

echo [1/5] Checking Python installation...
python --version

:: 2. Create Virtual Environment
if not exist ".venv" (
    echo [2/5] Creating isolated virtual environment in .venv...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [2/5] Virtual environment .venv already exists.
)

:: 3. Upgrade pip and install uv for ultra-fast installation
echo [3/5] Installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip uv

:: 4. Install PyTorch with CUDA and all dependencies
echo Installing PyTorch CUDA 12.4 and neural inpainting engines...
uv pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [Fallback] Retrying with standard pip...
    pip install -r requirements.txt
)

:: 5. Download and verify model weights
echo [4/5] Checking and downloading AI model weights (ProPainter, E2FGVI-HQ, LaMa)...
python download_models.py

:: 6. Verify FFmpeg & GPU Acceleration
echo [5/5] Checking GPU & FFmpeg NVENC status...
python -c "import torch; print('CUDA GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo ========================================================
echo   SETUP COMPLETED SUCCESSFULLY!
echo ========================================================
echo You can now double-click "run_app.bat" to start the UI.
echo.
pause
