@echo off
cd /d "%~dp0"
title GPU Video Watermark Inpainter
echo ====================================================
echo   Starting High-Performance GPU Watermark Inpainter
echo ====================================================
echo.
echo Initializing Python environment...
call .venv\Scripts\activate.bat

echo Launching Inpainter Server...
echo (Your default web browser will automatically open once the server is ready!)
python -u app.py
pause
