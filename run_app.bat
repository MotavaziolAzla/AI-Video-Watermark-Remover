@echo off
cd /d "%~dp0"
title AI Video Watermark Remover
echo ====================================================
echo   Starting AI Video Watermark Remover
echo ====================================================
echo.
echo Initializing Python environment...
call .venv\Scripts\activate.bat

echo Launching Inpainter Server...
echo (Your default web browser will automatically open once the server is ready!)
python -u app.py
pause
