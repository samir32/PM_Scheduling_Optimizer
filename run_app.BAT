@echo off
title PM Optimizer
echo ============================================
echo  PM Schedule Optimizer — Starting up...
echo ============================================
echo.

REM Check if streamlit is installed
where streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Streamlit not found. Installing all dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] pip install failed. Make sure Python is installed and on PATH.
        pause
        exit /b 1
    )
)

echo [OK] Launching app in your browser...
echo      Press Ctrl+C here to stop the app.
echo.
streamlit run pm_optimizer.py

pause
