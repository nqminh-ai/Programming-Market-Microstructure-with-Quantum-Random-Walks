@echo off
title QRW Market Microstructure Dashboard
echo ========================================================
echo   Starting QRW Market Microstructure Dashboard...
echo ========================================================

REM Check and activate virtual environment if it exists
IF EXIST "venv\Scripts\activate.bat" (
    echo [Info] Activating virtual environment - venv
    call venv\Scripts\activate.bat
) ELSE IF EXIST ".venv\Scripts\activate.bat" (
    echo [Info] Activating virtual environment - .venv
    call .venv\Scripts\activate.bat
) ELSE (
    echo [Info] No local virtual environment found, using global Python.
)

echo [Info] Launching Streamlit App...
python -m streamlit run src/dashboard/app.py

pause
