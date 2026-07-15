@echo off
:: ============================================================
::  NutriWise AI - Quick Launcher
::  Uses the local virtual environment automatically
:: ============================================================

title NutriWise AI

echo ============================================================
echo   NutriWise AI - Personalized Nutrition Coach
echo   Powered by IBM watsonx.ai Granite Models
echo ============================================================
echo.

:: Check if venv exists
if not exist "venv\Scripts\python.exe" (
    echo [SETUP] Virtual environment not found. Creating it now...
    python -m venv venv
    echo [SETUP] Installing dependencies...
    venv\Scripts\pip install -r requirements.txt --quiet
    echo [SETUP] Done.
    echo.
)

:: Check if .env has been filled in
findstr /C:"your_ibm_cloud_api_key_here" .env >nul 2>&1
if %errorlevel%==0 (
    echo [NOTE] .env file found but credentials are still placeholders.
    echo        Edit .env and replace the placeholder values with your
    echo        actual IBM watsonx.ai API Key and Project ID.
    echo        The app will still start but AI calls will not work.
    echo.
)

echo [INFO] Starting Flask server at http://127.0.0.1:5000
echo [INFO] Press Ctrl+C to stop.
echo.

:: Run the app with UTF-8 output so emoji in logs display correctly
set PYTHONIOENCODING=utf-8
venv\Scripts\python.exe app.py

pause
