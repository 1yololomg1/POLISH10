@echo off
REM Launcher script for Advanced Wireline Data Preprocessing System
REM Windows Batch File

echo Advanced Wireline Data Preprocessing System
echo ============================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    echo.
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import numpy, pandas, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Some dependencies may be missing
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo Please run: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo.
echo Starting application...
echo.

REM Run the application
python advanced_preprocessing_system10.py

REM If application exits, pause to see any error messages
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)

