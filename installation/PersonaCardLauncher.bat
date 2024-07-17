@echo off
setlocal enabledelayedexpansion

:: Check if Python is installed and is a compatible version
for /f "tokens=2 delims=." %%a in ('python -c "import sys; print(sys.version)"') do set python_minor=%%a
if %python_minor% lss 9 (
    echo Python version 3.9 or higher is required. Please run the installer again.
    pause
    exit /b 1
)

:: Check if Streamlit is installed
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo Streamlit is not installed. Attempting to install...
    pip install streamlit
    if %errorlevel% neq 0 (
        echo Failed to install Streamlit. Please run the installer again.
        pause
        exit /b 1
    )
)

:: Get the directory of the batch file
set "SCRIPT_DIR=%~dp0"

:: Launch the Streamlit app
start "" python -m streamlit run "%SCRIPT_DIR%PersonaCard.py"

exit /b 0