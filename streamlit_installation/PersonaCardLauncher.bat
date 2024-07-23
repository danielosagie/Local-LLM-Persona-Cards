@echo off
setlocal enabledelayedexpansion

:: Set up logging
set "LOGFILE=%TEMP%\PersonaCardLauncher.log"
echo PersonaCard Launcher Log > %LOGFILE%
echo Timestamp: %DATE% %TIME% >> %LOGFILE%

:: Check Python version
echo Checking Python version... >> %LOGFILE%
for /f "tokens=2 delims=." %%a in ('python -c "import sys; print(sys.version)" 2^>^&1') do set python_minor=%%a
if %ERRORLEVEL% neq 0 (
    echo Python not found or error occurred. >> %LOGFILE%
    echo Python is not installed or not in PATH. Please run the installer again.
    echo Python is not installed or not in PATH. Please run the installer again. >> %LOGFILE%
    pause
    exit /b 1
)
echo Python minor version: %python_minor% >> %LOGFILE%

if %python_minor% lss 9 (
    echo Python version 3.9 or higher is required. >> %LOGFILE%
    echo Python version 3.9 or higher is required. Please run the installer again.
    pause
    exit /b 1
)

:: Check Streamlit
echo Checking Streamlit installation... >> %LOGFILE%
python -c "import streamlit" 2>>%LOGFILE%
if %ERRORLEVEL% neq 0 (
    echo Streamlit not found. Attempting to install... >> %LOGFILE%
    pip install streamlit 2>>%LOGFILE%
    if %ERRORLEVEL% neq 0 (
        echo Failed to install Streamlit. >> %LOGFILE%
        echo Failed to install Streamlit. Please run the installer again.
        pause
        exit /b 1
    )
)

:: Get the directory of the batch file
set "SCRIPT_DIR=%~dp0"
echo Script directory: %SCRIPT_DIR% >> %LOGFILE%

:: Check if PersonaCard.py exists
if not exist "%SCRIPT_DIR%PersonaCard.py" (
    echo PersonaCard.py not found in %SCRIPT_DIR% >> %LOGFILE%
    echo PersonaCard.py not found. Please ensure the file is in the correct location.
    pause
    exit /b 1
)

:: Launch the Streamlit app
echo Launching Streamlit app... >> %LOGFILE%
echo Command: python -m streamlit run "%SCRIPT_DIR%PersonaCard.py" >> %LOGFILE%
start "" python -m streamlit run "%SCRIPT_DIR%PersonaCard.py" 2>>%LOGFILE%

if %ERRORLEVEL% neq 0 (
    echo Failed to launch Streamlit app. Check the log file for details: %LOGFILE%
    pause
    exit /b 1
)

echo Launcher completed successfully. >> %LOGFILE%
exit /b 0