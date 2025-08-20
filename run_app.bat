@echo off
setlocal ENABLEDELAYEDEXPANSION

rem Change to this script's directory
cd /d "%~dp0"

rem Choose Python launcher
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  set "PYTHON=py -3"
) else (
  set "PYTHON=python"
)

rem Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PYTHON% -m venv .venv
  if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment. Ensure Python 3 is installed and on PATH.
    pause
    exit /b 1
  )
)

rem Activate venv
call ".venv\Scripts\activate"

rem Upgrade pip and install dependencies
python -m pip install --upgrade pip
if exist requirements.txt (
  pip install -r requirements.txt
) else (
  echo requirements.txt not found. Skipping dependency installation.
)

echo.
echo Launching Streamlit app...
echo You can stop the app with Ctrl+C in this window.
echo.

streamlit run app.py

endlocal
