@echo off
REM One-click setup for Windows: creates .venv, installs the packages and runs the phase-2 demo.
REM Double-click this file in File Explorer, or type  setup_windows.bat  in a terminal.
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating the virtual environment .venv ...
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Package installation failed. Copy the red error text above and send it to Claude.
  pause
  exit /b 1
)
echo.
echo Packages installed. Running the phase-2 demo on synthetic data (about 5 minutes) ...
python tools\run_demo.py --phase2
echo.
echo Finished. Results are in the "results" folder.
pause
