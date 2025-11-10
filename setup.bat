@echo off
REM Setup script for LPGA Face Recognition System (Windows)

echo === LPGA Face Recognition Setup ===
echo.

REM Check Python version
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
echo.
echo Creating project directories...
if not exist "dataset" mkdir dataset
if not exist "output" mkdir output

echo.
echo === Setup Complete ===
echo.
echo To activate the environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the script:
echo   python cutlist_generator_v3.py --help
echo.
pause
