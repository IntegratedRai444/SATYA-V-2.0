@echo off
echo Starting Python Flask Server for SatyaAI...
echo.

cd server\python

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Installing/Updating dependencies...
pip install -r requirements-complete.txt

echo.
echo Starting Flask server on port 5001...
python app.py

pause
