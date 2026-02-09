@echo off
echo Activating Python virtual environment...
call .venv_311_new\Scripts\Activate.bat

echo Starting SatyaAI Python Backend...
echo.
echo Server will start at: http://localhost:8000
echo API Docs: http://localhost:8000/api/docs
echo Health Check: http://localhost:8000/health
echo.

python main_api.py

pause
