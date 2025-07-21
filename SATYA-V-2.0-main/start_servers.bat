@echo off
echo Starting SatyaAI Deepfake Detection System...
echo.

echo Starting Backend Server (Python)...
start "Backend Server" cmd /k "cd /d %~dp0 && python ai_working.py"

echo Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo Starting Frontend Server (React)...
start "Frontend Server" cmd /k "cd /d %~dp0\client && npm run dev"

echo.
echo Both servers are starting...
echo Backend will be available at the URL shown in the backend terminal
echo Frontend will be available at http://localhost:5173
echo.
echo Press any key to exit this launcher...
pause > nul 