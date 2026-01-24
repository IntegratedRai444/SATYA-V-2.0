@echo off
echo ========================================
echo Starting SatyaAI Services
echo ========================================

echo.
echo [1/3] Starting Node.js Backend on port 5001...
cd /d "%~dp0server"
start "Node Backend" cmd /k "npm run dev"

timeout /t 3 /nobreak > nul

echo.
echo [2/3] Starting Python ML Service on port 8000...
cd /d "%~dp0server\python"
start "Python ML Service" cmd /k "uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak > nul

echo.
echo [3/3] Starting Frontend on port 5173...
cd /d "%~dp0client"
start "Frontend" cmd /k "npm run dev"

echo.
echo ========================================
echo All services starting...
echo ========================================
echo.
echo Services will be available at:
echo - Frontend: http://localhost:5173
echo - Node Backend: http://localhost:5001
echo - Python ML: http://localhost:8000
echo.
echo Press any key to exit...
pause > nul
