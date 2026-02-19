# SatyaAI Python Service Startup Script
# This script starts the Python ML backend service

Write-Host "Starting SatyaAI Python ML Service..." -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Yellow

# Set working directory to Python server
Set-Location $PSScriptRoot\..\server\python

Write-Host "Working Directory: $(Get-Location)" -ForegroundColor Cyan

# Check if virtual environment exists
if (Test-Path ".venv_311_new") {
    Write-Host "Virtual environment found" -ForegroundColor Green
    
    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ".venv_311_new\Scripts\Activate.ps1"
    
    # Start Python service
    Write-Host "Starting Python ML service..." -ForegroundColor Yellow
    Write-Host "Service will be available at: http://localhost:8000" -ForegroundColor Cyan
    Write-Host "API Docs: http://localhost:8000/api/docs" -ForegroundColor Cyan
    Write-Host "Health Check: http://localhost:8000/health" -ForegroundColor Cyan
    Write-Host ""
    
    python main_api.py
}
else {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run: python -m venv .venv_311_new" -ForegroundColor Yellow
    exit 1
}
