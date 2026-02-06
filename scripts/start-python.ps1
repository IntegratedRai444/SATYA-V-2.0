# Start Python ML Service with Virtual Environment
Write-Host "🐍 Starting Python ML Service..." -ForegroundColor Green

# Activate Python 3.11 virtual environment
$venvPath = ".\venv_py311\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "✅ Activating Python 3.11 virtual environment..." -ForegroundColor Cyan
    & $venvPath
}
else {
    Write-Host "❌ Virtual environment not found at $venvPath" -ForegroundColor Red
    exit 1
}

# Change to Python directory
Set-Location "server\python"

# Check if main_api.py exists
if (-not (Test-Path "main_api.py")) {
    Write-Host "❌ main_api.py not found" -ForegroundColor Red
    exit 1
}

# Start the Python service
Write-Host "🚀 Starting FastAPI server..." -ForegroundColor Green
python main_api.py
