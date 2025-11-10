# SatyaAI Server Startup Script
# This script starts both the Python and Node.js servers

Write-Host "🚀 Starting SatyaAI Servers..." -ForegroundColor Green
Write-Host ""

# Function to check if a port is in use
function Test-Port {
    param([int]$Port)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

# Check if Python server is already running
if (Test-Port 5001) {
    Write-Host "✅ Python server is already running on port 5001" -ForegroundColor Green
} else {
    Write-Host "🐍 Starting Python server on port 5001..." -ForegroundColor Yellow
    Start-Process -FilePath "python" -ArgumentList "server/python/app.py" -WindowStyle Minimized
    Start-Sleep -Seconds 3
    
    if (Test-Port 5001) {
        Write-Host "✅ Python server started successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Failed to start Python server" -ForegroundColor Red
        Write-Host "Please check if Python is installed and requirements are met" -ForegroundColor Yellow
    }
}

# Check if Node.js server is already running
if (Test-Port 3000) {
    Write-Host "⚠️  Port 3000 is already in use. Stopping existing process..." -ForegroundColor Yellow
    # Try to find and stop the process using port 3000
    $processes = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess
    foreach ($processId in $processes) {
        try {
            Stop-Process -Id $processId -Force
            Write-Host "✅ Stopped process $processId" -ForegroundColor Green
        } catch {
            Write-Host "❌ Could not stop process $processId" -ForegroundColor Red
        }
    }
    Start-Sleep -Seconds 2
}

Write-Host "🟢 Starting Node.js development server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "📋 Server Information:" -ForegroundColor Cyan
Write-Host "  • Node.js Server: http://localhost:3000" -ForegroundColor White
Write-Host "  • Python AI Server: http://localhost:5001" -ForegroundColor White
Write-Host "  • Health Check: http://localhost:3000/health" -ForegroundColor White
Write-Host "  • API Documentation: http://localhost:3000/api" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Development Tools:" -ForegroundColor Cyan
Write-Host "  • Press Ctrl+C to stop the server" -ForegroundColor White
Write-Host "  • Check logs for any errors" -ForegroundColor White
Write-Host "  • Visit health endpoint to verify all services" -ForegroundColor White
Write-Host ""

# Start the Node.js development server
npm run dev