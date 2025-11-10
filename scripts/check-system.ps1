# SatyaAI System Status Check Script

Write-Host "🔍 SatyaAI System Status Check" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""

# Function to check if a port is in use
function Test-Port {
    param([int]$Port, [string]$ServiceName)
    try {
        $connection = New-Object System.Net.Sockets.TcpClient
        $connection.Connect("localhost", $Port)
        $connection.Close()
        Write-Host "✅ $ServiceName (Port $Port): RUNNING" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ $ServiceName (Port $Port): NOT RUNNING" -ForegroundColor Red
        return $false
    }
}

# Function to test HTTP endpoint
function Test-HttpEndpoint {
    param([string]$Url, [string]$ServiceName)
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ $ServiceName Health Check: HEALTHY" -ForegroundColor Green
            return $true
        } else {
            Write-Host "⚠️  $ServiceName Health Check: DEGRADED (Status: $($response.StatusCode))" -ForegroundColor Yellow
            return $false
        }
    } catch {
        Write-Host "❌ $ServiceName Health Check: FAILED" -ForegroundColor Red
        return $false
    }
}

# Check system requirements
Write-Host "📋 System Requirements:" -ForegroundColor Cyan
Write-Host "----------------------" -ForegroundColor Cyan

# Check Node.js
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js: NOT INSTALLED" -ForegroundColor Red
}

# Check Python
try {
    $pythonVersion = python --version
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python: NOT INSTALLED" -ForegroundColor Red
}

# Check npm
try {
    $npmVersion = npm --version
    Write-Host "✅ npm: v$npmVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ npm: NOT INSTALLED" -ForegroundColor Red
}

Write-Host ""

# Check services
Write-Host "🔧 Service Status:" -ForegroundColor Cyan
Write-Host "------------------" -ForegroundColor Cyan

$nodeRunning = Test-Port -Port 3000 -ServiceName "Node.js Server"
$pythonRunning = Test-Port -Port 5001 -ServiceName "Python AI Server"

Write-Host ""

# Check health endpoints if services are running
Write-Host "🏥 Health Checks:" -ForegroundColor Cyan
Write-Host "-----------------" -ForegroundColor Cyan

if ($nodeRunning) {
    Test-HttpEndpoint -Url "http://localhost:3000/health" -ServiceName "Node.js Server"
} else {
    Write-Host "⏭️  Node.js Server: SKIPPED (Service not running)" -ForegroundColor Gray
}

if ($pythonRunning) {
    Test-HttpEndpoint -Url "http://localhost:5001/health" -ServiceName "Python AI Server"
} else {
    Write-Host "⏭️  Python AI Server: SKIPPED (Service not running)" -ForegroundColor Gray
}

Write-Host ""

# Check file system
Write-Host "📁 File System:" -ForegroundColor Cyan
Write-Host "---------------" -ForegroundColor Cyan

$directories = @(
    @{Path="./data"; Name="Data Directory"},
    @{Path="./uploads"; Name="Uploads Directory"},
    @{Path="./logs"; Name="Logs Directory"},
    @{Path="./server/python"; Name="Python Server Directory"}
)

foreach ($dir in $directories) {
    if (Test-Path $dir.Path) {
        Write-Host "✅ $($dir.Name): EXISTS" -ForegroundColor Green
    } else {
        Write-Host "❌ $($dir.Name): MISSING" -ForegroundColor Red
    }
}

Write-Host ""

# Check important files
Write-Host "📄 Configuration Files:" -ForegroundColor Cyan
Write-Host "-----------------------" -ForegroundColor Cyan

$files = @(
    @{Path="./package.json"; Name="Package Configuration"},
    @{Path="./server/python/requirements.txt"; Name="Python Requirements"},
    @{Path="./server/python/app.py"; Name="Python Server"},
    @{Path="./server/index.ts"; Name="Node.js Server"},
    @{Path="./.env"; Name="Environment Configuration"}
)

foreach ($file in $files) {
    if (Test-Path $file.Path) {
        Write-Host "✅ $($file.Name): EXISTS" -ForegroundColor Green
    } else {
        Write-Host "❌ $($file.Name): MISSING" -ForegroundColor Red
    }
}

Write-Host ""

# Summary
Write-Host "📊 System Summary:" -ForegroundColor Cyan
Write-Host "------------------" -ForegroundColor Cyan

if ($nodeRunning -and $pythonRunning) {
    Write-Host "🎉 System Status: FULLY OPERATIONAL" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 Access your application at:" -ForegroundColor Yellow
    Write-Host "   • Main App: http://localhost:3000" -ForegroundColor White
    Write-Host "   • Health Check: http://localhost:3000/health" -ForegroundColor White
    Write-Host "   • API Docs: http://localhost:3000/api" -ForegroundColor White
} elseif ($nodeRunning -or $pythonRunning) {
    Write-Host "⚠️  System Status: PARTIALLY OPERATIONAL" -ForegroundColor Yellow
    Write-Host "   Some services are not running. Check the status above." -ForegroundColor White
} else {
    Write-Host "❌ System Status: NOT OPERATIONAL" -ForegroundColor Red
    Write-Host "   No services are running. Use start-servers.ps1 to start them." -ForegroundColor White
}

Write-Host ""
Write-Host "💡 Quick Actions:" -ForegroundColor Cyan
Write-Host "  • To start servers: .\start-servers.ps1" -ForegroundColor White
Write-Host "  • To check logs: npm run logs" -ForegroundColor White
Write-Host "  • To stop servers: Ctrl+C in the terminal running the servers" -ForegroundColor White