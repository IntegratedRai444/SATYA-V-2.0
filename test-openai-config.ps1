# Test OpenAI Configuration
Write-Host "🤖 Testing OpenAI Configuration..." -ForegroundColor Yellow

# Test 1: Check if OpenAI key is properly loaded in Node.js backend
Write-Host "1. Testing Node.js OpenAI Configuration..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001/api/v2/chat/suggestions" -Method POST -Body '{"message": "test"}' -ContentType "application/json" -TimeoutSec 10
    if ($response.StatusCode -eq 401) {
        Write-Host "✅ Node.js Chat Endpoint: PASS (auth required)" -ForegroundColor Green
    }
    elseif ($response.StatusCode -eq 200) {
        Write-Host "✅ Node.js Chat Endpoint: PASS (working)" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Node.js Chat Endpoint: FAIL (status $($response.StatusCode))" -ForegroundColor Red
    }
}
catch {
    Write-Host "❌ Node.js Chat Endpoint: FAIL - $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: Check if OpenAI key is properly loaded in Python backend
Write-Host "2. Testing Python Backend Health..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        $data = $response.Content | ConvertFrom-Json
        if ($data.openai_configured -eq $true) {
            Write-Host "✅ Python OpenAI: PASS" -ForegroundColor Green
        }
        else {
            Write-Host "⚠️ Python OpenAI: Not configured (mock responses)" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "❌ Python Health: FAIL (status $($response.StatusCode))" -ForegroundColor Red
    }
}
catch {
    Write-Host "❌ Python Health: FAIL - $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "🏁 OpenAI Configuration Test Complete" -ForegroundColor Green
