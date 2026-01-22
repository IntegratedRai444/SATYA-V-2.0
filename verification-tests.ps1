# SatyaAI Verification Tests
# Run these commands to verify all fixes are working

Write-Host "🚀 SATYAI VERIFICATION TESTS" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

# Test 1: Backend Health Check
Write-Host "1. Testing Backend Health..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001/health" -Method GET -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Backend Health: PASS" -ForegroundColor Green
    } else {
        Write-Host "❌ Backend Health: FAIL" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Backend Health: FAIL - $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: User Registration
Write-Host "2. Testing User Registration..." -ForegroundColor Yellow
try {
    $body = @{
        email = "test$(Get-Random -Maximum 9999)@example.com"
        password = "TestPass123!"
        user_metadata = @{
            role = "user"
        }
    } | ConvertTo-Json
    
    $response = Invoke-WebRequest -Uri "http://localhost:5001/api/v2/auth/signup" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10
    if ($response.StatusCode -eq 201) {
        Write-Host "✅ User Registration: PASS" -ForegroundColor Green
    } else {
        Write-Host "❌ User Registration: FAIL" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ User Registration: FAIL - $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: User Login
Write-Host "3. Testing User Login..." -ForegroundColor Yellow
try {
    $body = @{
        email = "test$(Get-Random -Maximum 9999)@example.com"
        password = "TestPass123!"
    } | ConvertTo-Json
    
    $response = Invoke-WebRequest -Uri "http://localhost:5001/api/v2/auth/login" -Method POST -Body $body -ContentType "application/json" -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ User Login: PASS" -ForegroundColor Green
    } else {
        Write-Host "❌ User Login: FAIL" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ User Login: FAIL - $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: Notifications Endpoint
Write-Host "4. Testing Notifications Endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001/api/v2/notifications" -Method GET -TimeoutSec 10
    if ($response.StatusCode -eq 401) {
        Write-Host "✅ Notifications Endpoint: PASS (auth required)" -ForegroundColor Green
    } else {
        Write-Host "❌ Notifications Endpoint: FAIL" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Notifications Endpoint: FAIL - $($_.Exception.Message)" -ForegroundColor Red
}

# Test 5: Chat Endpoint
Write-Host "5. Testing Chat Endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001/api/v2/chat/history" -Method GET -TimeoutSec 10
    if ($response.StatusCode -eq 401) {
        Write-Host "✅ Chat Endpoint: PASS (auth required)" -ForegroundColor Green
    } else {
        Write-Host "❌ Chat Endpoint: FAIL" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ Chat Endpoint: FAIL - $($_.Exception.Message)" -ForegroundColor Red
}

# Test 6: User Profile Endpoint
Write-Host "6. Testing User Profile Endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001/api/v2/user/profile" -Method GET -TimeoutSec 10
    if ($response.StatusCode -eq 401) {
        Write-Host "✅ User Profile Endpoint: PASS (auth required)" -ForegroundColor Green
    } else {
        Write-Host "❌ User Profile Endpoint: FAIL" -ForegroundColor Red
    }
} catch {
    Write-Host "❌ User Profile Endpoint: FAIL - $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "================================" -ForegroundColor Green
Write-Host "🏁 VERIFICATION TESTS COMPLETE" -ForegroundColor Green
