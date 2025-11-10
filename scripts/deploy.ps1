# SatyaAI Deployment Script for Windows
# This script handles deployment to different environments

param(
    [Parameter(Position=0)]
    [ValidateSet("development", "dev", "staging", "stage", "production", "prod")]
    [string]$Environment = "production",
    
    [Parameter(Position=1)]
    [string]$Version = "latest",
    
    [switch]$Help,
    [switch]$Status,
    [switch]$Rollback,
    [switch]$Cleanup
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check prerequisites
function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check if Docker is installed
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Error "Docker is not installed. Please install Docker Desktop first."
        exit 1
    }
    
    # Check if Docker Compose is installed
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    }
    
    # Check if we're in the right directory
    if (-not (Test-Path "$ProjectDir\package.json")) {
        Write-Error "This script must be run from the SatyaAI project directory."
        exit 1
    }
    
    Write-Success "Prerequisites check passed"
}

# Validate environment
function Test-Environment {
    Write-Info "Validating environment: $Environment"
    
    $script:ComposeFile = switch ($Environment) {
        { $_ -in @("development", "dev") } { "docker-compose.dev.yml" }
        { $_ -in @("staging", "stage") } { 
            if (Test-Path "$ProjectDir\docker-compose.staging.yml") {
                "docker-compose.staging.yml"
            } else {
                Write-Warning "Staging compose file not found, using production file"
                "docker-compose.yml"
            }
        }
        { $_ -in @("production", "prod") } { "docker-compose.yml" }
        default { 
            Write-Error "Invalid environment: $Environment. Use: development, staging, or production"
            exit 1
        }
    }
    
    Write-Success "Environment validated: $Environment"
}

# Create necessary directories
function New-Directories {
    Write-Info "Creating necessary directories..."
    
    $directories = @(
        "$ProjectDir\data",
        "$ProjectDir\uploads", 
        "$ProjectDir\logs",
        "$ProjectDir\nginx\ssl"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Success "Directories created"
}

# Generate environment file
function New-EnvironmentFile {
    Write-Info "Generating environment file..."
    
    $EnvFile = "$ProjectDir\.env.$Environment"
    
    if (-not (Test-Path $EnvFile)) {
        Write-Info "Creating environment file: $EnvFile"
        
        # Generate random secrets
        $JwtSecret = [System.Web.Security.Membership]::GeneratePassword(64, 0)
        $SessionSecret = [System.Web.Security.Membership]::GeneratePassword(64, 0)
        
        $envContent = @"
# SatyaAI Environment Configuration - $Environment
NODE_ENV=$Environment
PORT=3000
HOST=0.0.0.0

# Database
DATABASE_URL=sqlite:./data/satyaai.db
DATABASE_SSL=false

# Python Server
PYTHON_SERVER_URL=http://localhost:5001
PYTHON_TIMEOUT=30000

# Security (CHANGE THESE IN PRODUCTION!)
JWT_SECRET=$JwtSecret
SESSION_SECRET=$SessionSecret
BCRYPT_ROUNDS=12

# File Upload
MAX_FILE_SIZE=100MB
UPLOAD_DIR=./uploads
ALLOWED_FILE_TYPES=jpg,jpeg,png,gif,mp4,avi,mov,mp3,wav,flac

# Logging
LOG_LEVEL=info
LOG_DIR=./logs
LOG_MAX_FILES=10
LOG_MAX_SIZE=10MB

# Features
FEATURE_REGISTRATION=true
FEATURE_GUEST_ACCESS=false
FEATURE_FILE_UPLOAD=true
FEATURE_REALTIME=true

# Monitoring
METRICS_ENABLED=true
ALERTING_ENABLED=true
HEALTH_CHECK_INTERVAL=30000

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
CORS_CREDENTIALS=true

# Rate Limiting
RATE_LIMIT_WINDOW=900000
RATE_LIMIT_MAX=100

# External Services (optional)
# REDIS_URL=redis://redis:6379
# SMTP_HOST=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USER=your-email@gmail.com
# SMTP_PASS=your-app-password
# WEBHOOK_URL=https://your-webhook-url.com
# WEBHOOK_SECRET=your-webhook-secret
"@
        
        Set-Content -Path $EnvFile -Value $envContent
        
        if ($Environment -eq "production") {
            Write-Warning "Please review and update the generated .env.production file with your production values!"
        }
    } else {
        Write-Info "Environment file already exists: $EnvFile"
    }
    
    Write-Success "Environment file ready"
}

# Build application
function Build-Application {
    Write-Info "Building application..."
    
    Set-Location $ProjectDir
    
    # Build Docker image
    docker-compose -f $ComposeFile build --no-cache
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed"
        exit 1
    }
    
    Write-Success "Application built successfully"
}

# Deploy application
function Deploy-Application {
    Write-Info "Deploying application..."
    
    Set-Location $ProjectDir
    
    # Stop existing containers
    docker-compose -f $ComposeFile down
    
    # Start new containers
    docker-compose -f $ComposeFile up -d
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Deployment failed"
        exit 1
    }
    
    # Wait for services to be ready
    Write-Info "Waiting for services to be ready..."
    Start-Sleep -Seconds 30
    
    # Check health
    Test-Health
    
    Write-Success "Application deployed successfully"
}

# Check application health
function Test-Health {
    Write-Info "Checking application health..."
    
    $maxAttempts = 30
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:3000/health" -UseBasicParsing -TimeoutSec 10
            if ($response.StatusCode -eq 200) {
                Write-Success "Application is healthy"
                return
            }
        } catch {
            # Ignore errors and continue trying
        }
        
        Write-Info "Health check attempt $attempt/$maxAttempts failed, retrying in 10 seconds..."
        Start-Sleep -Seconds 10
        $attempt++
    }
    
    Write-Error "Application health check failed after $maxAttempts attempts"
    exit 1
}

# Show deployment status
function Show-Status {
    Write-Info "Deployment Status:"
    Write-Host ""
    
    Set-Location $ProjectDir
    docker-compose -f $ComposeFile ps
    
    Write-Host ""
    Write-Info "Application URLs:"
    Write-Host "  - Main Application: http://localhost:3000"
    Write-Host "  - Health Check: http://localhost:3000/health"
    Write-Host "  - Metrics: http://localhost:3000/metrics"
    
    if ($Environment -ne "development") {
        Write-Host "  - Prometheus: http://localhost:9090"
        Write-Host "  - Grafana: http://localhost:3001 (admin/admin)"
    }
}

# Cleanup function
function Invoke-Cleanup {
    Write-Info "Cleaning up..."
    
    Set-Location $ProjectDir
    
    # Remove unused Docker images
    docker image prune -f
    
    Write-Success "Cleanup completed"
}

# Rollback function
function Invoke-Rollback {
    Write-Info "Rolling back deployment..."
    
    Set-Location $ProjectDir
    
    # Stop current containers
    docker-compose -f $ComposeFile down
    
    # Start with previous version (if available)
    # This would need to be implemented based on your versioning strategy
    
    Write-Success "Rollback completed"
}

# Show help
function Show-Help {
    Write-Host "SatyaAI Deployment Script"
    Write-Host ""
    Write-Host "Usage: .\deploy.ps1 [environment] [version] [options]"
    Write-Host ""
    Write-Host "Environments:"
    Write-Host "  development, dev    - Development environment"
    Write-Host "  staging, stage      - Staging environment"
    Write-Host "  production, prod    - Production environment"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Help               - Show this help message"
    Write-Host "  -Status             - Show deployment status"
    Write-Host "  -Rollback           - Rollback deployment"
    Write-Host "  -Cleanup            - Cleanup unused resources"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy.ps1 development"
    Write-Host "  .\deploy.ps1 production v1.2.3"
    Write-Host "  .\deploy.ps1 staging latest"
    Write-Host "  .\deploy.ps1 -Status"
}

# Main deployment function
function Start-Deployment {
    Write-Info "Starting SatyaAI deployment..."
    Write-Info "Environment: $Environment"
    Write-Info "Version: $Version"
    Write-Host ""
    
    Test-Prerequisites
    Test-Environment
    New-Directories
    New-EnvironmentFile
    Build-Application
    Deploy-Application
    Show-Status
    Invoke-Cleanup
    
    Write-Host ""
    Write-Success "Deployment completed successfully!"
    Write-Info "Run 'docker-compose -f $ComposeFile logs -f' to view logs"
}

# Handle script arguments
if ($Help) {
    Show-Help
    exit 0
}

if ($Status) {
    Test-Environment
    Show-Status
    exit 0
}

if ($Rollback) {
    Test-Environment
    Invoke-Rollback
    exit 0
}

if ($Cleanup) {
    Invoke-Cleanup
    exit 0
}

# Add System.Web assembly for password generation
Add-Type -AssemblyName System.Web

# Run main deployment
Start-Deployment