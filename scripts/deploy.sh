#!/bin/bash

# SatyaAI Deployment Script
# This script handles deployment to different environments

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-production}"
VERSION="${2:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "$PROJECT_DIR/package.json" ]; then
        log_error "This script must be run from the SatyaAI project directory."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Validate environment
validate_environment() {
    log_info "Validating environment: $ENVIRONMENT"
    
    case $ENVIRONMENT in
        development|dev)
            COMPOSE_FILE="docker-compose.dev.yml"
            ;;
        staging|stage)
            COMPOSE_FILE="docker-compose.staging.yml"
            if [ ! -f "$PROJECT_DIR/$COMPOSE_FILE" ]; then
                COMPOSE_FILE="docker-compose.yml"
                log_warning "Staging compose file not found, using production file"
            fi
            ;;
        production|prod)
            COMPOSE_FILE="docker-compose.yml"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT. Use: development, staging, or production"
            exit 1
            ;;
    esac
    
    log_success "Environment validated: $ENVIRONMENT"
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$PROJECT_DIR/data"
    mkdir -p "$PROJECT_DIR/uploads"
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/nginx/ssl"
    
    log_success "Directories created"
}

# Generate environment file
generate_env_file() {
    log_info "Generating environment file..."
    
    ENV_FILE="$PROJECT_DIR/.env.$ENVIRONMENT"
    
    if [ ! -f "$ENV_FILE" ]; then
        log_info "Creating environment file: $ENV_FILE"
        
        cat > "$ENV_FILE" << EOF
# SatyaAI Environment Configuration - $ENVIRONMENT
NODE_ENV=$ENVIRONMENT
PORT=3000
HOST=0.0.0.0

# Database
DATABASE_URL=sqlite:./data/satyaai.db
DATABASE_SSL=false

# Python Server
PYTHON_SERVER_URL=http://localhost:5001
PYTHON_TIMEOUT=30000

# Security (CHANGE THESE IN PRODUCTION!)
JWT_SECRET=$(openssl rand -hex 32)
SESSION_SECRET=$(openssl rand -hex 32)
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
EOF
        
        if [ "$ENVIRONMENT" = "production" ]; then
            log_warning "Please review and update the generated .env.production file with your production values!"
        fi
    else
        log_info "Environment file already exists: $ENV_FILE"
    fi
    
    log_success "Environment file ready"
}

# Build application
build_application() {
    log_info "Building application..."
    
    cd "$PROJECT_DIR"
    
    # Build Docker image
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    
    log_success "Application built successfully"
}

# Deploy application
deploy_application() {
    log_info "Deploying application..."
    
    cd "$PROJECT_DIR"
    
    # Stop existing containers
    docker-compose -f "$COMPOSE_FILE" down
    
    # Start new containers
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check health
    check_health
    
    log_success "Application deployed successfully"
}

# Check application health
check_health() {
    log_info "Checking application health..."
    
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:3000/health > /dev/null; then
            log_success "Application is healthy"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Application health check failed after $max_attempts attempts"
    return 1
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo
    
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo
    log_info "Application URLs:"
    echo "  - Main Application: http://localhost:3000"
    echo "  - Health Check: http://localhost:3000/health"
    echo "  - Metrics: http://localhost:3000/metrics"
    
    if [ "$ENVIRONMENT" != "development" ]; then
        echo "  - Prometheus: http://localhost:9090"
        echo "  - Grafana: http://localhost:3001 (admin/admin)"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    cd "$PROJECT_DIR"
    
    # Remove unused Docker images
    docker image prune -f
    
    log_success "Cleanup completed"
}

# Rollback function
rollback() {
    log_info "Rolling back deployment..."
    
    cd "$PROJECT_DIR"
    
    # Stop current containers
    docker-compose -f "$COMPOSE_FILE" down
    
    # Start with previous version (if available)
    # This would need to be implemented based on your versioning strategy
    
    log_success "Rollback completed"
}

# Main deployment function
main() {
    log_info "Starting SatyaAI deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    echo
    
    check_prerequisites
    validate_environment
    create_directories
    generate_env_file
    build_application
    deploy_application
    show_status
    cleanup
    
    echo
    log_success "Deployment completed successfully!"
    log_info "Run 'docker-compose -f $COMPOSE_FILE logs -f' to view logs"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [environment] [version]"
        echo
        echo "Environments:"
        echo "  development, dev    - Development environment"
        echo "  staging, stage      - Staging environment"
        echo "  production, prod    - Production environment"
        echo
        echo "Examples:"
        echo "  $0 development"
        echo "  $0 production v1.2.3"
        echo "  $0 staging latest"
        exit 0
        ;;
    --status)
        show_status
        exit 0
        ;;
    --rollback)
        rollback
        exit 0
        ;;
    --cleanup)
        cleanup
        exit 0
        ;;
    *)
        main
        ;;
esac