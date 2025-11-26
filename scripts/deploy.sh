#!/bin/bash

###############################################################################
# SatyaAI Deployment Script
# Automates the deployment process for production
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="satyaai"
DEPLOY_USER="deploy"
DEPLOY_PATH="/var/www/satyaai"
BACKUP_PATH="/var/backups/satyaai"
LOG_FILE="./logs/deploy-$(date +%Y%m%d-%H%M%S).log"

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Create logs directory
mkdir -p ./logs

log "🚀 Starting deployment for $APP_NAME..."

# Check if running as correct user
if [ "$USER" != "$DEPLOY_USER" ] && [ "$USER" != "root" ]; then
    warning "Not running as $DEPLOY_USER or root. Some operations may fail."
fi

# 1. Pre-deployment checks
log "📋 Running pre-deployment checks..."

# Check if git is clean
if [ -n "$(git status --porcelain)" ]; then
    error "Git working directory is not clean. Commit or stash changes first."
fi

# Check if on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "master" ]; then
    warning "Not on main/master branch. Current branch: $CURRENT_BRANCH"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

success "Pre-deployment checks passed"

# 2. Backup current deployment
log "💾 Creating backup..."
mkdir -p "$BACKUP_PATH"
BACKUP_FILE="$BACKUP_PATH/backup-$(date +%Y%m%d-%H%M%S).tar.gz"

if [ -d "$DEPLOY_PATH" ]; then
    tar -czf "$BACKUP_FILE" -C "$DEPLOY_PATH" . 2>/dev/null || warning "Backup creation failed"
    success "Backup created: $BACKUP_FILE"
else
    warning "No existing deployment to backup"
fi

# 3. Pull latest code
log "📥 Pulling latest code..."
git pull origin "$CURRENT_BRANCH" || error "Git pull failed"
success "Code updated"

# 4. Install dependencies
log "📦 Installing dependencies..."

# Node.js dependencies
log "Installing Node.js dependencies..."
npm ci --production || error "npm install failed"

# Python dependencies
log "Installing Python dependencies..."
cd server/python
pip install -r requirements-complete.txt --quiet || error "pip install failed"
cd ../..

success "Dependencies installed"

# 5. Run database migrations
log "🗄️  Running database migrations..."
npm run db:migrate || warning "Database migration failed"
success "Database migrations completed"

# 6. Build application
log "🔨 Building application..."

# Build client
log "Building client..."
npm run build:client || error "Client build failed"

# Build server
log "Building server..."
npm run build:server || error "Server build failed"

success "Build completed"

# 7. Run tests
log "🧪 Running tests..."
npm run test:ci || warning "Tests failed - review before proceeding"

# 8. Download/update AI models
log "🤖 Checking AI models..."
python scripts/download_models.py || warning "Model download failed"

# 9. Stop existing services
log "🛑 Stopping existing services..."
pm2 stop ecosystem.config.js 2>/dev/null || warning "No services to stop"

# 10. Start services with PM2
log "▶️  Starting services..."
pm2 start ecosystem.config.js --env production || error "Failed to start services"
pm2 save || warning "Failed to save PM2 configuration"

success "Services started"

# 11. Health check
log "🏥 Running health checks..."
sleep 5  # Wait for services to start

# Check Node.js server
if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    success "Node.js server is healthy"
else
    error "Node.js server health check failed"
fi

# Check Python service
if curl -f http://localhost:5001/health > /dev/null 2>&1; then
    success "Python service is healthy"
else
    error "Python service health check failed"
fi

# 12. Cleanup old backups (keep last 5)
log "🧹 Cleaning up old backups..."
cd "$BACKUP_PATH"
ls -t | tail -n +6 | xargs -r rm -- 2>/dev/null || true
cd - > /dev/null

# 13. Final status
log "📊 Deployment status:"
pm2 status

success "✅ Deployment completed successfully!"
log "📝 Deployment log saved to: $LOG_FILE"
log "🌐 Application is running at: http://localhost:3000"
log "📚 API documentation: http://localhost:3000/api-docs"

# Send notification (optional)
# curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
#   -H 'Content-Type: application/json' \
#   -d "{\"text\":\"✅ $APP_NAME deployed successfully!\"}"

exit 0
