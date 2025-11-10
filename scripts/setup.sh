#!/bin/bash

# SatyaAI Setup Script
# Prepares the system for SatyaAI deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [[ -f /etc/debian_version ]]; then
            OS="debian"
        elif [[ -f /etc/redhat-release ]]; then
            OS="redhat"
        else
            OS="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        error "Unsupported operating system: $OSTYPE"
    fi
    
    log "Detected OS: $OS"
}

# Update system packages
update_system() {
    log "Updating system packages..."
    
    case $OS in
        "debian")
            sudo apt update && sudo apt upgrade -y
            ;;
        "redhat")
            sudo yum update -y
            ;;
        "macos")
            # Check if Homebrew is installed
            if ! command -v brew &> /dev/null; then
                log "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew update
            ;;
    esac
    
    success "System packages updated"
}

# Install Node.js
install_nodejs() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        log "Node.js already installed: $NODE_VERSION"
        
        # Check if version is 18 or higher
        MAJOR_VERSION=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/v//')
        if [[ $MAJOR_VERSION -lt 18 ]]; then
            warning "Node.js version is too old. Installing Node.js 18..."
        else
            return 0
        fi
    fi
    
    log "Installing Node.js 18..."
    
    case $OS in
        "debian")
            curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
            sudo apt-get install -y nodejs
            ;;
        "redhat")
            curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
            sudo yum install -y nodejs
            ;;
        "macos")
            brew install node@18
            ;;
    esac
    
    success "Node.js installed: $(node --version)"
}

# Install Python 3
install_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version)
        log "Python already installed: $PYTHON_VERSION"
        
        # Check if version is 3.8 or higher
        VERSION_CHECK=$(python3 -c "import sys; print(sys.version_info >= (3, 8))")
        if [[ $VERSION_CHECK == "False" ]]; then
            warning "Python version is too old. Please upgrade to Python 3.8+"
        fi
    else
        log "Installing Python 3..."
        
        case $OS in
            "debian")
                sudo apt install -y python3 python3-pip python3-venv python3-dev
                ;;
            "redhat")
                sudo yum install -y python3 python3-pip python3-devel
                ;;
            "macos")
                brew install python@3.11
                ;;
        esac
        
        success "Python installed: $(python3 --version)"
    fi
    
    # Install pip if not available
    if ! command -v pip3 &> /dev/null; then
        log "Installing pip..."
        curl https://bootstrap.pypa.io/get-pip.py | python3
    fi
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    
    case $OS in
        "debian")
            sudo apt install -y \
                build-essential \
                curl \
                git \
                nginx \
                sqlite3 \
                libsqlite3-dev \
                libjpeg-dev \
                libpng-dev \
                libfreetype6-dev \
                liblcms2-dev \
                libopenjp2-7-dev \
                libtiff5-dev \
                libffi-dev \
                libssl-dev \
                python3-opencv \
                ffmpeg
            ;;
        "redhat")
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y \
                curl \
                git \
                nginx \
                sqlite \
                sqlite-devel \
                libjpeg-turbo-devel \
                libpng-devel \
                freetype-devel \
                lcms2-devel \
                openjpeg2-devel \
                libtiff-devel \
                libffi-devel \
                openssl-devel \
                opencv-python3 \
                ffmpeg
            ;;
        "macos")
            brew install \
                git \
                nginx \
                sqlite \
                jpeg \
                libpng \
                freetype \
                lcms2 \
                openjpeg \
                libtiff \
                libffi \
                openssl \
                opencv \
                ffmpeg
            ;;
    esac
    
    success "System dependencies installed"
}

# Install PM2
install_pm2() {
    if command -v pm2 &> /dev/null; then
        log "PM2 already installed: $(pm2 --version)"
        return 0
    fi
    
    log "Installing PM2..."
    npm install -g pm2
    
    success "PM2 installed: $(pm2 --version)"
}

# Create deploy user
create_deploy_user() {
    if id "deploy" &>/dev/null; then
        log "Deploy user already exists"
        return 0
    fi
    
    log "Creating deploy user..."
    
    case $OS in
        "debian"|"redhat")
            sudo useradd -m -s /bin/bash deploy
            sudo usermod -aG sudo deploy
            ;;
        "macos")
            warning "Manual user creation required on macOS"
            return 0
            ;;
    esac
    
    success "Deploy user created"
}

# Setup directories
setup_directories() {
    log "Setting up directories..."
    
    # Create application directory
    sudo mkdir -p /var/www/satyaai
    sudo chown deploy:deploy /var/www/satyaai
    
    # Create backup directory
    sudo mkdir -p /var/backups/satyaai
    sudo chown deploy:deploy /var/backups/satyaai
    
    # Create log directory
    sudo mkdir -p /var/log/satyaai
    sudo chown deploy:deploy /var/log/satyaai
    
    success "Directories created"
}

# Configure firewall
configure_firewall() {
    log "Configuring firewall..."
    
    case $OS in
        "debian")
            if command -v ufw &> /dev/null; then
                sudo ufw allow 22/tcp
                sudo ufw allow 80/tcp
                sudo ufw allow 443/tcp
                sudo ufw allow 3000/tcp
                sudo ufw --force enable
                success "UFW firewall configured"
            else
                warning "UFW not installed, skipping firewall configuration"
            fi
            ;;
        "redhat")
            if command -v firewall-cmd &> /dev/null; then
                sudo firewall-cmd --permanent --add-service=ssh
                sudo firewall-cmd --permanent --add-service=http
                sudo firewall-cmd --permanent --add-service=https
                sudo firewall-cmd --permanent --add-port=3000/tcp
                sudo firewall-cmd --reload
                success "Firewalld configured"
            else
                warning "Firewalld not installed, skipping firewall configuration"
            fi
            ;;
        "macos")
            warning "Manual firewall configuration required on macOS"
            ;;
    esac
}

# Generate SSL certificate (self-signed for development)
generate_ssl_cert() {
    log "Generating SSL certificate..."
    
    SSL_DIR="/etc/nginx/ssl"
    sudo mkdir -p $SSL_DIR
    
    if [[ ! -f "$SSL_DIR/cert.pem" ]]; then
        sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout $SSL_DIR/key.pem \
            -out $SSL_DIR/cert.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        success "Self-signed SSL certificate generated"
        warning "For production, replace with a proper SSL certificate"
    else
        log "SSL certificate already exists"
    fi
}

# Setup environment file
setup_environment() {
    log "Setting up environment configuration..."
    
    ENV_FILE="/var/www/satyaai/.env"
    
    if [[ ! -f "$ENV_FILE" ]]; then
        cat > "$ENV_FILE" << EOF
# SatyaAI Production Configuration
NODE_ENV=production
PORT=3000
PYTHON_SERVER_PORT=5001

# Security (CHANGE THESE IN PRODUCTION!)
JWT_SECRET=$(openssl rand -base64 32)
SESSION_SECRET=$(openssl rand -base64 32)

# CORS
CORS_ORIGIN=http://localhost:3000

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_SKIP_SUCCESSFUL_REQUESTS=false

# File Upload
MAX_FILE_SIZE=52428800

# Database
DATABASE_URL=sqlite:./db.sqlite

# Logging
LOG_LEVEL=info
EOF
        
        sudo chown deploy:deploy "$ENV_FILE"
        sudo chmod 600 "$ENV_FILE"
        
        success "Environment file created"
        warning "Please review and update the environment variables in $ENV_FILE"
    else
        log "Environment file already exists"
    fi
}

# Main setup function
main() {
    log "Starting SatyaAI system setup..."
    
    detect_os
    update_system
    install_nodejs
    install_python
    install_system_deps
    install_pm2
    create_deploy_user
    setup_directories
    configure_firewall
    generate_ssl_cert
    setup_environment
    
    success "🎉 SatyaAI system setup completed!"
    log "Next steps:"
    log "1. Switch to deploy user: sudo su - deploy"
    log "2. Run deployment script: ./scripts/deploy.sh"
    log "3. Update environment variables in /var/www/satyaai/.env"
}

# Handle script arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "deps")
        detect_os
        install_system_deps
        ;;
    "user")
        create_deploy_user
        ;;
    "dirs")
        setup_directories
        ;;
    "ssl")
        generate_ssl_cert
        ;;
    *)
        echo "Usage: $0 {setup|deps|user|dirs|ssl}"
        exit 1
        ;;
esac