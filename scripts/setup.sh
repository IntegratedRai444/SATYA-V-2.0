#!/bin/bash

# SatyaAI Quick Setup Script
# This script sets up everything you need to run SatyaAI

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                  SatyaAI Setup Script                      ║"
echo "║          AI-Powered Deepfake Detection Platform            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Node.js
echo -e "${BLUE}Checking Node.js...${NC}"
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v)
    echo -e "${GREEN}✓ Node.js $NODE_VERSION installed${NC}"
else
    echo -e "${RED}✗ Node.js not found${NC}"
    echo "Please install Node.js 18+ from https://nodejs.org"
    exit 1
fi

# Check Python
echo -e "${BLUE}Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ $PYTHON_VERSION installed${NC}"
else
    echo -e "${RED}✗ Python not found${NC}"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

# Install Node.js dependencies
echo -e "\n${BLUE}Installing Node.js dependencies...${NC}"
npm install
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Node.js dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install Node.js dependencies${NC}"
    exit 1
fi

# Install client dependencies
echo -e "\n${BLUE}Installing client dependencies...${NC}"
cd client
npm install
cd ..
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Client dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install client dependencies${NC}"
    exit 1
fi

# Install Python dependencies
echo -e "\n${BLUE}Installing Python dependencies...${NC}"
pip3 install -r server/python/requirements-complete.txt --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ Some Python dependencies may have failed${NC}"
fi

# Download AI models
echo -e "\n${BLUE}Downloading AI models...${NC}"
python3 scripts/download_models.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ AI models downloaded${NC}"
else
    echo -e "${YELLOW}⚠ Some models may not have downloaded${NC}"
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo -e "\n${BLUE}Creating .env file...${NC}"
    cat > .env << EOF
# Node.js Server
NODE_ENV=development
PORT=3000

# Python Server
PYTHON_PORT=5001
ENABLE_GPU=false

# Database
DATABASE_URL=./database/db.sqlite

# Security
JWT_SECRET=$(openssl rand -hex 32)
SESSION_SECRET=$(openssl rand -hex 32)

# CORS
FRONTEND_URL=http://localhost:3000

# File Upload
MAX_FILE_SIZE=104857600
UPLOAD_DIR=./uploads

# Optional: OpenAI (for chat assistant)
# OPENAI_API_KEY=sk-your-key-here
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${YELLOW}⚠ .env file already exists${NC}"
fi

# Create necessary directories
echo -e "\n${BLUE}Creating directories...${NC}"
mkdir -p uploads
mkdir -p database
mkdir -p server/python/models
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}"

# Setup complete
echo -e "\n${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Setup Complete! 🎉                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}To start the application:${NC}"
echo ""
echo "  ${GREEN}npm run dev:all${NC}     # Start all servers"
echo ""
echo "Or start individually:"
echo "  ${GREEN}npm run dev${NC}        # Node.js server (port 3000)"
echo "  ${GREEN}cd client && npm run dev${NC}  # React frontend (port 5173)"
echo "  ${GREEN}cd server/python && python3 app.py${NC}  # Python AI (port 5001)"
echo ""
echo -e "${BLUE}Then open:${NC} http://localhost:5173"
echo ""
echo -e "${YELLOW}For deployment, see:${NC} DEPLOYMENT_GUIDE.md"
