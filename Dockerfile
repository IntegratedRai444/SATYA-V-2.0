# Multi-stage Dockerfile for SatyaAI

# Stage 1: Build frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/client

# Copy client package files
COPY client/package*.json ./

# Install client dependencies
RUN npm ci --only=production

# Copy client source
COPY client/ ./

# Build client
RUN npm run build

# Stage 2: Build backend
FROM node:18-alpine AS backend-builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy server source
COPY server/ ./server/
COPY tsconfig.json ./

# Build server
RUN npm run build:server

# Stage 3: Python environment
FROM python:3.14-slim AS python-env

WORKDIR /app/python

# Install Python dependencies
COPY server/python/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source
COPY server/python/ ./

# Stage 4: Production image
FROM node:18-alpine

# Install Python
RUN apk add --no-cache python3 py3-pip

WORKDIR /app

# Copy Node.js dependencies
COPY --from=backend-builder /app/node_modules ./node_modules
COPY --from=backend-builder /app/dist ./dist
COPY --from=backend-builder /app/package.json ./

# Copy frontend build
COPY --from=frontend-builder /app/client/dist ./client/dist

# Copy Python environment
COPY --from=python-env /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=python-env /app/python ./server/python

# Create necessary directories
RUN mkdir -p /app/uploads /app/database /app/logs

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000
ENV PYTHON_PORT=5001

# Expose ports
EXPOSE 3000 5001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

# Start application
CMD ["npm", "start"]
