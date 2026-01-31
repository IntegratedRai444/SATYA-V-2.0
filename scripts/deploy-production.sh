#!/bin/bash

# Production Deployment Script for SatyaAI
# This script sets up and deploys the application with SSL

set -e

echo "🚀 Starting SatyaAI Production Deployment..."

# Check if required environment variables are set
if [ -z "$DATABASE_URL" ] || [ -z "$JWT_SECRET" ] || [ -z "$SUPABASE_URL" ]; then
    echo "❌ Missing required environment variables!"
    echo "Please set: DATABASE_URL, JWT_SECRET, SUPABASE_URL, SUPABASE_ANON_KEY"
    exit 1
fi

# Create necessary directories
mkdir -p nginx/ssl logs uploads models

# Setup SSL certificates
echo "🔐 Setting up SSL certificates..."
./scripts/setup-ssl.sh

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Main application is healthy!"
else
    echo "❌ Main application health check failed!"
    docker-compose -f docker-compose.prod.yml logs nodejs
    exit 1
fi

if curl -f http://localhost:9090 > /dev/null 2>&1; then
    echo "✅ Prometheus is running!"
else
    echo "⚠️ Prometheus may not be ready yet"
fi

if curl -f http://localhost:3001 > /dev/null 2>&1; then
    echo "✅ Grafana is running!"
else
    echo "⚠️ Grafana may not be ready yet"
fi

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "📊 Service URLs:"
echo "  - Main App: https://localhost"
echo "  - Grafana: https://localhost:3001"
echo "  - Prometheus: http://localhost:9090"
echo ""
echo "📝 To view logs:"
echo "  docker-compose -f docker-compose.prod.yml logs -f"
echo ""
echo "🛑 To stop services:"
echo "  docker-compose -f docker-compose.prod.yml down"
