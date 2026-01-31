#!/bin/bash

# SSL Certificate Setup Script for SatyaAI
# This script generates self-signed certificates for development
# For production, use Let's Encrypt instead

set -e

SSL_DIR="./nginx/ssl"
CERT_FILE="$SSL_DIR/nginx-selfsigned.crt"
KEY_FILE="$SSL_DIR/nginx-selfsigned.key"

# Create SSL directory if it doesn't exist
mkdir -p "$SSL_DIR"

# Generate self-signed certificate
if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo "Generating self-signed SSL certificate..."
    
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$KEY_FILE" \
        -out "$CERT_FILE" \
        -subj "/C=US/ST=State/L=City/O=SatyaAI/CN=localhost" \
        -addext "subjectAltName=DNS:localhost,DNS:www.localhost,IP:127.0.0.1"
    
    echo "SSL certificate generated successfully!"
else
    echo "SSL certificate already exists."
fi

# Set proper permissions
chmod 600 "$KEY_FILE"
chmod 644 "$CERT_FILE"

echo "SSL setup complete!"
echo ""
echo "For production deployment, replace the self-signed certificate with Let's Encrypt:"
echo "1. Install certbot: sudo apt-get install certbot python3-certbot-nginx"
echo "2. Get certificate: sudo certbot --nginx -d yourdomain.com"
echo "3. Update nginx.conf to use Let's Encrypt certificates"
