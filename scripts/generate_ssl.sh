#!/bin/bash
# SSL Certificate Generation Script
# Usage: ./generate_ssl.sh [domain]

DOMAIN=${1:-localhost}
SSL_DIR="./infra/nginx/ssl"

echo "Generating SSL certificates for $DOMAIN"

# Create SSL directory
mkdir -p $SSL_DIR

# Generate private key
openssl genrsa -out $SSL_DIR/key.pem 2048

# Generate certificate signing request
cat > /tmp/cert.conf << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = VN
ST = Ho Chi Minh City
L = Ho Chi Minh City
O = Traffic Forecast
OU = IT
CN = $DOMAIN

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = www.$DOMAIN
EOF

# Generate self-signed certificate
openssl req -new -x509 -key $SSL_DIR/key.pem -out $SSL_DIR/cert.pem -days 365 -config /tmp/cert.conf

# Set permissions
chmod 600 $SSL_DIR/key.pem
chmod 644 $SSL_DIR/cert.pem

echo "SSL certificates generated successfully!"
echo "Certificate: $SSL_DIR/cert.pem"
echo "Private key: $SSL_DIR/key.pem"
echo ""
echo "WARNING: This is a self-signed certificate!"
echo "   For production, use Let's Encrypt or a proper CA certificate."
echo ""
echo "To use with Let's Encrypt (recommended for production):"
echo "sudo apt install certbot"
echo "sudo certbot certonly --standalone -d $DOMAIN"
echo "Then update nginx.conf with the real certificate paths."