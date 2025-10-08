#!/bin/bash
# Remote Deployment Script for Traffic Forecast System
# Usage: ./deploy.sh [environment] [host]
# Example: ./deploy.sh production user@your-server.com

set -e

ENVIRONMENT=${1:-development}
REMOTE_HOST=${2}

if [ -z "$REMOTE_HOST" ]; then
    echo "Usage: $0 [environment] [user@host]"
    echo "Example: $0 production user@traffic-server.com"
    exit 1
fi

echo "Deploying Traffic Forecast System to $REMOTE_HOST ($ENVIRONMENT)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="traffic-forecast-node-radius"
REMOTE_DIR="/opt/$PROJECT_NAME"
BACKUP_DIR="/opt/${PROJECT_NAME}_backups"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    command -v rsync >/dev/null 2>&1 || { log_error "rsync is required but not installed."; exit 1; }
    command -v ssh >/dev/null 2>&1 || { log_error "ssh is required but not installed."; exit 1; }
}

# Setup remote server
setup_remote_server() {
    log_info "Setting up remote server..."

    ssh $REMOTE_HOST << EOF
        # Update system
        sudo apt update && sudo apt upgrade -y

        # Install required packages
        sudo apt install -y python3 python3-pip python3-venv git htop tmux ufw fail2ban

        # Install Docker (optional)
        sudo apt install -y docker.io docker-compose
        sudo systemctl enable docker
        sudo systemctl start docker
        sudo usermod -aG docker \$USER

        # Create project directory
        sudo mkdir -p $REMOTE_DIR
        sudo mkdir -p $BACKUP_DIR
        sudo chown -R \$USER:\$USER $REMOTE_DIR
        sudo chown -R \$USER:\$USER $BACKUP_DIR

        # Setup firewall
        sudo ufw allow ssh
        sudo ufw allow 8000/tcp
        sudo ufw --force enable

        # Setup log rotation
        sudo tee /etc/logrotate.d/$PROJECT_NAME > /dev/null <<EOL
$REMOTE_DIR/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 \$USER \$USER
}
EOL

        echo "Remote server setup completed!"
EOF
}

# Backup existing deployment
backup_existing() {
    log_info "Creating backup of existing deployment..."

    ssh $REMOTE_HOST << EOF
        if [ -d "$REMOTE_DIR" ]; then
            TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
            BACKUP_FILE="$BACKUP_DIR/backup_\$TIMESTAMP.tar.gz"

            echo "Creating backup: \$BACKUP_FILE"
            cd $REMOTE_DIR/..
            tar -czf \$BACKUP_FILE $PROJECT_NAME/

            # Keep only last 5 backups
            cd $BACKUP_DIR
            ls -t backup_*.tar.gz | tail -n +6 | xargs -r rm
        fi
EOF
}

# Deploy code
deploy_code() {
    log_info "Deploying code to remote server..."

    # Create temporary exclude file
    cat > /tmp/deploy_exclude.txt << EOF
.git
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
node_modules/
*.log
data/
models/
.cache/
.pytest_cache/
.coverage
htmlcov/
.DS_Store
Thumbs.db
.vscode/
.idea/
*.swp
*.swo
*~
EOF

    # Sync code
    rsync -avz --delete --exclude-from=/tmp/deploy_exclude.txt \
        ./ $REMOTE_HOST:$REMOTE_DIR/

    # Cleanup
    rm /tmp/deploy_exclude.txt
}

# Setup environment
setup_environment() {
    log_info "Setting up environment on remote server..."

    ssh $REMOTE_HOST << EOF
        cd $REMOTE_DIR

        # Create virtual environment
        python3 -m venv venv
        source venv/bin/activate

        # Install dependencies
        pip install --upgrade pip
        pip install -r requirements.txt

        # Create necessary directories
        mkdir -p data models logs configs

        # Set permissions
        chmod +x scripts/*.sh
        chmod +x *.py

        echo "Environment setup completed!"
EOF
}

# Configure services
configure_services() {
    log_info "Configuring services..."

    # Create systemd service for API
    cat > /tmp/traffic-api.service << EOF
[Unit]
Description=Traffic Forecast API
After=network.target

[Service]
Type=simple
User=\$USER
WorkingDirectory=$REMOTE_DIR
Environment=PATH=$REMOTE_DIR/venv/bin
ExecStart=$REMOTE_DIR/venv/bin/uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Create systemd service for scheduler
    cat > /tmp/traffic-scheduler.service << EOF
[Unit]
Description=Traffic Forecast Scheduler
After=network.target

[Service]
Type=simple
User=\$USER
WorkingDirectory=$REMOTE_DIR
Environment=PATH=$REMOTE_DIR/venv/bin
ExecStart=$REMOTE_DIR/venv/bin/python apps/scheduler/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Copy services to remote
    scp /tmp/traffic-api.service $REMOTE_HOST:/tmp/
    scp /tmp/traffic-scheduler.service $REMOTE_HOST:/tmp/

    # Install services
    ssh $REMOTE_HOST << EOF
        sudo mv /tmp/traffic-api.service /etc/systemd/system/
        sudo mv /tmp/traffic-scheduler.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable traffic-api
        sudo systemctl enable traffic-scheduler
EOF

    # Cleanup
    rm /tmp/traffic-api.service /tmp/traffic-scheduler.service
}

# Post-deployment tasks
post_deployment() {
    log_info "Running post-deployment tasks..."

    ssh $REMOTE_HOST << EOF
        cd $REMOTE_DIR

        # Initial data collection (optional)
        echo "Running initial setup..."
        source venv/bin/activate

        # Run initial collectors if data doesn't exist
        if [ ! -f "data/nodes.json" ]; then
            echo "Running initial data collection..."
            python run_collectors.py
        fi

        # Test API
        echo "Testing API..."
        timeout 10 bash -c 'until curl -f http://localhost:8000/; do sleep 2; done' && echo "API is responding!" || echo "API test failed"

        echo "Post-deployment completed!"
EOF
}

# Health check
health_check() {
    log_info "Performing health check..."

    ssh $REMOTE_HOST << EOF
        echo "=== System Health Check ==="
        echo "CPU Usage:"
        top -bn1 | head -3

        echo -e "\nMemory Usage:"
        free -h

        echo -e "\nDisk Usage:"
        df -h $REMOTE_DIR

        echo -e "\nService Status:"
        sudo systemctl status traffic-api --no-pager -l | head -10
        sudo systemctl status traffic-scheduler --no-pager -l | head -10

        echo -e "\nProcess Check:"
        ps aux | grep -E "(uvicorn|python.*scheduler)" | grep -v grep

        echo -e "\nPort Check:"
        netstat -tlnp | grep :8000
EOF
}

# Rollback function
rollback() {
    log_warn "Rolling back to previous version..."

    ssh $REMOTE_HOST << EOF
        # Find latest backup
        LATEST_BACKUP=\$(ls -t $BACKUP_DIR/backup_*.tar.gz | head -1)

        if [ -n "\$LATEST_BACKUP" ]; then
            echo "Rolling back using: \$LATEST_BACKUP"

            # Stop services
            sudo systemctl stop traffic-api traffic-scheduler

            # Remove current deployment
            cd /opt
            rm -rf $PROJECT_NAME

            # Extract backup
            tar -xzf \$LATEST_BACKUP

            # Restart services
            sudo systemctl start traffic-api traffic-scheduler

            echo "Rollback completed!"
        else
            echo "No backup found for rollback!"
        fi
EOF
}

# Main deployment flow
main() {
    check_prerequisites

    case $ENVIRONMENT in
        production)
            log_info "Production deployment mode"
            ;;
        staging)
            log_info "Staging deployment mode"
            ;;
        development)
            log_info "Development deployment mode"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac

    setup_remote_server
    backup_existing
    deploy_code
    setup_environment
    configure_services
    post_deployment
    health_check

    log_info "Deployment completed successfully!"
    log_info "API should be available at: http://$REMOTE_HOST:8000"
    log_info "Swagger docs at: http://$REMOTE_HOST:8000/docs"
}

# Handle command line arguments
case "\${3:-}" in
    --rollback)
        rollback
        ;;
    --health-check)
        health_check
        ;;
    *)
        main
        ;;
esac