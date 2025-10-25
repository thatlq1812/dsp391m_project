#!/bin/bash
# Multi-User Access Setup Script
# For Traffic Forecast System on GCP VM

set -e

echo "========================================="
echo "Traffic Forecast - User Access Setup"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
PROJECT_DIR="/opt/traffic-forecast"
CONDA_PATH="$HOME/miniconda3"

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root/sudo
if [ "$EUID" -ne 0 ]; then 
    print_error "Please run as root or with sudo"
    exit 1
fi

# Step 1: Create groups
print_step "Creating user groups..."
groupadd -f traffic-admin
groupadd -f traffic-readonly
print_info "Groups created: traffic-admin, traffic-readonly"

# Step 2: Setup shared project directory
print_step "Setting up shared project directory..."

# Check if project exists in home directory
HOME_PROJECT="$HOME/dsp391m_project"
if [ -d "$HOME_PROJECT" ]; then
    # Move to shared location
    if [ ! -d "$PROJECT_DIR" ]; then
        mkdir -p /opt
        cp -r "$HOME_PROJECT" "$PROJECT_DIR"
        print_info "Project moved to $PROJECT_DIR"
    fi
else
    print_error "Project not found at $HOME_PROJECT"
    print_info "Please run gcp_setup.sh first"
    exit 1
fi

# Set ownership and permissions
chown -R root:traffic-admin "$PROJECT_DIR"
chmod -R 775 "$PROJECT_DIR"
print_info "Project directory configured for team access"

# Step 3: Setup shared conda environment
print_step "Configuring shared conda environment..."
if [ -d "$CONDA_PATH" ]; then
    # Create symlink for shared access
    if [ ! -L "/opt/miniconda3" ]; then
        ln -s "$CONDA_PATH" /opt/miniconda3
    fi
    chmod -R 755 "$CONDA_PATH"
    print_info "Conda environment accessible to all users"
else
    print_error "Conda not found at $CONDA_PATH"
fi

# Step 4: User creation function
create_user() {
    local username=$1
    local access_level=$2  # admin or readonly
    
    print_step "Creating user: $username ($access_level)"
    
    # Create user if not exists
    if id "$username" &>/dev/null; then
        print_info "User $username already exists"
    else
        adduser --disabled-password --gecos "" "$username"
        print_info "User $username created"
    fi
    
    # Add to appropriate group
    if [ "$access_level" = "admin" ]; then
        usermod -aG traffic-admin "$username"
        usermod -aG sudo "$username"
        print_info "Added to traffic-admin and sudo groups"
    else
        usermod -aG traffic-readonly "$username"
        print_info "Added to traffic-readonly group"
    fi
    
    # Setup user environment
    local user_home=$(eval echo ~$username)
    
    # Create .bashrc additions
    cat >> "$user_home/.bashrc" << 'EOF'

# Traffic Forecast Project
export PATH="/opt/miniconda3/bin:$PATH"
alias traffic-cd='cd /opt/traffic-forecast'
alias traffic-activate='conda activate dsp'
alias traffic-status='sudo systemctl status traffic-forecast.service'
alias traffic-logs='tail -f /opt/traffic-forecast/logs/service.log'

# Auto-activate environment
if [ -d "/opt/traffic-forecast" ]; then
    echo "Traffic Forecast Project - Type 'traffic-cd' to navigate"
fi
EOF
    
    chown "$username:$username" "$user_home/.bashrc"
    print_info "User environment configured"
    
    # Setup SSH directory
    local ssh_dir="$user_home/.ssh"
    if [ ! -d "$ssh_dir" ]; then
        mkdir -p "$ssh_dir"
        chmod 700 "$ssh_dir"
        touch "$ssh_dir/authorized_keys"
        chmod 600 "$ssh_dir/authorized_keys"
        chown -R "$username:$username" "$ssh_dir"
        print_info "SSH directory created"
    fi
    
    echo ""
    echo "User $username created successfully!"
    echo "SSH public key file: $ssh_dir/authorized_keys"
    echo ""
}

# Step 5: Interactive user creation
echo ""
echo "User Creation Options:"
echo "1. Create admin user (full access + sudo)"
echo "2. Create readonly user (view only)"
echo "3. Skip user creation"
echo ""
read -p "Select option (1-3): " option

case $option in
    1)
        read -p "Enter username for admin user: " admin_user
        create_user "$admin_user" "admin"
        
        # Set password
        echo ""
        echo "Set password for $admin_user:"
        passwd "$admin_user"
        ;;
    2)
        read -p "Enter username for readonly user: " readonly_user
        create_user "$readonly_user" "readonly"
        
        # Set password
        echo ""
        echo "Set password for $readonly_user:"
        passwd "$readonly_user"
        ;;
    3)
        print_info "Skipping user creation"
        ;;
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

# Step 6: SSH Configuration
print_step "Configuring SSH access..."

# Backup original config
cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

# Enable password authentication (for initial setup)
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
sed -i 's/^#*PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Restart SSH service
systemctl restart sshd
print_info "SSH configured for password and key-based authentication"

# Step 7: Firewall Configuration
print_step "Configuring firewall..."
if command -v ufw &> /dev/null; then
    ufw allow ssh
    ufw --force enable
    print_info "Firewall configured (SSH allowed)"
else
    print_info "UFW not installed, skipping firewall configuration"
fi

# Step 8: Create user guide
print_step "Creating user guide..."
cat > "$PROJECT_DIR/TEAM_ACCESS.md" << 'EOF'
# Team Access Guide

## Connection Information

**Server**: [Ask admin for IP/hostname]
**Project Location**: /opt/traffic-forecast

## Connecting to VM

### Option 1: Password Authentication
```bash
ssh USERNAME@SERVER_IP
# Enter your password when prompted
```

### Option 2: SSH Key Authentication (Recommended)

1. Generate SSH key on your local machine:
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

2. Copy your public key:
```bash
cat ~/.ssh/id_rsa.pub
```

3. Send the public key to the admin

4. Connect:
```bash
ssh USERNAME@SERVER_IP
```

## Quick Start

After connecting:

```bash
# Navigate to project
traffic-cd
# or: cd /opt/traffic-forecast

# Activate environment
traffic-activate
# or: conda activate dsp

# Check system status
traffic-status
# or: sudo systemctl status traffic-forecast.service

# View logs
traffic-logs
# or: tail -f logs/service.log
```

## Common Tasks

### Check Schedule
```bash
conda activate dsp
python scripts/collect_and_render.py --print-schedule
```

### Run Manual Collection
```bash
conda activate dsp
python scripts/collect_and_render.py --once
```

### View Database
```bash
sqlite3 traffic_history.db "SELECT COUNT(*) FROM traffic_history;"
```

### Check Recent Data
```bash
ls -lt data/node/ | head -10
```

## Access Levels

### Admin Users
- Full project access (read/write)
- Can start/stop services
- Can modify configurations
- Sudo privileges

### Readonly Users
- View project files
- Read logs and data
- Cannot modify files
- Cannot restart services

## File Locations

- **Project**: /opt/traffic-forecast
- **Logs**: /opt/traffic-forecast/logs
- **Data**: /opt/traffic-forecast/data
- **Config**: /opt/traffic-forecast/configs
- **Database**: /opt/traffic-forecast/traffic_history.db

## Support

- Documentation: /opt/traffic-forecast/DEPLOY.md
- Contact: [Admin contact]
EOF

print_info "Team access guide created: $PROJECT_DIR/TEAM_ACCESS.md"

# Completion
echo ""
echo "========================================="
echo -e "${GREEN}User Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Project Location: $PROJECT_DIR"
echo "User Guide: $PROJECT_DIR/TEAM_ACCESS.md"
echo ""
echo "Created Groups:"
echo "  - traffic-admin: Full access + sudo"
echo "  - traffic-readonly: Read-only access"
echo ""
echo "SSH Configuration:"
echo "  - Password authentication: ENABLED"
echo "  - Public key authentication: ENABLED"
echo ""
echo "Next Steps for Team Members:"
echo ""
echo "1. Share connection details:"
echo "   Server: [YOUR_SERVER_IP]"
echo "   Username: [their_username]"
echo "   Password: [their_password]"
echo ""
echo "2. For SSH key setup:"
echo "   - Team member generates key: ssh-keygen -t rsa -b 4096"
echo "   - Share their public key with you"
echo "   - Add to: /home/USERNAME/.ssh/authorized_keys"
echo ""
echo "3. Share user guide:"
echo "   cat $PROJECT_DIR/TEAM_ACCESS.md"
echo ""
echo "========================================="
