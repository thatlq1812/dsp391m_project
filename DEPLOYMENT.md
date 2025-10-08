# Traffic Forecast System - Deployment Guide

## Quick Start Deployment

### Option 1: Automated Script (Recommended)

```bash
# Make executable
chmod +x deploy.sh

# Deploy to server
./deploy.sh production user@your-server.com

# Check health
./deploy.sh production user@your-server.com --health-check
```

### Option 2: Docker Deployment

```bash
# Generate SSL certificates
./generate_ssl.sh yourdomain.com

# Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose logs -f
```

### Option 3: Manual Deployment

```bash
# 1. Server setup
sudo apt update && sudo apt install -y python3 python3-pip git nginx

# 2. Clone and setup
git clone <repo> /opt/traffic-forecast
cd /opt/traffic-forecast
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure environment
cp .env_template .env
# Edit .env with your API keys

# 4. Setup services
sudo cp infra/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable traffic-api traffic-scheduler
sudo systemctl start traffic-api traffic-scheduler

# 5. Setup nginx
sudo cp infra/nginx/nginx.conf /etc/nginx/nginx.conf
sudo systemctl reload nginx
```

## Post-Deployment Checklist

- [ ] API accessible at https://yourdomain.com
- [ ] Swagger docs at https://yourdomain.com/docs
- [ ] SSL certificate valid
- [ ] Services running: `sudo systemctl status traffic-api`
- [ ] Logs rotating properly
- [ ] Firewall configured
- [ ] Backups scheduled
- [ ] Monitoring alerts setup

## URLs After Deployment

- **API Base**: `https://yourdomain.com`
- **Health Check**: `https://yourdomain.com/health`
- **API Docs**: `https://yourdomain.com/docs`
- **Forecast Endpoint**: `https://yourdomain.com/api/v1/nodes/{node_id}/forecast`

## Monitoring Commands

```bash
# Service status
sudo systemctl status traffic-api traffic-scheduler

# Logs
sudo journalctl -u traffic-api -f
sudo journalctl -u traffic-scheduler -f

# System resources
htop
df -h
free -h

# API health
curl -f https://yourdomain.com/health
```

## Troubleshooting

### API Not Responding
```bash
# Check service
sudo systemctl status traffic-api

# Check logs
sudo journalctl -u traffic-api -n 50

# Test locally
curl http://localhost:8000/
```

### SSL Issues
```bash
# Check certificate
openssl x509 -in /etc/nginx/ssl/cert.pem -text -noout

# Renew Let's Encrypt
sudo certbot renew
sudo systemctl reload nginx
```

### Performance Issues
```bash
# Monitor resources
top
iotop
nload

# Check nginx status
sudo systemctl status nginx
sudo nginx -t
```

## Backup & Recovery

```bash
# Manual backup
tar -czf backup_$(date +%Y%m%d).tar.gz /opt/traffic-forecast-node-radius/

# Automated backup (add to crontab)
# 0 2 * * * /path/to/backup-script.sh

# Restore
tar -xzf backup_file.tar.gz -C /opt/
sudo systemctl restart traffic-api traffic-scheduler
```

## Security Checklist

- [ ] SSH key authentication only
- [ ] Firewall configured (ufw)
- [ ] Fail2Ban installed
- [ ] SSL/TLS enabled
- [ ] API keys in environment variables
- [ ] File permissions correct (600 for keys)
- [ ] Regular security updates
- [ ] Log monitoring enabled

## Support

For issues or questions:
1. Check logs: `sudo journalctl -u traffic-api -f`
2. Review configuration files
3. Test API endpoints manually
4. Check system resources
5. Review firewall rules

## Scaling

### Horizontal Scaling
- Deploy multiple API instances
- Use load balancer (nginx upstream)
- Shared database/redis for state

### Vertical Scaling
- Increase VM CPU/memory
- Optimize database queries
- Add caching layers
- Use CDN for static assets

Happy deploying!