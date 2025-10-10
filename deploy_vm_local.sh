#!/bin/bash
# Traffic Forecast Deployment Script for Google Cloud VM (Local Copy Version)
# Run this script on your Ubuntu VM after copying project files

echo "=== Traffic Forecast System Deployment (Local Copy) ==="
echo "Starting deployment at $(date)"

# Assume project is already copied to ~/dsp391m_project
if [ ! -d "~/dsp391m_project" ]; then
    echo "❌ Error: Project directory not found at ~/dsp391m_project"
    echo "Please copy the project files first:"
    echo "  gcloud compute scp --recurse . traffic-collector-vm:~/dsp391m_project --zone=asia-southeast1-a"
    exit 1
fi

cd ~/dsp391m_project

# 1. Update system
echo "Step 1: Updating system..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip git wget curl build-essential

# 2. Install Miniconda
echo "Step 2: Installing Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc

# Verify conda installation
conda --version

# 3. Setup conda environment
echo "Step 3: Setting up conda environment..."
conda env create -f environment.yml
conda activate dsp

# 4. Install Python dependencies
echo "Step 4: Installing Python dependencies..."
pip install -r requirements.txt

# 5. Configure environment
echo "Step 5: Configuring environment..."
cp env_template .env
echo "Please edit .env file with your API keys:"
echo "  GOOGLE_MAPS_API_KEY=your_google_maps_api_key"
echo "  OPEN_METEO_API_KEY=your_open_meteo_key (optional)"
echo ""
echo "For now, creating basic .env without keys (will use mock data)"
cat > .env << EOF
# Google Maps API Key (required for traffic data)
GOOGLE_MAPS_API_KEY=

# Open-Meteo API (optional, for weather data)
OPEN_METEO_API_KEY=

# Project settings
PROJECT_NAME=traffic-forecast-vm
LOG_LEVEL=INFO
EOF

# 6. Test deployment
echo "Step 6: Testing deployment..."
conda run -n dsp python scripts/collect_and_render.py --once --no-visualize

# Check if test was successful
if [ $? -eq 0 ]; then
    echo "✓ Test collection successful!"
    ls -la data/node/
else
    echo "✗ Test collection failed. Check logs above."
fi

# 7. Setup cron job for 24 hours (every 15 minutes)
echo "Step 7: Setting up cron job for 24-hour collection..."

# Create a wrapper script for cron
cat > run_collection.sh << 'EOF'
#!/bin/bash
# Collection wrapper for cron job
cd /home/fxlqt/dsp391m_project
source $HOME/miniconda3/bin/activate
conda activate dsp
python scripts/collect_and_render.py --interval 900 --no-visualize >> collect_$(date +\%Y\%m\%d).log 2>&1
EOF

chmod +x run_collection.sh

# Add to crontab (runs every 15 minutes)
(crontab -l ; echo "*/15 * * * * /home/fxlqt/dsp391m_project/run_collection.sh") | crontab -

echo "✓ Cron job added. Will run every 15 minutes."

# 8. Setup cleanup job (after 24 hours)
echo "Step 8: Setting up cleanup after 24 hours..."
echo "0 10 * * * rm -rf /home/fxlqt/dsp391m_project/data/node/* /home/fxlqt/dsp391m_project/data/images/*" | crontab -

# 9. Setup monitoring
echo "Step 9: Setting up monitoring..."
cat > monitor.sh << 'EOF'
#!/bin/bash
echo "=== Traffic Forecast Monitor ==="
echo "Date: $(date)"
echo "Disk usage:"
df -h /
echo ""
echo "Data directory size:"
du -sh data/
echo ""
echo "Recent collections:"
ls -la data/node/ | tail -5
echo ""
echo "Log files:"
ls -la *.log 2>/dev/null || echo "No log files yet"
echo ""
echo "Running processes:"
ps aux | grep python | grep -v grep || echo "No Python processes running"
EOF

chmod +x monitor.sh

echo ""
echo "=== Deployment Complete! ==="
echo "VM External IP: $(curl -s http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H "Metadata-Flavor: Google")"
echo ""
echo "Data storage locations:"
echo "  - Collection data: ./data/node/ (timestamped folders)"
echo "  - Images: ./data/images/"
echo "  - Cache: ./cache/"
echo "  - Logs: collect_YYYYMMDD.log"
echo ""
echo "Monitoring commands:"
echo "  ./monitor.sh              # Check system status"
echo "  tail -f collect_*.log     # Monitor collection logs"
echo "  crontab -l               # View scheduled jobs"
echo ""
echo "To stop collection after 24 hours:"
echo "  crontab -r              # Remove all cron jobs"
echo ""
echo "Collection will start automatically every 15 minutes."
echo "Check back in 1-2 hours to verify data is being collected."