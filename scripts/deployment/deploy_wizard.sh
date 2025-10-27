#!/bin/bash
# Interactive Cloud Deployment Wizard
# Makes deployment even easier with prompts
set +H  # Disable history expansion to avoid "event ! not found" errors

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear

cat << 'EOF'

 
 Traffic Forecast - Cloud Deployment Wizard 
 Academic v4.0 
 

EOF

echo ""
echo -e "${CYAN}This wizard will help you deploy a 7-day traffic data"
echo -e "collection system on Google Cloud Platform.${NC}"
echo ""

# Step 1: Check gcloud
echo -e "${BLUE}[1/7] Checking gcloud CLI...${NC}"
if ! command -v gcloud &> /dev/null; then
 echo -e "${RED} gcloud CLI not found!${NC}"
 echo ""
 echo "Please install Google Cloud SDK first:"
 echo "https://cloud.google.com/sdk/docs/install"
 exit 1
fi
echo -e "${GREEN} gcloud CLI found${NC}"
sleep 1

# Step 2: Check authentication
echo ""
echo -e "${BLUE}[2/7] Checking authentication...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
 echo -e "${YELLOW} Not authenticated${NC}"
 read -p "Run 'gcloud auth login' now? (y/n): " -n 1 -r
 echo
 if [[ $REPLY =~ ^[Yy]$ ]]; then
 gcloud auth login
 else
 echo "Please run: gcloud auth login"
 exit 1
 fi
fi
ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
echo -e "${GREEN} Authenticated as: $ACCOUNT${NC}"
sleep 1

# Step 3: Project ID
echo ""
echo -e "${BLUE}[3/7] GCP Project Configuration${NC}"
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
if [ -n "$CURRENT_PROJECT" ]; then
 echo "Current project: $CURRENT_PROJECT"
 read -p "Use this project? (y/n): " -n 1 -r
 echo
 if [[ $REPLY =~ ^[Yy]$ ]]; then
 export GCP_PROJECT_ID="$CURRENT_PROJECT"
 else
 read -p "Enter project ID: " GCP_PROJECT_ID
 export GCP_PROJECT_ID
 fi
else
 read -p "Enter your GCP project ID: " GCP_PROJECT_ID
 export GCP_PROJECT_ID
fi

gcloud config set project $GCP_PROJECT_ID
echo -e "${GREEN} Project set: $GCP_PROJECT_ID${NC}"
sleep 1

# Step 4: API Mode
echo ""
echo -e "${BLUE}[4/7] API Configuration${NC}"
echo ""
echo "Choose API mode:"
echo " 1) Mock API (FREE - simulated data, $0 API cost)"
echo " 2) Real API (Production - real data, ~$168 for 7 days)"
echo ""
read -p "Enter choice (1 or 2): " -n 1 -r API_CHOICE
echo ""

if [ "$API_CHOICE" = "2" ]; then
 echo ""
 echo -e "${YELLOW}Real API mode selected${NC}"
 echo "You'll need a Google Maps API key."
 echo ""
 read -p "Enter Google Maps API key: " GOOGLE_MAPS_API_KEY
 export GOOGLE_MAPS_API_KEY
 export USE_REAL_API="true"
 echo -e "${GREEN} Real API configured${NC}"
 echo -e "${YELLOW} Estimated cost: ~\$180 for 7 days${NC}"
else
 export USE_REAL_API="false"
 echo -e "${GREEN} Mock API selected (FREE)${NC}"
fi
sleep 1

# Step 5: Zone
echo ""
echo -e "${BLUE}[5/7] Region Configuration${NC}"
echo ""
echo "Recommended zones:"
echo " 1) asia-southeast1-b (Singapore - closest to Vietnam)"
echo " 2) us-central1-a (US - cheaper)"
echo " 3) europe-west1-b (Belgium)"
echo ""
read -p "Enter choice (1/2/3) or press Enter for default [1]: " ZONE_CHOICE
echo ""

case $ZONE_CHOICE in
 2)
 export GCP_ZONE="us-central1-a"
 ;;
 3)
 export GCP_ZONE="europe-west1-b"
 ;;
 *)
 export GCP_ZONE="asia-southeast1-b"
 ;;
esac

echo -e "${GREEN} Zone: $GCP_ZONE${NC}"
sleep 1

# Step 6: Summary
echo ""
echo -e "${BLUE}[6/7] Deployment Summary${NC}"
echo ""
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo " Project ID: $GCP_PROJECT_ID"
echo " Zone: $GCP_ZONE"
echo " API Mode: $([ "$USE_REAL_API" = "true" ] && echo "Real API" || echo "Mock API (FREE)")"
echo " Duration: 7 days"
echo " VM Type: e2-standard-2 (2 vCPU, 8GB RAM)"
echo ""
echo -e "${CYAN}Estimated Cost:${NC}"
if [ "$USE_REAL_API" = "true" ]; then
 echo " VM (7 days): ~\$12"
 echo " Google API (7 days): ~\$168"
 echo " Total: ~\$180"
else
 echo " VM (7 days): ~\$12"
 echo " Google API: \$0 (Mock)"
 echo " Total: ~\$12"
fi
echo ""
echo -e "${CYAN}What will happen:${NC}"
echo " 1. Create VM instance on GCP"
echo " 2. Install dependencies (Miniconda, Python)"
echo " 3. Clone repository and setup environment"
echo " 4. Create systemd service for continuous collection"
echo " 5. Start data collection (adaptive schedule)"
echo ""
echo "Time: 10-15 minutes (automated)"
echo ""
echo ""

read -p "Proceed with deployment? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
 echo "Deployment cancelled."
 exit 0
fi

# Step 7: Deploy
echo ""
echo -e "${BLUE}[7/7] Starting Deployment...${NC}"
echo ""
echo -e "${YELLOW}This will take 10-15 minutes. Please wait...${NC}"
echo ""

sleep 2

# Run deployment script
./scripts/deploy_week_collection.sh

# Success
echo ""
echo -e "${GREEN}${NC}"
echo -e "${GREEN} ${NC}"
echo -e "${GREEN} Deployment Successful! ${NC}"
echo -e "${GREEN} ${NC}"
echo -e "${GREEN}${NC}"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo ""
echo "1. Monitor collection:"
echo " ./scripts/monitor_collection.sh"
echo ""
echo "2. View logs:"
echo " gcloud compute ssh traffic-collector-v4 --zone=$GCP_ZONE \\"
echo " --command=\"tail -f ~/dsp391m_project/logs/collector.log\""
echo ""
echo "3. After 7 days, download data:"
echo " ./scripts/download_data.sh"
echo ""
echo "4. Delete VM (important!):"
echo " gcloud compute instances delete traffic-collector-v4 --zone=$GCP_ZONE"
echo ""
echo -e "${CYAN}Documentation:${NC}"
echo " cat CLOUD_DEPLOY.md # Full guide"
echo " cat DEPLOY_NOW.md # Quick start"
echo " ./scripts/cloud_quickref.sh # Command reference"
echo ""
echo -e "${GREEN}Happy data collecting! ${NC}"
echo ""
