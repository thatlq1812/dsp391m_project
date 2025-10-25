#!/bin/bash
# Quick Start Script for Traffic Forecast System
# Fastest way to get started

echo "========================================="
echo "Traffic Forecast - Quick Start"
echo "Academic v4.0"
echo "========================================="
echo ""

# Check if in project directory
if [ ! -f "environment.yml" ]; then
    echo "ERROR: Not in project directory!"
    echo "Please cd to the project root first"
    exit 1
fi

echo "Choose your setup:"
echo ""
echo "1. Development Setup (Mock API - FREE)"
echo "2. Production Setup (Real Google API - costs money)"
echo "3. Skip setup, just run collection"
echo ""
read -p "Select option (1-3): " option

case $option in
    1)
        echo ""
        echo "Setting up for DEVELOPMENT (Mock API - FREE)..."
        
        # Install dependencies
        if ! command -v conda &> /dev/null; then
            echo "Installing Miniconda..."
            bash scripts/install_dependencies.sh
        fi
        
        # Create environment
        if ! conda env list | grep -q "^dsp "; then
            echo "Creating conda environment..."
            conda env create -f environment.yml
        fi
        
        # Configure for mock API
        if [ ! -f ".env" ]; then
            cp .env.template .env
        fi
        
        # Ensure mock API enabled
        python -c "
import yaml
with open('configs/project_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['google_directions']['use_mock_api'] = True
with open('configs/project_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Mock API enabled')
"
        
        echo ""
        echo "Development setup complete!"
        echo ""
        echo "Next steps:"
        echo "  conda activate dsp"
        echo "  python scripts/collect_and_render.py --once"
        ;;
        
    2)
        echo ""
        echo "Setting up for PRODUCTION (Real API)..."
        echo ""
        read -p "Enter your Google Maps API Key: " api_key
        
        # Install dependencies
        if ! command -v conda &> /dev/null; then
            echo "Installing Miniconda..."
            bash scripts/install_dependencies.sh
        fi
        
        # Create environment
        if ! conda env list | grep -q "^dsp "; then
            echo "Creating conda environment..."
            conda env create -f environment.yml
        fi
        
        # Configure for real API
        if [ ! -f ".env" ]; then
            cp .env.template .env
        fi
        
        echo "GOOGLE_MAPS_API_KEY=$api_key" >> .env
        
        # Disable mock API
        python -c "
import yaml
with open('configs/project_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['google_directions']['use_mock_api'] = False
with open('configs/project_config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Real API enabled')
"
        
        echo ""
        echo "Production setup complete!"
        echo ""
        echo "WARNING: Using real Google API"
        echo "Estimated cost: \$720/month"
        echo ""
        echo "Next steps:"
        echo "  conda activate dsp"
        echo "  python scripts/collect_and_render.py --print-schedule"
        echo "  python scripts/collect_and_render.py --adaptive"
        ;;
        
    3)
        echo ""
        echo "Skipping setup..."
        echo ""
        echo "Make sure you have:"
        echo "  - Conda environment 'dsp' created"
        echo "  - Configuration in configs/project_config.yaml"
        echo ""
        echo "To run:"
        echo "  conda activate dsp"
        echo "  python scripts/collect_and_render.py --once"
        ;;
        
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Quick reference:"
echo ""
echo "View schedule:"
echo "  python scripts/collect_and_render.py --print-schedule"
echo ""
echo "Single collection:"
echo "  python scripts/collect_and_render.py --once"
echo ""
echo "Continuous (adaptive):"
echo "  python scripts/collect_and_render.py --adaptive"
echo ""
echo "Documentation:"
echo "  DEPLOY.md - Full deployment guide"
echo "  notebooks/RUNBOOK.ipynb - Interactive guide"
echo "========================================="
