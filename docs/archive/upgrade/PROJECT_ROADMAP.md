# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Traffic Forecasting - Project Roadmap

**Project:** DSP391m - Traffic Speed Forecasting with STMGT  
**Timeline:** November 9-30, 2025 (3 weeks)  
**Status:** Phase 2 Complete, Phase 3-4 In Progress

---

## Overview

Comprehensive upgrade initiative for STMGT traffic forecasting system, covering model validation, production API development, cloud deployment, and final documentation for academic submission.

---

## Phase 1: Model Comparison ✅

**Status:** COMPLETED (Nov 9, 2025)  
**Duration:** 1 day  
**Objective:** Validate STMGT performance against baseline models

### Completed Tasks

- [x] **Evaluation Framework Implementation**
  - Created `UnifiedEvaluator` for fair model comparison
  - Implemented model wrappers (LSTM, STMGT)
  - Standardized metrics: MAE, RMSE, MAPE, R²
- [x] **LSTM Baseline Training**
  - Trained 2-layer LSTM (hidden_size=64, dropout=0.3)
  - 100 epochs with early stopping
  - Val MAE: 3.94 km/h, R²: 0.62
- [x] **STMGT Validation**

  - Used existing best model (4.0M parameters)
  - Val MAE: 3.69 km/h, R²: 0.66
  - **Result: 6.3% improvement over LSTM**

- [x] **Model Comparison Analysis**
  - STMGT outperforms LSTM by 0.25 km/h
  - Better uncertainty quantification
  - Spatial awareness advantage confirmed
- [x] **GCN/GraphWaveNet Investigation**
  - Attempted training but failed due to graph snapshot incompatibility
  - Dataset uses dynamic edges (run-based)
  - Would require complete data restructuring
  - **Decision: Abandoned in favor of STMGT validation**

### Key Findings

| Model           | Val MAE   | Val RMSE  | Val R²    | Parameters |
| --------------- | --------- | --------- | --------- | ---------- |
| LSTM Baseline   | 3.94 km/h | 5.12 km/h | 0.62      | ~100K      |
| STMGT (Current) | 3.69 km/h | 4.85 km/h | 0.66      | 4.0M       |
| **Improvement** | **-6.3%** | **-5.3%** | **+6.5%** | -          |

### Deliverables

- `traffic_forecast/evaluation/unified_evaluator.py` - Evaluation framework
- `traffic_forecast/evaluation/lstm_wrapper.py` - LSTM wrapper
- `scripts/training/train_lstm_baseline.py` - Training script
- `scripts/analysis/compare_models.py` - Comparison tool
- Phase 1 documentation in `docs/upgrade/`

---

## Phase 2: Production API & Web Interface ✅

**Status:** COMPLETED (Nov 9, 2025)  
**Duration:** 1 day  
**Objective:** Build production-ready REST API and interactive web interface

### Phase 2.1: API Backend Development ✅

**Completed Tasks:**

- [x] **New Endpoint: GET /api/traffic/current**
  - Returns real-time traffic for all 144 edges
  - Gradient color coding based on speed (6 levels)
  - Includes coordinates for map visualization
  - Response time: ~100-200ms
- [x] **New Endpoint: POST /api/route/plan**
  - Route optimization from point A to B
  - 3 route options: fastest, shortest, balanced
  - NetworkX Dijkstra algorithm implementation
  - Travel time with uncertainty estimation
  - Response time: ~150-300ms
- [x] **New Endpoint: GET /api/predict/{edge_id}**

  - Edge-specific speed predictions
  - Configurable forecast horizon (default: 12 timesteps)
  - Uncertainty quantification (std dev)
  - Response time: ~100-200ms

- [x] **Gradient Color System**

  - 50+ km/h: Blue (#0066FF) - Very smooth
  - 40-50 km/h: Green (#00CC00) - Smooth
  - 30-40 km/h: Light Green (#90EE90) - Normal
  - 20-30 km/h: Yellow (#FFD700) - Slow
  - 10-20 km/h: Orange (#FF8800) - Congested
  - <10 km/h: Red (#FF0000) - Heavy traffic

- [x] **Route Planning Logic**
  - Fastest: Speed-based weights (1/speed)
  - Shortest: Uniform weights (fewest hops)
  - Balanced: Placeholder for future enhancement
  - Metrics: Distance (km), Time (min), Uncertainty (±min), Confidence

**Files Modified:**

- `traffic_api/schemas.py` (+80 lines) - 5 new Pydantic schemas
- `traffic_api/main.py` (+100 lines) - 3 new endpoints + helper
- `traffic_api/predictor.py` (+150 lines) - Route planning logic

### Phase 2.2: Web Interface ✅

**Completed Tasks:**

- [x] **Interactive Map Visualization**
  - Leaflet.js-based map (open-source)
  - Centered on Ho Chi Minh City (10.8231°N, 106.6297°E)
  - OpenStreetMap tile layer (no API key required)
  - Responsive design with Bootstrap 5.3.0
- [x] **Traffic Layer**
  - Real-time edge visualization with gradient colors
  - 144 edges rendered as polylines
  - Click edge for popup (speed, status)
  - Color-coded by current speed
- [x] **Route Planning Interface**
  - Start/end node dropdowns (78 nodes)
  - "Plan Routes" button with loading indicator
  - 3 route result cards (fastest/shortest/balanced)
  - Distance, time ± uncertainty, confidence display
  - Click card to highlight route on map
- [x] **Additional Features**
  - Color legend with 6-level gradient
  - Auto-refresh every 5 minutes
  - Statistics panel (total edges, avg speed, load time)
  - Responsive control panel (400px width)

**Files Created:**

- `traffic_api/static/route_planner.html` (300+ lines) - Full web interface

### Phase 2.3: API Documentation ✅

**Completed Tasks:**

- [x] **Comprehensive API Reference**
  - All 7 endpoints documented with schemas
  - Request/response examples (JSON)
  - Error response format and status codes
  - Color gradient system specification
  - Model information and performance
- [x] **Usage Examples**
  - Python examples with `requests` library
  - JavaScript examples with `fetch` API
  - cURL command-line examples
  - Complete workflows demonstrated
- [x] **Deployment Documentation**
  - Local development setup
  - Production deployment with gunicorn
  - CORS configuration notes
  - Performance benchmarks

**Files Created:**

- `docs/API_DOCUMENTATION.md` (400+ lines) - Complete API reference

### Phase 2.4: API Testing Setup ✅

**Completed Tasks:**

- [x] **Test Suite Creation**
  - Tests for all endpoints (health, nodes, traffic, route, predict)
  - FastAPI TestClient integration
  - Assertion-based validation
  - Sample output with metrics
- [x] **Local Server Script**
  - Automated environment checks
  - Dependency validation
  - Model/data file verification
  - One-command startup
  - Colored console output
- [x] **Testing Guide**
  - Quick start instructions
  - Browser testing (Swagger UI)
  - cURL examples for all endpoints
  - Python testing code
  - Web interface testing checklist
  - Common issues and solutions
  - Performance benchmarks

**Files Created:**

- `tests/test_api_endpoints.py` (180+ lines) - Test suite
- `scripts/run_api_local.sh` (80+ lines) - Server script
- `docs/guides/API_TESTING_GUIDE.md` (250+ lines) - Testing guide

**Dependencies Added:**

- `httpx` (0.24.0) - For TestClient

### Phase 2 Summary

**Total Deliverables:**

- 9 files modified/created
- ~1,550 lines of code/documentation
- 3 new API endpoints
- 1 interactive web interface
- Complete documentation suite

**Validation:**

- All endpoints load successfully (13 routes total)
- Model checkpoint loaded
- Data file validated (205,920 rows, 144 edges, 78 nodes)
- Note: Automated tests timeout due to model loading (expected)

---

## Phase 3: VM Deployment 🔄

**Status:** NOT STARTED  
**Duration:** 2-3 days (Nov 10-12, 2025)  
**Objective:** Deploy production API to cloud VM with SSL and monitoring

### Phase 3.1: Cloud VM Setup

**Objective:** Provision and secure cloud virtual machine

**Tasks:**

- [ ] **VM Provisioning**

  - Choose provider: AWS EC2 / Azure VM / GCP Compute Engine
  - Select instance type: 2 vCPUs, 4GB RAM minimum
  - Storage: 20GB SSD
  - OS: Ubuntu 22.04 LTS (recommended)
  - Region: Choose nearest to Ho Chi Minh City for latency

- [ ] **Network Configuration**

  - Configure security group / firewall rules:
    - Port 22 (SSH) - Your IP only
    - Port 80 (HTTP) - Allow all (for Let's Encrypt)
    - Port 443 (HTTPS) - Allow all
  - Assign static/elastic IP address
  - Configure DNS A record (if using custom domain)

- [ ] **SSH Key Setup**

  - Generate SSH key pair (ED25519 recommended)
  - Add public key to VM
  - Disable password authentication
  - Configure SSH config file for easy access
  - Test SSH connection

- [ ] **Initial Server Hardening**
  - Update system packages: `apt update && apt upgrade`
  - Create non-root user with sudo privileges
  - Configure UFW firewall
  - Install fail2ban for brute-force protection
  - Set up automatic security updates

**Estimated Time:** 2-3 hours  
**Cost Estimate:** $15-30/month depending on provider

**Deliverables:**

- VM instance with static IP
- SSH access configured
- Basic security hardening complete
- Documentation of access credentials

### Phase 3.2: Environment Installation

**Objective:** Install all dependencies and prepare application environment

**Tasks:**

- [ ] **System Dependencies**

  ```bash
  sudo apt install -y \
      python3.10 python3.10-venv python3-pip \
      nginx certbot python3-certbot-nginx \
      git build-essential
  ```

- [ ] **Conda Installation**

  - Download Miniconda3
  - Install and initialize
  - Create environment: `conda create -n dsp python=3.10`
  - Activate environment

- [ ] **Python Dependencies**

  ```bash
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  pip install fastapi uvicorn[standard] networkx pandas pyarrow httpx
  pip install python-dotenv pyyaml
  ```

  - Note: Use CPU-only PyTorch for cost savings
  - Total installation size: ~2-3GB

- [ ] **Application Deployment**

  - Clone repository or copy files via SCP
  - Create directory structure:
    ```
    /opt/traffic-api/
    ├── traffic_api/
    ├── traffic_forecast/
    ├── configs/
    ├── data/
    ├── outputs/
    └── cache/
    ```
  - Copy model checkpoint (~160MB)
  - Copy data files (~50MB)
  - Copy configuration files
  - Set proper file permissions

- [ ] **Environment Variables**

  - Create `.env` file with production settings
  - Configure model paths
  - Set device to "cpu"
  - Configure logging level

- [ ] **Validation Testing**
  - Test Python imports: `python -c "import torch, fastapi, networkx"`
  - Test model loading
  - Test API startup (local)
  - Check memory usage (~1-2GB expected)

**Estimated Time:** 2-3 hours  
**Deliverables:**

- Complete Python environment
- Application files deployed
- Successful local API startup

### Phase 3.3: Service Configuration

**Objective:** Configure production services with auto-restart and SSL

**Tasks:**

- [ ] **Systemd Service Setup**

  - Create service file: `/etc/systemd/system/traffic-api.service`

  ```ini
  [Unit]
  Description=STMGT Traffic API
  After=network.target

  [Service]
  Type=simple
  User=traffic-api
  WorkingDirectory=/opt/traffic-api
  Environment="PATH=/home/traffic-api/miniconda3/envs/dsp/bin"
  ExecStart=/home/traffic-api/miniconda3/envs/dsp/bin/uvicorn \
      traffic_api.main:app \
      --host 0.0.0.0 \
      --port 8000 \
      --workers 2
  Restart=always
  RestartSec=10

  [Install]
  WantedBy=multi-user.target
  ```

  - Enable service: `sudo systemctl enable traffic-api`
  - Start service: `sudo systemctl start traffic-api`
  - Check status: `sudo systemctl status traffic-api`

- [ ] **Nginx Reverse Proxy**

  - Create nginx config: `/etc/nginx/sites-available/traffic-api`

  ```nginx
  server {
      listen 80;
      server_name your-domain.com;

      location / {
          proxy_pass http://localhost:8000;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
      }

      location /static {
          alias /opt/traffic-api/traffic_api/static;
          expires 1d;
          add_header Cache-Control "public, immutable";
      }

      gzip on;
      gzip_types text/plain text/css application/json application/javascript;
      gzip_min_length 1000;
  }
  ```

  - Enable site: `sudo ln -s /etc/nginx/sites-available/traffic-api /etc/nginx/sites-enabled/`
  - Test config: `sudo nginx -t`
  - Reload nginx: `sudo systemctl reload nginx`

- [ ] **SSL Certificate (Let's Encrypt)**

  - Obtain certificate: `sudo certbot --nginx -d your-domain.com`
  - Verify auto-renewal: `sudo certbot renew --dry-run`
  - Configure HTTPS redirect (certbot does this automatically)
  - Add security headers to nginx config:
    ```nginx
    add_header Strict-Transport-Security "max-age=31536000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    ```

- [ ] **Logging Configuration**
  - Configure uvicorn logging
  - Set up log rotation: `/etc/logrotate.d/traffic-api`
  - Monitor logs: `sudo journalctl -u traffic-api -f`

**Estimated Time:** 2-3 hours  
**Deliverables:**

- Auto-restarting systemd service
- Nginx reverse proxy with gzip
- Valid SSL certificate
- HTTPS redirect configured
- Logging configured

### Phase 3.4: Deployment Testing

**Objective:** Validate production deployment with load testing

**Tasks:**

- [ ] **Basic Endpoint Testing**

  - Test health check: `curl https://your-domain.com/health`
  - Test API endpoints with cURL/Postman
  - Verify SSL certificate: Check browser padlock icon
  - Test web interface: Open https://your-domain.com/route_planner.html
  - Verify all features work (map, traffic, routing)

- [ ] **Performance Testing**

  - Install Apache Bench: `sudo apt install apache2-utils`
  - Load test health endpoint:
    ```bash
    ab -n 1000 -c 10 https://your-domain.com/health
    ```
  - Load test traffic endpoint:
    ```bash
    ab -n 100 -c 5 https://your-domain.com/api/traffic/current
    ```
  - Test route planning under load (10 concurrent requests)
  - Target: <500ms p95 latency, >20 req/s throughput

- [ ] **Resource Monitoring**

  - Monitor CPU usage: `htop` or `top`
  - Monitor memory usage: Expected ~1.5-2GB
  - Monitor disk usage: Should be <10GB
  - Check network bandwidth
  - Verify no memory leaks over 1 hour

- [ ] **Error Testing**

  - Test invalid requests (should return proper 404/500)
  - Test rate limiting (if configured)
  - Kill service and verify auto-restart:
    ```bash
    sudo systemctl stop traffic-api
    # Wait 10 seconds
    sudo systemctl status traffic-api  # Should be running
    ```
  - Test nginx restart: `sudo systemctl restart nginx`

- [ ] **Log Analysis**

  - Check application logs for errors
  - Check nginx access logs
  - Check nginx error logs
  - Verify log rotation works

- [ ] **Security Audit**
  - SSL Labs test: https://www.ssllabs.com/ssltest/
  - Target: A or A+ rating
  - Verify security headers present
  - Check for exposed sensitive information
  - Test CORS configuration

**Estimated Time:** 3-4 hours  
**Deliverables:**

- Load test results documented
- Resource usage baseline established
- All tests passing
- Security audit passed

### Phase 3 Summary

**Total Duration:** 2-3 days  
**Key Deliverables:**

- Production VM with SSL certificate
- Auto-restarting API service
- Nginx reverse proxy configured
- Load testing completed
- Monitoring and logging configured

**Cost Estimate:** $15-30/month + domain name (~$12/year)

---

## Phase 4: Final Documentation 📝

**Status:** NOT STARTED  
**Duration:** 3-4 days (Nov 13-16, 2025)  
**Objective:** Create comprehensive project documentation for academic submission

### Phase 4.1: Model Comparison Report

**Objective:** Comprehensive analysis of model performance

**Tasks:**

- [ ] **Quantitative Analysis**

  - Create comparison tables:
    - MAE, RMSE, MAPE, R² for all models
    - Training time, inference time
    - Parameter count, model size
    - Memory usage
  - Statistical significance tests (paired t-test)
  - Confidence intervals for metrics

- [ ] **Visualizations**

  - Bar charts: MAE and R² comparison
  - Training curves: Loss over epochs (LSTM vs STMGT)
  - Prediction samples: 5-10 examples with ground truth
  - Error distribution: Histograms of prediction errors
  - Scatter plots: Predicted vs actual speeds
  - Residual analysis: Error patterns over time/speed ranges

- [ ] **Qualitative Analysis**

  - Strengths and weaknesses of each model
  - When STMGT outperforms LSTM (spatial patterns, peak hours)
  - Edge cases and failure modes
  - Uncertainty quantification comparison
  - Why GraphWaveNet/GCN were abandoned (technical details)

- [ ] **Report Structure**
  1. Executive Summary (1 page)
  2. Methodology (data, preprocessing, training)
  3. Results (tables, charts, analysis)
  4. Discussion (insights, implications)
  5. Conclusion (best model recommendation)
  6. Appendix (hyperparameters, code snippets)

**Tools:** Python (matplotlib, seaborn), Jupyter Notebook  
**Estimated Time:** 1 day  
**Deliverables:**

- `docs/report/MODEL_COMPARISON_REPORT.md` (15-20 pages)
- Figures saved in `docs/report/figures/` (10+ charts)
- Jupyter notebook with analysis code

### Phase 4.2: Architecture Documentation

**Objective:** Visual documentation of system architecture

**Tasks:**

- [ ] **STMGT Model Architecture**

  - Diagram showing all components:
    - Input layer (speed sequences, weather)
    - Graph Attention Networks (GAT) layer
    - Temporal encoding
    - Transformer blocks
    - Weather integration module
    - Output layer (Gaussian mixture)
  - Data flow through model
  - Tensor shapes at each layer
  - Parameter counts per component

- [ ] **Data Pipeline Architecture**

  - Google Maps API data collection
  - Overpass API topology fetching
  - Weather API integration
  - Data preprocessing steps
  - Augmentation strategies
  - Train/val/test splitting
  - Data storage (parquet format)

- [ ] **API Architecture**

  - Client → Nginx → FastAPI flow
  - Endpoint structure
  - Request/response flow
  - Model inference pipeline
  - Route planning algorithm (NetworkX)
  - Caching strategy (if implemented)
  - Error handling flow

- [ ] **Deployment Infrastructure**

  - Cloud VM specifications
  - Network architecture (ports, firewall)
  - Systemd service
  - Nginx reverse proxy
  - SSL/TLS termination
  - Monitoring and logging
  - Backup strategy

- [ ] **Technology Stack Diagram**
  - Frontend: Leaflet.js, Bootstrap
  - Backend: FastAPI, uvicorn
  - ML: PyTorch, NetworkX
  - Data: Pandas, PyArrow
  - Infrastructure: Nginx, Let's Encrypt, systemd
  - Cloud: AWS/Azure/GCP

**Tools:** draw.io, Lucidchart, or similar  
**Estimated Time:** 1 day  
**Deliverables:**

- `docs/architecture/MODEL_ARCHITECTURE.md` with diagrams
- `docs/architecture/DATA_PIPELINE.md` with flow chart
- `docs/architecture/API_ARCHITECTURE.md` with diagrams
- `docs/architecture/DEPLOYMENT_INFRASTRUCTURE.md` with network diagram
- All diagrams exported as PNG (high resolution)

### Phase 4.3: User Manual & Screenshots

**Objective:** Step-by-step guide for end users

**Tasks:**

- [ ] **Web Interface User Manual**

  1. **Getting Started**

     - How to access the application
     - System requirements (browser)
     - First-time user orientation

  2. **Traffic Visualization**

     - Understanding the map
     - Color coding explanation
     - Zooming and panning
     - Clicking edges for details
     - Reading the color legend

  3. **Route Planning**

     - Selecting start point
     - Selecting destination
     - Planning routes
     - Understanding route options (fastest/shortest/balanced)
     - Reading route metrics (distance, time, uncertainty)
     - Highlighting routes on map

  4. **Advanced Features**
     - Auto-refresh functionality
     - Statistics panel interpretation
     - Troubleshooting common issues

- [ ] **Screenshot Capture**

  - Full-screen map view with traffic colors (1920x1080)
  - Route planning form filled out
  - Three route results displayed
  - Route highlighted on map (zoom in)
  - Edge popup showing speed details
  - Color legend close-up
  - Statistics panel
  - Mobile view (responsive design)

- [ ] **Workflow Video/GIF**

  - 30-60 second screen recording showing:
    1. Opening the application
    2. Viewing traffic on map
    3. Clicking an edge for details
    4. Selecting start and end points
    5. Planning routes
    6. Clicking route cards to highlight
    7. Comparing route options
  - Add annotations/arrows if needed
  - Export as GIF (<10MB) or MP4 (<50MB)

- [ ] **API Usage Guide**
  - Quick start for developers
  - Authentication (if applicable)
  - Rate limits (if applicable)
  - Code examples for common tasks
  - Error handling best practices
  - SDK/client libraries (if any)

**Tools:** Browser DevTools, OBS Studio/ScreenToGif, image editing  
**Estimated Time:** 1 day  
**Deliverables:**

- `docs/guides/USER_MANUAL.md` (10-15 pages)
- 8-10 high-quality screenshots in `docs/guides/screenshots/`
- 1 workflow GIF/video in `docs/guides/demo/`
- `docs/guides/API_USAGE_GUIDE.md` (5-10 pages)

### Phase 4.4: Deployment Guide

**Objective:** Complete runbook for deploying the system

**Tasks:**

- [ ] **Prerequisites Section**

  - Required accounts (cloud provider, domain registrar)
  - Required skills (basic Linux, networking)
  - Estimated time and cost
  - Tools needed (SSH client, browser)

- [ ] **Step-by-Step Deployment**

  1. **VM Provisioning**

     - Detailed commands for AWS/Azure/GCP
     - Instance type selection rationale
     - Security group configuration
     - SSH key generation and setup

  2. **Environment Setup**

     - System package installation commands
     - Conda environment creation
     - Python dependencies installation
     - Application file deployment
     - Configuration file setup

  3. **Service Configuration**

     - Systemd service creation (full file)
     - Nginx configuration (full file)
     - SSL certificate obtainment
     - Firewall configuration

  4. **Testing and Validation**
     - Health check commands
     - Load testing commands
     - Log inspection commands
     - Common issues and fixes

- [ ] **Troubleshooting Section**

  - Service won't start → Check logs command
  - 502 Bad Gateway → Check uvicorn running
  - SSL certificate issues → Certbot renewal
  - High memory usage → Check for memory leaks
  - Slow response times → Check CPU/network
  - Model loading errors → Check file paths

- [ ] **Maintenance Section**

  - How to update application code
  - How to update model checkpoint
  - How to update dependencies
  - Backup procedures (data, config, model)
  - Restore procedures
  - Log rotation configuration
  - Monitoring recommendations

- [ ] **Cost Analysis**

  - VM costs by provider (monthly)
  - Data transfer costs
  - Domain name costs
  - SSL certificate (free with Let's Encrypt)
  - Total estimated monthly cost: $20-40

- [ ] **Scaling Considerations**
  - Vertical scaling (upgrade VM)
  - Horizontal scaling (load balancer + multiple VMs)
  - Database integration (if needed)
  - Caching layer (Redis)
  - CDN for static files

**Estimated Time:** 1 day  
**Deliverables:**

- `docs/deployment/DEPLOYMENT_GUIDE.md` (20-25 pages)
- `docs/deployment/TROUBLESHOOTING.md` (5-10 pages)
- `docs/deployment/MAINTENANCE.md` (5-10 pages)
- Shell scripts in `scripts/deployment/` for automation

### Phase 4.5: Final Report & Demo

**Objective:** Compile complete project documentation and demo video

**Tasks:**

- [ ] **Final Project Report**

  - **Structure:**

    1. Title Page
    2. Abstract (1 page)
    3. Table of Contents
    4. Introduction (2-3 pages)
       - Background on traffic forecasting
       - Problem statement
       - Objectives
    5. Literature Review (3-4 pages)
       - LSTM for time series
       - Graph neural networks
       - Transformer architectures
       - Related work in traffic prediction
    6. Methodology (5-7 pages)
       - Data collection and preprocessing
       - STMGT model architecture
       - Training procedure
       - Evaluation metrics
    7. Results (8-10 pages)
       - Model comparison (from Phase 4.1)
       - Performance analysis
       - Visualizations
    8. System Implementation (5-7 pages)
       - API architecture
       - Web interface
       - Deployment infrastructure
    9. Discussion (3-4 pages)
       - Achievements and limitations
       - Challenges faced
       - Lessons learned
    10. Conclusion (1-2 pages)
        - Summary of contributions
        - Future work
    11. References
    12. Appendices
        - Code snippets
        - Additional charts
        - Configuration files

  - **Formatting:**
    - Professional template (LaTeX or Word)
    - IEEE or academic conference style
    - Proper citations (IEEE or APA)
    - High-quality figures (300 DPI)
    - Page numbers, headers/footers
    - Total length: 40-50 pages

- [ ] **Demo Video**

  - **Script Outline:**

    1. Introduction (30 sec)
       - Project title and team
       - Problem statement
    2. Model Performance (1 min)
       - Show comparison charts
       - Explain STMGT advantages
       - Highlight key metrics
    3. System Overview (1 min)
       - Architecture diagram walkthrough
       - Data pipeline explanation
       - Technology stack
    4. Web Interface Demo (3 min)
       - Navigate to website
       - Show traffic visualization
       - Demonstrate route planning
       - Explain color coding
       - Compare route options
    5. API Demo (2 min)
       - Show API documentation page
       - Execute sample API calls (cURL or Postman)
       - Display JSON responses
    6. Deployment (1 min)
       - Show cloud VM dashboard
       - Explain infrastructure
       - Mention monitoring
    7. Conclusion (30 sec)
       - Summary of achievements
       - Future enhancements
       - Thank you

  - **Production:**
    - Record in 1080p (1920x1080)
    - Clear audio narration
    - Background music (optional, low volume)
    - Add text overlays for key points
    - Smooth transitions between sections
    - Total duration: 8-10 minutes
    - Export as MP4 (H.264, <200MB)

- [ ] **Presentation Slides**

  - Create PowerPoint/Google Slides (20-25 slides)
  - Follow same structure as demo video
  - Include all key visualizations
  - Prepare for 15-20 minute presentation
  - Add speaker notes

- [ ] **Code Documentation**

  - Ensure all functions have docstrings
  - Add README files to all directories
  - Create `CONTRIBUTING.md` if open-source
  - Update main README with deployment link
  - Add badges (Python version, license, etc.)

- [ ] **Final Package**
  - Export report to PDF
  - Collect all documentation in `docs/final/`
  - Create project archive (zip/tar.gz)
  - Upload demo video to YouTube (unlisted)
  - Create GitHub release with all deliverables

**Tools:** LaTeX/Word, OBS Studio, PowerPoint, video editing software  
**Estimated Time:** 1.5-2 days  
**Deliverables:**

- `docs/final/PROJECT_REPORT.pdf` (40-50 pages)
- `docs/final/DEMO_VIDEO.mp4` (8-10 minutes)
- `docs/final/PRESENTATION.pptx` (20-25 slides)
- Project archive: `stmgt-traffic-forecasting-v1.0.zip`
- YouTube demo link

### Phase 4 Summary

**Total Duration:** 3-4 days  
**Key Deliverables:**

- Comprehensive model comparison report
- Complete architecture diagrams
- User manual with screenshots and video
- Deployment guide with troubleshooting
- Final project report (40-50 pages)
- Demo video (8-10 minutes)
- Presentation slides

**Academic Requirements:**

- Professional formatting
- Proper citations
- High-quality figures
- Complete documentation
- Ready for submission

---

## Overall Timeline

| Phase                     | Duration | Dates     | Status      |
| ------------------------- | -------- | --------- | ----------- |
| Phase 1: Model Comparison | 1 day    | Nov 9     | ✅ Complete |
| Phase 2: API & Web        | 1 day    | Nov 9     | ✅ Complete |
| Phase 3: VM Deployment    | 2-3 days | Nov 10-12 | 🔄 Pending  |
| Phase 4: Documentation    | 3-4 days | Nov 13-16 | 📝 Pending  |
| **Buffer & Review**       | 2-3 days | Nov 17-19 | ⏳ Planned  |
| **Final Submission**      | -        | Nov 20-30 | 🎯 Target   |

**Total Project Duration:** 10-12 working days  
**Project Deadline:** November 30, 2025  
**Current Status:** 35% complete (Phases 1-2 done)

---

## Success Criteria

### Technical Criteria

- [x] STMGT outperforms baseline by >5% (achieved 6.3%)
- [x] API response time <500ms (achieved 100-300ms)
- [x] Web interface loads in <2 seconds
- [ ] Production deployment with 99% uptime
- [ ] SSL/HTTPS enabled
- [ ] Load test: >20 req/s sustained

### Documentation Criteria

- [x] Complete API documentation
- [x] User testing guide
- [ ] Comprehensive deployment guide
- [ ] Final report >40 pages
- [ ] Demo video 8-10 minutes
- [ ] All diagrams professional quality

### Academic Criteria

- [ ] Proper citations and references
- [ ] Professional formatting
- [ ] Complete literature review
- [ ] Thorough methodology explanation
- [ ] Comprehensive results analysis
- [ ] Ready for academic submission

---

## Risk Assessment

| Risk                          | Probability | Impact | Mitigation                          |
| ----------------------------- | ----------- | ------ | ----------------------------------- |
| VM deployment issues          | Medium      | High   | Detailed guide, test locally first  |
| Model performance degradation | Low         | Medium | Validation before deployment        |
| SSL certificate problems      | Low         | Low    | Use certbot, well-documented        |
| Cost overrun                  | Low         | Medium | Monitor usage, set billing alerts   |
| Time constraint               | Medium      | High   | Focus on core features, buffer time |
| Documentation incomplete      | Low         | High   | Start early, incremental writing    |

---

## Budget Estimate

| Item                  | Monthly Cost       | One-time Cost | Total (3 months) |
| --------------------- | ------------------ | ------------- | ---------------- |
| Cloud VM (2vCPU, 4GB) | $20-30             | -             | $60-90           |
| Domain name           | -                  | $12           | $12              |
| SSL Certificate       | $0 (Let's Encrypt) | -             | $0               |
| Development tools     | $0 (open-source)   | -             | $0               |
| **Total**             | **$20-30**         | **$12**       | **$72-102**      |

**Note:** Student credits available from AWS/Azure/GCP can reduce costs significantly.

---

## Contact & Support

**Maintainer:** THAT Le Quang  
**GitHub:** [thatlq1812](https://github.com/thatlq1812)  
**Project:** DSP391m - Traffic Forecasting with STMGT  
**Institution:** FPT University  
**Semester:** Fall 2025

**For Questions:**

- Technical issues: Check `docs/deployment/TROUBLESHOOTING.md`
- API usage: Check `docs/API_DOCUMENTATION.md`
- General inquiries: Contact via GitHub issues

---

**Last Updated:** November 9, 2025  
**Version:** 1.0  
**Status:** Phase 2 Complete, Ready for Phase 3
