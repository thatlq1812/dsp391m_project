# Traffic Forecast Node-Radius v2.0

A real-time traffic forecasting system for Ho Chi Minh City using node-radius graph modeling, weather integration, and machine learning.

## Table of Contents

- [English](#english)
  - [Overview](#overview)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [API Documentation](#api-documentation)
  - [Project Structure](#project-structure)
  - [Deployment on Google Cloud](#deployment-on-google-cloud)
  - [Contributing](#contributing)
  - [License](#license)
- [Tiếng Việt](#tiếng-việt)

---

## English

### Overview

This project implements a v2.0 traffic forecasting system for Ho Chi Minh City (HCMC) using:

- **Node-Radius Graph**: Spatial modeling of traffic nodes and their relationships
- **Real Data Collectors**: Open-Meteo (weather), Google Directions (traffic), Overpass (OSM nodes)
- **ML Pipeline**: Feature engineering, model training (Linear Regression/LSTM), batch inference
- **FastAPI Service**: REST API for real-time predictions
- **Scheduler**: Automated periodic data collection and processing
- **Visualization**: Interactive maps and heatmaps

### Features

- Real-time Traffic Data: Google Directions API integration
- Weather Integration: Open-Meteo forecasts (temp, rain, wind)
- Spatial Modeling: Node-radius graph with k-nearest neighbors
- Machine Learning: Linear Regression baseline + LSTM for time-series
- FastAPI Backend: RESTful API with automatic docs
- Visualization: Matplotlib-based maps and heatmaps
- Automated Pipelines: APScheduler for periodic tasks
- Container Ready: Docker support for easy deployment

### Requirements

- **Python**: 3.8+
- **System**: Linux/Windows/macOS
# Traffic Forecast Node-Radius v2.0

Light, practical README for running the project locally and on a Google Cloud VM.

## Quick summary
- Project: traffic forecasting for Ho Chi Minh City using node-radius graphs, weather, and ML.
- Collectors support area selection modes: `bbox` and `point_radius` (also called `circle`).
- Priority for area selection: CLI args > environment variables > `configs/project_config.yaml`.

This README shows how to run collectors with CLI options and how to transfer & run the project on a GCP VM.

----------------------

## Minimum requirements
- Python 3.8+
- git
- 8GB RAM recommended

## Files you should know
- `configs/project_config.yaml` — central configuration (collectors area, params)
- `collectors/` — open_meteo, google, overpass collectors (support CLI overrides)
- `collectors/area_utils.py` — shared helper that resolves `mode` → bbox
- `run_collectors.py` — helper to run multiple collectors (keeps previous behavior)
- `apps/api/` — FastAPI app
- `apps/scheduler/` — scheduler entrypoint

----------------------

## Running collectors locally (recommended flow)

1) Create a Python virtual environment and install deps:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) Examples of collector CLI usage

- Run Overpass with bbox (min_lat,min_lon,max_lat,max_lon):

```bash
python collectors/overpass/collector.py --bbox "10.67,106.60,10.90,106.84"
```

- Run Overpass with a point + radius (meters):

```bash
python collectors/overpass/collector.py --mode point_radius --center "106.7,10.7" --radius 5000
```

- Run Open-Meteo for a point-radius area (filters nodes by bbox computed from center/radius):

```bash
python collectors/open_meteo/collector.py --mode point_radius --center "106.7,10.7" --radius 5000
```

- Run Google collector inside a smaller bbox:

```bash
python collectors/google/collector.py --bbox "10.75,106.60,10.85,106.75"
```

Notes:
- CLI args override environment variables and values in `configs/project_config.yaml`.
- BBOX format used by the collectors and Overpass is `[min_lat, min_lon, max_lat, max_lon]`.
- `point_radius` is converted internally to a bbox using a spherical approximation (sufficient for tens of km). For high accuracy use `pyproj`.

----------------------

## Typical local workflow

1. Collect raw nodes (Overpass):

```bash
python collectors/overpass/collector.py --bbox "10.67,106.60,10.90,106.84"
```

2. Collect weather (Open-Meteo) and map to nodes:

```bash
python collectors/open_meteo/collector.py --bbox "10.67,106.60,10.90,106.84"
```

3. Collect traffic edges (Google):

```bash
python collectors/google/collector.py --bbox "10.67,106.60,10.90,106.84"
```

4. Normalize & feature build:

```bash
python pipelines/normalize/normalize.py
python pipelines/features/build_features.py
```

5. Train model:

```bash
python pipelines/model/train.py
```

6. Start API and Scheduler (two terminals):

```bash
uvicorn apps.api.main:app --reload --port 8000
python apps/scheduler/main.py
```

----------------------

## Transfer project to a Google Cloud VM and run (two recommended methods)

You can either push the repo to a remote (GitHub/GitLab) and clone on the VM, or directly copy files using `gcloud compute scp`.

Prerequisites (local machine):
- Install `gcloud` SDK and authenticate: `gcloud init`
- (Optional) Create a GitHub remote and push your branch

Method A — Recommended: push to remote and clone on VM

1. Create a Git remote (GitHub) and push your branch. Example (one-time):

```bash
   # Check logs
   sudo journalctl -u traffic-api -n 50

```

2. Create a VM on GCP (example):

```bash
   # Test locally
   curl http://localhost:8000/
   ```

2. **Database connection issues**:
   ```bash
   # Check PostgreSQL status
```

3. SSH to VM and clone the repo:

```bash
   sudo systemctl status postgresql

   # Test connection
   psql -h localhost -U traffic_user -d traffic_forecast
   ```

3. **Out of memory errors**:
   ```bash
```

4. Set environment variables (example using API key):

```bash
   # Monitor memory usage
   free -h

   # Increase swap space if needed

5. Run collectors (example point_radius):

```bash
   sudo fallocate -l 1G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile

Method B — Copy files directly using `gcloud compute scp` (no remote required)

1. From your local machine (ensure gcloud is authenticated):

```bash
   ```

4. **SSL certificate issues**:

2. SSH to VM and run (on VM):

```bash
   ```bash
   # Check certificate validity
   openssl x509 -in /etc/nginx/ssl/cert.pem -text -noout

   # Renew Let's Encrypt certificate
   sudo certbot renew
   sudo systemctl reload nginx

----------------------

## Notes and production tips
- For containers: build Docker images and run with Docker Compose (see `infra/`).
- For scheduled runs: use Cloud Scheduler + Cloud Functions or run `apps/scheduler/main.py` under a process manager (systemd / supervisord).
- For large `point_radius` areas or global coverage, replace spherical approx with `pyproj`/`geographiclib`.
- Keep secrets out of `configs/` — use environment variables or GCP Secret Manager.

----------------------

If you want, I can:
- Add a short `docs/deploy_gcp.md` with screenshots and exact IAM roles.
- Create a small systemd unit file and an example `docker-compose` for production runs.

---
Updated: October 08, 2025
   ```

### Scaling Strategies

#### Horizontal Scaling

```bash
# Load balancer with multiple API instances
# Use nginx upstream for multiple backend servers

upstream traffic_api {
    server api1.example.com:8000;
    server api2.example.com:8000;
    server api3.example.com:8000;
}
```

#### Vertical Scaling

```bash
# Increase VM resources
gcloud compute instances set-machine-type traffic-forecast-vm \
    --machine-type n1-standard-8 --zone asia-southeast1-a

# Optimize Python application
# Use async/await for I/O operations
# Implement caching (Redis)
# Database query optimization
```

#### Auto-scaling

```bash
# GCP auto-scaling based on CPU utilization
gcloud compute instance-groups managed set-autoscaling traffic-forecast-group \
    --max-num-replicas 10 \
    --min-num-replicas 2 \
    --target-cpu-utilization 0.6 \
    --cool-down-period 60
```

This comprehensive deployment guide covers everything from initial setup to production monitoring and scaling strategies.

### Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Create Pull Request

### License

MIT License - see [LICENSE](LICENSE) for details.

---

## Tiếng Việt

### Tổng quan

Dự án này triển khai hệ thống dự báo giao thông thời gian thực phiên bản 2.0 cho Thành phố Hồ Chí Minh sử dụng:

- **Đồ thị Node-Radius**: Mô hình hóa không gian các nút giao thông và mối quan hệ
- **Thu thập dữ liệu thực**: Open-Meteo (thời tiết), Google Directions (giao thông), Overpass (nút OSM)
- **Pipeline ML**: Kỹ thuật đặc trưng, huấn luyện mô hình (Hồi quy tuyến tính/LSTM), suy luận hàng loạt
- **Dịch vụ FastAPI**: API REST cho dự đoán thời gian thực
- **Lập lịch**: Tự động thu thập và xử lý dữ liệu định kỳ
- **Trực quan hóa**: Bản đồ và heatmap tương tác

### Tính năng

- Dữ liệu giao thông thời gian thực: Tích hợp Google Directions API
- Tích hợp thời tiết: Dự báo Open-Meteo (nhiệt độ, mưa, gió)
- Mô hình hóa không gian: Đồ thị node-radius với k láng giềng gần nhất
- Học máy: Hồi quy tuyến tính cơ bản + LSTM cho chuỗi thời gian
- Backend FastAPI: API REST với tài liệu tự động
- Trực quan hóa: Bản đồ và heatmap dựa trên Matplotlib
- Pipeline tự động: APScheduler cho tác vụ định kỳ
- Sẵn sàng container: Hỗ trợ Docker để triển khai dễ dàng

### Yêu cầu

- **Python**: 3.8+
- **Hệ thống**: Linux/Windows/macOS
- **Bộ nhớ**: 8GB+ RAM khuyến nghị
- **Lưu trữ**: 10GB+ cho dữ liệu và mô hình
- **API**: Khóa Google Maps API, Open-Meteo (miễn phí)

### Cài đặt

1. **Clone Repository**:
   ```bash
   git clone <repository-url>
   cd traffic-forecast-node-radius
   ```

2. **Tạo Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Trên Windows: venv\Scripts\activate
   ```

3. **Cài đặt Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Thiết lập Environment**:
   ```bash
   cp .env_template .env
   # Chỉnh sửa .env với API key của bạn
   ```

### Cấu hình

Tất cả cấu hình được tập trung trong `configs/project_config.yaml`:

```yaml
project:
  name: traffic-forecast-node-radius
  version: v2.0
  bbox: [10.67, 106.60, 10.90, 106.84]  # Ranh giới HCMC

collectors:
  google_directions:
    api_key_env: GOOGLE_MAPS_API_KEY
    k_neighbors: 3
    radius_km: 0.8

pipelines:
  model:
    type: linear_regression  # hoặc lstm
    test_size: 0.2
```

### Sử dụng

#### Khởi động nhanh

1. **Thu thập dữ liệu ban đầu**:
   ```bash
   python run_collectors.py  # Chạy tất cả collectors
   ```

2. **Xử lý dữ liệu**:
   ```bash
   python pipelines/normalize/normalize.py
   python pipelines/features/build_features.py
   ```

3. **Huấn luyện mô hình**:
   ```bash
   python pipelines/model/train.py
   ```

4. **Khởi động dịch vụ**:
   ```bash
   # API (terminal 1)
   uvicorn apps.api.main:app --reload --port 8000

   # Scheduler (terminal 2)
   python apps/scheduler/main.py
   ```

5. **Trực quan hóa kết quả**:
   ```bash
   python visualize.py
   ```

#### Sử dụng API

```bash
# Lấy dự đoán cho nút
curl "http://localhost:8000/v1/nodes/node_123/forecast?horizon=15"

# Phản hồi
{
  "node_id": "node_123",
  "horizon_min": 15,
  "speed_kmh_pred": 41.08,
  "congestion_level": 2
}
```

### Tài liệu API

- **Base URL**: `http://localhost:8000`
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

#### Endpoints

| Phương thức | Endpoint | Mô tả |
|-------------|----------|--------|
| GET | `/` | Kiểm tra sức khỏe |
| GET | `/v1/nodes/{node_id}/forecast` | Lấy dự báo tốc độ |

### Cấu trúc dự án

```
├── apps/                    # Ứng dụng
│   ├── api/                # Dịch vụ FastAPI
│   └── scheduler/          # Jobs APScheduler
├── collectors/             # Thu thập dữ liệu
│   ├── google/            # Google Directions
│   ├── open_meteo/        # Dữ liệu thời tiết
│   └── overpass/          # Nút OSM
├── configs/               # File cấu hình
├── data/                  # Lưu trữ dữ liệu
├── doc/                   # Tài liệu
├── infra/                 # Cơ sở hạ tầng (Docker, etc.)
├── models/                # Mô hình đã huấn luyện
├── pipelines/             # Xử lý dữ liệu
│   ├── features/         # Kỹ thuật đặc trưng
│   ├── model/            # Huấn luyện/suy luận ML
│   └── normalize/        # Chuẩn hóa dữ liệu
├── tests/                 # Unit tests
├── .env_template         # Template environment
├── requirements.txt      # Dependencies Python
├── visualize.py          # Script trực quan hóa
└── README.md
```

### Triển khai trên Google Cloud

#### Tổng quan kiến trúc

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GCP VM        │    │   Cloud Storage │    │   BigQuery      │
│   (Compute)     │◄──►│   (Data Lake)   │◄──►│   (Data Warehouse│
│                 │    │                 │    │                 │
│ • Collectors    │    │ • Raw data      │    │ • Analytics     │
│ • Training      │    │ • Models        │    │ • Reports       │
│ • API Service   │    │ • Configs       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   Cloud Run     │
                    │   (API)         │
                    └─────────────────┘
```

#### Thiết lập VM (Compute Engine)

1. **Tạo VM Instance**:
   ```bash
   gcloud compute instances create traffic-forecast-vm \
     --zone=asia-southeast1-a \
     --machine-type=n1-standard-4 \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=50GB \
     --scopes=https://www.googleapis.com/auth/cloud-platform
   ```

2. **Cài đặt Dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip git
   git clone <repo>
   cd traffic-forecast-node-radius
   pip install -r requirements.txt
   ```

3. **Cấu hình Service Account**:
   ```bash
   # Tạo service account
   gcloud iam service-accounts create traffic-forecast-sa \
     --description="Service account for traffic forecast" \
     --display-name="Traffic Forecast SA"

   # Cấp quyền
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:traffic-forecast-sa@PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.admin"

   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:traffic-forecast-sa@PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/bigquery.admin"
   ```

4. **Thiết lập Cloud Storage**:
   ```bash
   # Tạo bucket
   gsutil mb -p PROJECT_ID -c regional -l asia-southeast1 gs://traffic-forecast-data

   # Upload config
   gsutil cp configs/project_config.yaml gs://traffic-forecast-data/config/
   ```

5. **Chạy Job Training**:
   ```bash
   # Download dữ liệu mới nhất
   gsutil -m cp -r gs://traffic-forecast-data/data ./data

   # Train model
   python pipelines/model/train.py

   # Upload model
   gsutil cp models/*.pkl gs://traffic-forecast-data/models/
   ```

#### Jobs theo lịch với Cloud Scheduler

1. **Tạo Cloud Function cho Thu thập dữ liệu**:
   ```python
   # functions/main.py
   def collect_data(request):
       import subprocess
       result = subprocess.run(['python', 'run_collectors.py'])
       return f'Collection completed with code {result.returncode}'
   ```

2. **Deploy Function**:
   ```bash
   gcloud functions deploy collect-data \
     --runtime python39 \
     --trigger-http \
     --allow-unauthenticated \
     --source functions/
   ```

3. **Lên lịch với Cloud Scheduler**:
   ```bash
   gcloud scheduler jobs create http collect-daily \
     --schedule="0 2 * * *" \
     --uri="https://REGION-PROJECT_ID.cloudfunctions.net/collect-data" \
     --http-method=POST
   ```

#### Tối ưu chi phí

- **Preemptible VMs**: Cho jobs training (~70% rẻ hơn)
- **Spot Instances**: Cho xử lý dữ liệu
- **Auto-scaling**: Scale API theo traffic
- **Storage Classes**: Sử dụng Nearline/Coldline cho dữ liệu lịch sử

### Đóng góp

1. Fork repository
2. Tạo nhánh tính năng: `git checkout -b feature/new-feature`
3. Commit thay đổi: `git commit -am 'Add new feature'`
4. Push lên nhánh: `git push origin feature/new-feature`
5. Tạo Pull Request

### Giấy phép

Giấy phép MIT - xem [LICENSE](LICENSE) để biết chi tiết.