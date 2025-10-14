# Traffic Forecast – Deployment & Operations Playbook

This guide tổng hợp toàn bộ tài liệu vận hành của dự án `traffic-forecast-node-radius`. Bạn có thể theo thứ tự từ chuẩn bị môi trường đến triển khai, vận hành và giám sát hệ thống. Mọi đường dẫn, lệnh đều tương đối với thư mục gốc của repo.

---

## 1. Giới thiệu thành phần hệ thống

- **Collectors** (`traffic_forecast.collectors`): thu thập dữ liệu từ Overpass (mạng đường), Open-Meteo (thời tiết), Google Directions (tốc độ giao thông). Dữ liệu lưu dưới `data/`.
- **Pipelines** (`traffic_forecast.pipelines`):
  - `normalize`: chuẩn hóa dữ liệu cạnh giao thông → ảnh chiết suất node.
  - `features`: ghép traffic – weather – forecast thành `features_nodes_v2.json`.
  - `preprocess`: (mới) làm sạch, scale, split train/val, lưu scaler và metadata.
  - `model`: huấn luyện và suy luận (train, infer) dựa trên dữ liệu đã preprocess.
  - `enrich`: (tùy chọn) geocode sự kiện và đánh giá ảnh hưởng.
- **CLI** (`traffic_forecast.cli`): tiện ích chạy collectors, visualize trực tiếp.
- **Scheduler** (`traffic_forecast.scheduler`): dùng APScheduler để điều phối collectors/pipelines định kỳ.
- **API** (`traffic_forecast.api`): FastAPI phục vụ dự báo cho từng nút giao.
- **Scripts & Tools**: các tiện ích chạy nhanh (`scripts/collect_and_render.py`, `tools/*`).
- **Tài liệu**: `doc/README.md` phân nhóm tài liệu chi tiết; file này gói gọn các bước thực thi chính.

---

## 2. Chuẩn bị môi trường

### 2.1. Yêu cầu hệ thống
- Python 3.8+
- (Khuyến nghị) Conda/Miniconda để quản lý môi trường
- Git, Docker (khi triển khai)

### 2.2. Thiết lập môi trường
```bash
git clone https://github.com/thatlq1812/dsp391m_project.git
cd dsp391m_project
conda env create -f environment.yml   # hoặc python -m venv .venv && source .venv/bin/activate
conda activate dsp
pip install -r requirements.txt
```

### 2.3. Cấu hình biến môi trường
Sao chép và chỉnh sửa `.env_template`:
```bash
cp .env_template .env
```

Điền các khóa/URL cần thiết (ví dụ):
```
GOOGLE_MAPS_API_KEY=...
OPENMETEO_BASE_URL=https://api.open-meteo.com/v1/forecast
OVERPASS_URL=https://overpass-api.de/api/interpreter
POSTGRES_DSN=postgresql://user:pass@localhost:5432/traffic
REDIS_URL=redis://localhost:6379/0
TZ=Asia/Ho_Chi_Minh
```

---

## 3. Chạy collectors & pipeline dữ liệu

### 3.1. Thu thập dữ liệu một lần
```bash
python scripts/collect_and_render.py --once
```
Tùy chọn:
- `--bbox min_lat,min_lon,max_lat,max_lon` để giới hạn khu vực.
- `--no-visualize` nếu muốn bỏ qua bước vẽ biểu đồ.

Lệnh trên sẽ:
1. Gọi `traffic_forecast.cli.run_collectors` để chạy Overpass → Open-Meteo → Google.
2. Sinh `data/node/<timestamp>/collectors/...` và `data/images/<timestamp>.png`.
3. Hợp nhất vào `data/node/<timestamp>/run_data.json`.

### 3.2. Chạy collectors định kỳ
```bash
python scripts/collect_and_render.py --interval 900  # 15 phút
```

### 3.3. Chạy từng module thủ công
```bash
python -m traffic_forecast.collectors.overpass.collector
python -m traffic_forecast.collectors.open_meteo.collector
python -m traffic_forecast.collectors.google.collector
```

---

## 4. Xử lý & huấn luyện mô hình

### 4.1. Chuẩn hóa dữ liệu (preprocess)
Pipeline đọc `data/features_nodes_v2.json`, làm sạch, thống kê trực quan, chia train/val, lưu scaler và metadata:
```bash
PYTHONPATH=. python3 -m traffic_forecast.pipelines.preprocess.preprocess
```
Đầu ra chính:
- `data/processed/train.parquet`, `data/processed/val.parquet`
- `models/feature_scaler.pkl`
- `data/processed/metadata.json` (feature list, imputation, liên kết scaler)
- `data/processed/summary.json`, `data/processed/report.md`, `data/processed/samples_head.csv`
- `data/processed/plots/target_distribution.png` (nếu matplotlib khả dụng)

Mẹo: kiểm tra nhanh bằng `python3 -m json.tool data/processed/summary.json` hoặc mở `report.md` để xem missing ratio và mẫu dữ liệu.

### 4.2. Huấn luyện mô hình
```bash
PYTHONPATH=. python3 -m traffic_forecast.pipelines.model.train
```
Lệnh tự động chạy preprocessing nếu thiếu artefact. Đầu ra:
- `models/linear_v2.pkl` (hoặc model tương ứng cấu hình)
- `models/model_metadata.json` (siêu dữ liệu, metrics, scaler path)
- `models/training_report.json` (tóm tắt training, hyperparameters)
- `data/processed/val_predictions.csv` (ground-truth & prediction trên tập validation)

**Lựa chọn mô hình** (*`configs/project_config.yaml` → `pipelines.model`*):
- `type: linear_regression` (mặc định) – nhanh, dễ giải thích.
- `type: random_forest` – phi tuyến, tận dụng scikit-learn RandomForestRegressor.
- `type: gradient_boosting` – GradientBoostingRegressor (tree boosting). Thêm hyperparameter qua `params:` (ví dụ `n_estimators`, `max_depth`, `learning_rate`).
Các model nâng cao hơn (GNN, sequence-based) có thể tích hợp bằng cách mở rộng `traffic_forecast/models/registry.py` và cập nhật pipeline.

### 4.3. Suy luận hàng loạt
```bash
PYTHONPATH=. python3 -m traffic_forecast.pipelines.model.infer
```
Trả về:
- `data/predictions.json`
- `data/predictions_summary.json` (số lượng, model dùng, timestamp)

### 4.4. Nâng cấp mô hình (tùy chọn)
- **Tree-based**: điều chỉnh `pipelines.model.type` sang `random_forest` hoặc `gradient_boosting`, bổ sung tham số trong `params` (ví dụ `n_estimators`, `max_depth`).
- **Mô hình chuỗi thời gian / GNN**: cần mở rộng pipeline để xây chuỗi lịch sử và ma trận lân cận (edges Overpass). Sau khi chuẩn bị dữ liệu, thêm factory mới vào `traffic_forecast/models/registry.py` và cập nhật `DEPLOY.md` với lệnh huấn luyện tương ứng.

---

## 5. Phục vụ API & Dashboard

### 5.1. API FastAPI
```bash
uvicorn traffic_forecast.api.main:app --reload --port 8000
```
Endpoint chính:
- `/`               – mô tả API
- `/health`         – kiểm tra file `data/predictions.json`, biến DB
- `/v1/nodes/{node_id}/forecast?horizon=15`

### 5.2. Live Dashboard
```bash
python scripts/live_dashboard.py
```
Ứng dụng FastAPI nhỏ phục vụ dữ liệu `data/nodes.json` và `data/traffic_snapshot_normalized.json`. Có thể dùng `scripts/serve_latest_run.py` để copy dữ liệu mới nhất vào `data/`.

---

## 6. Scheduler & vận hành định kỳ

### 6.1. Chạy scheduler nội bộ
```bash
python -m traffic_forecast.scheduler.main
```
Scheduler đọc `configs/project_config.yaml` và lên lịch cho các job:
- Collector Open-Meteo (nowcast & forecast)
- Collector Google
- Pipeline normalize, features, infer …

### 6.2. Dọn dẹp dữ liệu cũ
```bash
python scripts/cleanup_runs.py --days 7
```
Xóa thư mục trong `data/node` & `data/images` cũ hơn số ngày chỉ định.

---

## 7. Triển khai hạ tầng (tóm tắt từ doc/operations)

### 7.1. Kiến trúc triển khai
- **Docker Compose** (`docker-compose.prod.yml`) gồm API, scheduler, worker phụ.
- **CSDL**: PostgreSQL (schema trong `infra/sql/schema.sql`), Redis (cache).
- **Giám sát**: logs qua `docker logs`, có thể tích hợp Prometheus/Alertmanager (xem thêm trong `doc/operations/deployment.md`).

### 7.2. Provision VM trên GCP/AWS
1. Tạo VM Ubuntu 22.04, mở port 22/80/443/8000.
2. Cài Docker, Docker Compose, Git.
3. Đồng bộ `.env`, `docker-compose.prod.yml`, `infra/nginx/nginx.conf`.
4. Triển khai:
   ```bash
   docker compose -f docker-compose.prod.yml up -d
   ```
5. Cấu hình Nginx reverse proxy (xem `doc/operations/vm_provisioning.md`).

### 7.3. Runbook vận hành GCP (rút gọn)
- **Khởi động dịch vụ**: `docker compose start api scheduler`
- **Theo dõi collectors**: kiểm tra logs `docker logs <collector-container>`
- **Sao lưu**: snapshot `data/`, `models/`, backup Postgres bằng `pg_dump`.
- **Khôi phục**: copy dữ liệu vào VM mới, chạy lại `docker compose up`.

---

## 8. Thử nghiệm & scripts bổ trợ
- `tools/check_edges.py` – kiểm tra số lượng cạnh sinh ra sau normalize.
- `tools/debug_google_api.py` – test key Google Directions.
- `tools/test_google_limited.py` – chạy thử collector Google với 10 cạnh.
- `tools/test_rate_limiter.py` – mô phỏng rate limiter.
- `scripts/serve_latest_run.py` – copy dữ liệu run mới nhất về `data/`.

---

## 9. Quy trình tổng quát từ đầu đến cuối

1. **Chuẩn bị**: clone repo, tạo môi trường, cấu hình `.env`.
2. **Thu thập dữ liệu**: chạy collectors (`python scripts/collect_and_render.py --once`) để lấy dữ liệu mẫu.
3. **Chuẩn hóa & huấn luyện**:
   ```bash
   PYTHONPATH=. python3 -m traffic_forecast.pipelines.features.build_features
   PYTHONPATH=. python3 -m traffic_forecast.pipelines.preprocess.preprocess
   PYTHONPATH=. python3 -m traffic_forecast.pipelines.model.train
   ```
4. **Sinh dự báo**: `PYTHONPATH=. python3 -m traffic_forecast.pipelines.model.infer`.
5. **Kiểm tra API/dashboard**: chạy FastAPI và dashboard nếu cần.
6. **Triển khai production**:
   - Chuẩn bị VM/Docker, sao chép `.env`.
   - `docker compose -f docker-compose.prod.yml up -d`.
   - Cấu hình scheduler.
7. **Vận hành & giám sát**:
   - Theo dõi logs collectors, API.
   - Chạy cleanup, backup định kỳ.

---

## 10. Phụ lục & tài liệu chi tiết

Để tham khảo sâu hơn (bao gồm schema chi tiết, báo cáo, yêu cầu), xem:
- `doc/README.md` – index nội dung tài liệu.
- `doc/getting-started/*` – hướng dẫn khởi tạo, cấu hình nâng cao.
- `doc/reference/*` – mô hình dữ liệu, thiết kế schema.
- `doc/operations/*` – runbook GCP, hướng dẫn triển khai chi tiết.
- `doc/history/progress.md` – nhật ký tiến độ.
