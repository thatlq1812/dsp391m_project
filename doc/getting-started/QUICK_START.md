Đây là **hướng dẫn thao tác chuẩn** để AI Agent tự khởi động dự án “traffic-forecast-node-radius” với **Google Maps + Open-Meteo + Overpass**.
# 0) Đầu vào bắt buộc
* `.env`:
```
GOOGLE_MAPS_API_KEY=...
OPENMETEO_BASE_URL=https://api.open-meteo.com/v1/forecast
OVERPASS_URL=https://overpass-api.de/api/interpreter
PARQUET_DIR=./data/parquet
REDIS_URL=redis://localhost:6379/0
POSTGRES_DSN=postgresql://user:pass@localhost:5432/traffic
TZ=Asia/Ho_Chi_Minh
```
* Vùng chạy: HCMC `bbox=[10.67,106.60,10.90,106.84]` (có thể đổi).
# 1) Khởi tạo repo
**Work Order 1 — Scaffold**
* Sinh cấu trúc:
```
repo/{apps/api,apps/scheduler,collectors/{google,open_meteo,overpass},pipelines/{normalize,enrich,features,model,serving},infra/{docker,sql},configs, data,tests}
```
* Tạo `PROJECT_SPEC.yaml`:
```
project: traffic-forecast-node-radius
timezone: Asia/Ho_Chi_Minh
cadence: {traffic_minutes: 10, weather_minutes: 10, inference_sla_seconds: 60}
horizons_min: [5,15,30,60]
```
* Sinh `docker-compose.yml` (Postgres, Redis, API).
**DoD:** `docker compose up -d` chạy OK.
# 2) DB & migrations
**Work Order 2 — DB schema**
* Tạo bảng `nodes`, `forecasts`, `ingest_log` (như đã nêu trước đó).
**DoD:**Kết nối Postgres OK, bảng tồn tại.
# 3) Collector Overpass (one-off)
**Work Order 3 — overpass seed**
* Truy vấn Overpass tạo lưới node chính:
```
[out:json][timeout:120];
(
way["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]
(10.67,106.60,10.90,106.84);
);
out geom;
```
* Sinh `nodes.parquet`:
```
node_id,lat,lon,road_type,lane_count?,speed_limit?,free_flow_kmh,radius_group
```
**DoD:** `nodes.parquet` ≥ 3k node HCMC.
# 4) Collector Open-Meteo
**Work Order 4 — weather**
* Với danh sách node (lat,lon), gom theo lưới 0.05°; gọi:
```
GET /v1/forecast?latitude=..&longitude=..&hourly=temperature_2m,precipitation,wind_speed_10m&current=temperature_2m,precipitation,wind_speed_10m&timezone=Asia%2FHo_Chi_Minh
```
* Project về từng node; ghi `weather_snapshot.parquet`.
**DoD:** file snapshot theo timestamp, rows ≈ số node.
# 5) Collector Google Directions → suy tốc độ
**Work Order 5 — traffic_from_google**
* Chọn tập cạnh đại diện giữa node kề (k-NN=3 trong bán kính 800m).
* Với mỗi cặp `(A,B)`:
* `distance_km = haversine(A,B)`
* Gọi Directions:
```
GET https://maps.googleapis.com/maps/api/directions/json
?origin={A.lat},{A.lon}
&destination={B.lat},{B.lon}
&departure_time=now
&traffic_model=best_guess
&key=$GOOGLE_MAPS_API_KEY
```
* Lấy `duration_in_traffic_sec` và `duration_sec` (freeflow≈`duration_sec`).
* Suy tốc độ:
* `avg_speed_kmh = distance_km / (duration_in_traffic_sec/3600)`
* `freeflow_speed_kmh = distance_km / (duration_sec/3600)`
* Gộp về **node**: tốc độ node = median tốc độ các cạnh chạm node.
* Ghi `traffic_snapshot.parquet`:
```
ts,node_id,avg_speed_kmh,vehicle_count? (null),freeflow_speed_kmh
```
**Rate limit:** batch 50–100 cặp/lượt, quay vòng theo 10 phút; có thể dùng **Distance Matrix** cho nhiều điểm.
**DoD:** có snapshot mỗi 10′, >80% node có giá trị.
# 6) Normalize → Enrich
**Work Order 6 — standardize**
* Resample 10′; forward-fill 30′; chuẩn TZ VN.
* Tính `congestion_level` theo free-flow:
```
ratio = avg_speed_kmh / max(1e-3, freeflow_speed_kmh)
level = 0(>=0.9),1(0.8–0.9),2(0.65–0.8),3(0.5–0.65),4(0.35–0.5),5(<0.35)
```
* Lưu `features_nodes.parquet`:
```
ts,node_id,lag_1..lag_6,weather_feats,time_onehot,degree,free_flow_kmh,congestion_level
```
**DoD:** file features có đầy đủ cột, không NaN quan trọng.
# 7) Baseline model
**Work Order 7 — LSTM multi-horizon**
* Train rolling window per node hoặc global model với `node_id_embed`.
* Output `preds_nodes.parquet`:
```
ts_generated,node_id,horizon_min,speed_kmh_pred,flow_pred?,congestion_pred,sigma
```
**DoD:** sinh đủ 4 horizons T+5/15/30/60, latency ≤ 1s/1000 node trên dev.
# 8) Serving API (FastAPI)
**Work Order 8 — endpoints**
* `GET /v1/nodes/{id}/forecast?horizons=5,15,30`
* `POST /v1/routes/eta {origin,dest,ts}`
* Map tuyến → chuỗi node gần đường (snap-to-road dùng Roads API nếu bật).
* ETA = tổng thời gian qua các node theo `speed_kmh_pred`.
* `GET /v1/areas/heatmap?radius=500&ts=...`
* `GET /v1/alerts/live` (anomaly từ `congestion_level` + spike detector).
**Cache:**Redis TTL 30–40′, fallback parquet gần nhất.
**DoD:**OpenAPI hợp lệ, 4 endpoint trả dữ liệu thật.
# 9) Scheduler
**Work Order 9 — APScheduler**
* Mỗi 10′: `collect_open_meteo`, `collect_google_directions` (lệch nhau 15s).
* Mỗi 10′ +30s: `run_infer_batch`.
**DoD:** 3 chu kỳ liên tiếp không lỗi; ingest_lag_sec < 120.
# 10) Monitoring
**Work Order 10 — Prometheus**
* Expose:
```
ingest_lag_sec, ingest_rows, ingest_latency_ms
inference_latency_ms, inference_fail_count
mae_{5,15,30,60}, rmse_{...}, macro_f1_{...}
missing_rate_pct, drift_psi
```
**DoD:** `/metrics` có số đo, cảnh báo nếu `inference_latency_ms>1000`.
# 11) Kiểm thử nhanh
**Work Order 11 — E2E smoke**
* Seed Overpass → OK.
* Chạy 1 vòng collectors → có `traffic_snapshot.parquet`, `weather_snapshot.parquet`.
* Build features → OK.
* Infer → sinh `preds_nodes.parquet`.
* Gọi `/v1/nodes/{id}/forecast` trả JSON hợp lệ.
**DoD:**Hoàn tất E2E trong ≤ 15 phút từ lúc có `.env`.
# 12) Tiêu chí nghiệm thu
* **SLA chu kỳ:** ingest→infer ≤ 60s với 5k node.
* **Độ phủ:** ≥ 80% node có `avg_speed_kmh` mỗi chu kỳ.
* **Độ tin cậy:** 0 lỗi collector trong 24h, retry/backoff hoạt động.
* **API:** 4 endpoint phản hồi < 200ms từ cache, < 1.2s từ parquet.
# 13) Gợi ý tối ưu chi phí Google
* Dùng **Distance Matrix** cho cụm điểm thay vì nhiều lệnh Directions lẻ.
* Hash cặp `(A,B)` theo ngày; **chỉ gọi lại** khi chênh ETA>10% hoặc sau 30′.
* Lưu `ETag` nội bộ cho tuyến lặp lại cao.
Bạn muốn tôi sinh **gói khởi tạo** gồm: `PROJECT_SPEC.yaml`, `DB_SCHEMA.sql`, `docker-compose.yml`, khung FastAPI, scheduler, và mẫu collectors theo đúng nội dung trên để tải về không?
