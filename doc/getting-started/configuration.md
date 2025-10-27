Chuẩn v2 cho AI Agent.
# 1) `configs/nodes_schema_v2.json`
```json
{
"version": "2.0",
"entity": "node",
"fields": [
{"name":"node_id","type":"string","required":true},
{"name":"lat","type":"float"},
{"name":"lon","type":"float"},
{"name":"district","type":"string","nullable":true},
{"name":"province","type":"string","nullable":true},
{"name":"connected_nodes","type":"array<string>"},
{"name":"degree","type":"int"},
{"name":"radius_group","type":"int"},
{"name":"road_name","type":"string","nullable":true},
{"name":"road_type","type":"string"},
{"name":"lane_count","type":"int","nullable":true},
{"name":"speed_limit_kmh","type":"int","nullable":true},
{"name":"free_flow_kmh","type":"float"},
{"name":"ts","type":"timestamp"},
{"name":"avg_speed_kmh","type":"float","nullable":true},
{"name":"congestion_level","type":"int","nullable":true},
{"name":"eta_delta_ratio","type":"float","nullable":true},
{"name":"temperature_c","type":"float","nullable":true},
{"name":"rain_mm","type":"float","nullable":true},
{"name":"wind_speed_kmh","type":"float","nullable":true},
{"name":"forecast_temp_t5_c","type":"float","nullable":true},
{"name":"forecast_temp_t15_c","type":"float","nullable":true},
{"name":"forecast_temp_t30_c","type":"float","nullable":true},
{"name":"forecast_temp_t60_c","type":"float","nullable":true},
{"name":"forecast_rain_t5_mm","type":"float","nullable":true},
{"name":"forecast_rain_t15_mm","type":"float","nullable":true},
{"name":"forecast_rain_t30_mm","type":"float","nullable":true},
{"name":"forecast_rain_t60_mm","type":"float","nullable":true},
{"name":"forecast_wind_t5_kmh","type":"float","nullable":true},
{"name":"forecast_wind_t15_kmh","type":"float","nullable":true},
{"name":"forecast_wind_t30_kmh","type":"float","nullable":true},
{"name":"forecast_wind_t60_kmh","type":"float","nullable":true},
{"name":"lags_speed","type":"array<float>","nullable":true},
{"name":"time_onehot","type":"json","nullable":true},
{"name":"feature_vector","type":"array<float>","nullable":true},
{"name":"weight","type":"float","default":1.0},
{"name":"source_traffic","type":"string","nullable":true},
{"name":"fetch_time","type":"timestamp"},
{"name":"confidence","type":"float","nullable":true},
{"name":"missing_flags","type":"json","nullable":true}
],
"primary_key": ["node_id","ts"]
}
```
# 2) Collector bổ sung: `collectors/open_meteo/forecast.py`
* Input: `nodes.parquet`, `ts_now`.
* Gọi Open-Meteo (hourly) và nội suy thời gian thực cho T+5/15/30/60.
* Output parquet: `weather_forecast_YYYYMMDD_HHMM.parquet` với các cột:
```
ts,node_id,
forecast_temp_t5_c,forecast_temp_t15_c,forecast_temp_t30_c,forecast_temp_t60_c,
forecast_rain_t5_mm,forecast_rain_t15_mm,forecast_rain_t30_mm,forecast_rain_t60_mm,
forecast_wind_t5_kmh,forecast_wind_t15_kmh,forecast_wind_t30_kmh,forecast_wind_t60_kmh
```
# 3) Join features: `pipelines/features/build_features.py`
* Join theo `node_id` + cửa sổ thời gian gần `ts`.
* Thứ tự join: `traffic_snapshot` → `weather_snapshot` → `weather_forecast`.
* Điền thiếu: forward-fill 30′ cho snapshot; giữ nguyên forecast (không FF).
* Ghi `features_nodes.parquet` theo v2, đảm bảo đủ các trường forecast_*.
# 4) Cập nhật normalize rule: `pipelines/normalize/rules.yaml`
```yaml
time:
tz: Asia/Ho_Chi_Minh
resample_minutes: 10
weather_forecast:
horizons_min: [5,15,30,60]
interpolate: linear # nội suy phút từ hourly
clamp:
rain_mm: [0, 200]
wind_speed_kmh: [0, 200]
```
# 5) Kiểm tra chất lượng (DoD)
* `weather_forecast.parquet` có ≥ 95% node với đủ 12 cột forecast_*.
* `features_nodes.parquet` chứa đầy đủ 12 trường forecast và pass schema v2.
* Tỉ lệ null ở nhóm forecast_* ≤ 5% mỗi chu kỳ.
* Model input pipeline đọc được schema v2 không lỗi.
# 6) Điều chỉnh model input
* Thêm các cột forecast_* vào `feature_vector`.
* Cập nhật scaler/encoder để fit các cột mới.
* Ghi lại artifact phiên bản: `model_meta.version = "features_v2"`.
# 7) Lịch chạy (APScheduler)
* Giữ nhịp: mỗi 10′.
* Thứ tự:
1. `collect_open_meteo_nowcast`
2. `collect_open_meteo_forecast` mới
3. `collect_google_directions`
4. `build_features_v2`
5. `infer_batch`
# 8) Migrate API (không đổi contract)
* REST không cần đổi. Nội bộ `preds` sẽ dùng thêm forecast_* khi suy đoán.
# 9) Sample row (v2)
```json
{
"node_id":"HCM_00411","ts":"2025-10-08T17:20:00+07:00",
"avg_speed_kmh":22.4,"congestion_level":4,"eta_delta_ratio":1.95,
"temperature_c":31.2,"rain_mm":0.2,"wind_speed_kmh":14.0,
"forecast_temp_t5_c":31.1,"forecast_temp_t15_c":30.9,"forecast_temp_t30_c":30.6,"forecast_temp_t60_c":30.3,
"forecast_rain_t5_mm":0.4,"forecast_rain_t15_mm":0.8,"forecast_rain_t30_mm":1.2,"forecast_rain_t60_mm":1.4,
"forecast_wind_t5_kmh":15.0,"forecast_wind_t15_kmh":16.2,"forecast_wind_t30_kmh":17.1,"forecast_wind_t60_kmh":18.0
}
```
# 10) Acceptance test tự động: `tests/test_schema_v2.py`
* Validate `nodes_schema_v2.json` vs `features_nodes.parquet`.
* Assert đầy đủ 12 trường forecast_* và kiểu dữ liệu float.
* Fail nếu missing_rate > 5%.
Chỉ cần thả file schema trên vào `configs/`, thêm collector forecast, và cập nhật DAG.
