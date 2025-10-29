# Huong Dan Su Dung He Thong v5.0

**Nguoi bao tri:**Le Quang That (Xiel) - SE183256 
**Phien ban:** 5.0.0 
**Ngay:** 29/10/2025

---

## Thay Doi Chinh

### 1. Loai Bo Hoan Toan Mock API

- CHI dung Google Directions API that
- KHONG co fallback mock nua
- Can co GOOGLE_MAPS_API_KEY hop le

### 2. Cache Thong Minh

- Overpass topology: Cache vinh vien (chi collect 1 lan)
- Weather: Cache 1 gio
- Traffic: KHONG cache (real-time data)

### 3. Tang Coverage

- Radius: 1024m → 4096m (tang 4 lan)
- Nodes: 64 → 128 (tang gap doi)
- Khoang cach toi thieu giua cac nodes: 200m

### 4. Chi Phi Du Kien

- 1 lan collect: 384 API calls = $1.92
- 1 thang (25 lan/ngay): $1,440
- Co the giam xuong $720/thang neu giam tan suat

---

## Cai Dat Nhanh

### Buoc 1: Thiet Lap API Key

```bash
export GOOGLE_MAPS_API_KEY="your-api-key-here"
```

LUU Y: API key phai co quyen truy cap Directions API

### Buoc 2: Kich Hoat Moi Truong

```bash
conda activate dsp
cd /d/UNI/DSP391m/project
```

### Buoc 3: Collect Topology (Chi 1 Lan)

```bash
python traffic_forecast/collectors/overpass/collector_v5.py
```

Ket qua:

- Tao file `cache/overpass_topology.json`
- 128 nodes (major intersections)
- Khoang cach toi thieu 200m

### Buoc 4: Collect Traffic Data

```bash
# Test mode (chi 10 edges)
export GOOGLE_TEST_LIMIT=10
python traffic_forecast/collectors/google/collector_v5.py

# Full collection (384 edges)
unset GOOGLE_TEST_LIMIT
python traffic_forecast/collectors/google/collector_v5.py
```

### Buoc 5: Collection Day Du

```bash
# Chay tat ca (Overpass + Google + Weather)
python scripts/collect_and_render.py --once
```

---

## Cau Truc File

### File Moi

```
configs/
 project_config.yaml # Config v5.0 (da cap nhat)
 project_config_backup.yaml # Backup v4.0

traffic_forecast/collectors/
 google/collector_v5.py # Real API only
 overpass/collector_v5.py # Voi caching
 overpass/node_selector_v5.py # Voi distance filtering

cache/
 overpass_topology.json # Topology cache (vinh vien)

doc/
 README_V5.md # Tai lieu tieng Anh
 HUONG_DAN_V5.md # Tai lieu nay
```

---

## Thuat Toan Chon Node

### Tieu Chi

1. **Degree**: >= 6 duong ket noi
2. **Importance**: >= 40.0 diem
3. **Distance**: >= 200m tu cac nodes khac
4. **Road Type**: Chi motorway, trunk, primary
5. **Top 128**: Xep hang theo importance

### Cach Tinh Diem Importance

```
motorway: 10 diem
trunk: 9 diem
primary: 8 diem

Tong diem = tong diem cac duong + (so loai duong * 2)
```

### Loc Theo Khoang Cach

- Sap xep nodes theo importance (cao → thap)
- Chon lan luot, bo qua neu < 200m tu node da chon
- Ket qua: Phan bo deu tren toan vung

---

## Giam Sat He Thong

### Kiem Tra Ti Le Thanh Cong

```bash
# Kiem tra log gan nhat
tail -100 data_runs/*/manifest.json | grep "success_rate"
```

Muc tieu: > 95%

### Kiem Tra Chi Phi

```bash
# Dem API calls trong ngay
find data_runs -name "traffic_edges.json" -mtime -1 | \
 xargs jq 'length' | \
 awk '{sum+=$1} END {print sum " API calls"}'
```

### Kiem Tra Cache

```bash
# Xem cache
ls -lh cache/

# Lam moi cache (neu can)
python traffic_forecast/collectors/overpass/collector_v5.py --force-refresh
```

---

## Xu Ly Loi Thuong Gap

### Loi 1: Khong Co API Key

```
ValueError: GOOGLE_MAPS_API_KEY environment variable not set
```

Giai phap:

```bash
export GOOGLE_MAPS_API_KEY="key-hop-le"
```

### Loi 2: Ti Le That Bai Cao

```
Success rate: 75%
```

Nguyen nhan:

- Het quota API
- API key chua kich hoat Directions API
- Mang khong on dinh

Giai phap:

1. Kiem tra quota tren Google Cloud Console
2. Kich hoat Directions API
3. Kiem tra ket noi mang

### Loi 3: Cache Khong Hoat Dong

```
Cache file not found
```

Giai phap:

```bash
mkdir -p cache
python traffic_forecast/collectors/overpass/collector_v5.py
```

---

## So Sanh v4.0 vs v5.0

| Metric | v4.0 | v5.0 | Thay Doi |
| --------------- | ----- | ------ | -------- |
| Nodes | 64 | 128 | +100% |
| Radius | 1024m | 4096m | +300% |
| Mock API | Co | Khong | Loai bo |
| Cache Overpass | Khong | Co | Them moi |
| Distance filter | Khong | 200m | Them moi |
| Retry | Khong | 3 lan | Them moi |
| Chi phi/thang | $720 | $1,440 | +100% |

---

## Cac Buoc Tiep Theo

### Uu Tien 1 (Ngay lap tuc)

1. Test collection voi real API
2. Kiem tra cache hoat dong
3. Do luong success rate
4. Tinh chi phi thuc te

### Uu Tien 2 (Tuan toi)

1. Tao visualizations
2. Feature importance analysis
3. Cross-validation
4. Cap nhat notebooks

### Uu Tien 3 (Dai han)

1. Train lai ML models
2. Deploy API endpoints
3. Cap nhat dashboard
4. Don dep tai lieu

---

## Lenh Thuong Dung

```bash
# Collection day du
python scripts/collect_and_render.py --once

# Test mode (10 edges)
export GOOGLE_TEST_LIMIT=10
python traffic_forecast/collectors/google/collector_v5.py

# Lam moi cache
python traffic_forecast/collectors/overpass/collector_v5.py --force-refresh

# Xem log
tail -f logs/collection.log
```

---

## Lien He

**Email:** fxlqthat@gmail.com 
**GitHub:** thatlq1812

---

**KET THUC HUONG DAN**
