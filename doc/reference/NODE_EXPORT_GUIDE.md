# Node Information Export Guide
## Tổng Quan
Hệ thống hiện có **277 nodes** (traffic intersections) với các thông tin sau:
### [DONE] Thông Tin Hiện Có (Trong `data/nodes.json`)
| Field | Mô Tả | Ví Dụ |
| ------------- | ---------------- | ---------------------------------- |
| `node_id` | ID duy nhất | `node-10.768754-106.703297` |
| `lat` | Vĩ độ | `10.7687542` |
| `lon` | Kinh độ | `106.7032974` |
| `road_type` | Loại đường chính | `primary`, `secondary`, `tertiary` |
| `lane_count` | Số làn xe | `2` |
| `speed_limit` | Tốc độ giới hạn | `50` (km/h) |
### [DONE] Thông Tin Có Thể Tạo Ra
| Field | Mô Tả | Cách Tạo |
| ------------------- | ---------------- | ------------------------------------------- |
| `google_maps_link` | Link Google Maps | `https://www.google.com/maps?q={lat},{lon}` |
| `distance_to_point` | Khoảng cách | Tính bằng haversine |
| `bounding_box` | Vùng bao quanh | Min/max lat/lon |
### Thông Tin Cần Thu Thập Lại (Từ OSM)
**Sau khi chạy lại collector với NodeSelector mới**, sẽ có thêm:
| Field | Mô Tả | Nguồn |
| ---------------------- | ----------------------- | ------------------------------ |
| `street_names` | Tên các đường giao nhau | OSM way tags `name` |
| `intersection_name` | Mô tả nút giao | Auto-generated từ street_names |
| `way_ids` | OSM way IDs | OSM data |
| `degree` | Số đường giao nhau | Tính từ connections |
| `importance_score` | Điểm quan trọng | NodeSelector algorithm |
| `connected_road_types` | Loại đường kết nối | OSM way tags `highway` |
---
## Cách Sử Dụng
### 1. Xem Thông Tin Hiện Tại
```bash
# Phân tích cấu trúc data hiện có
python tools/show_node_info.py
# Tạo CSV nhanh với Google Maps links
python tools/show_node_info.py --generate-csv
```
**Output**: `data/nodes_quick.csv` với format:
```csv
Node ID,Latitude,Longitude,Google Maps Link,Road Type,Lanes,Speed Limit
node-10.768754-106.703297,10.7687542,106.7032974,"https://www.google.com/maps?q=...",primary,2,50
```
### 2. Export Chi Tiết (Với Thông Tin Hiện Tại)
```bash
# Export tất cả nodes ra CSV, JSON và Markdown
python tools/export_nodes_info.py --format all
# Chỉ export 10 nodes đầu (test)
python tools/export_nodes_info.py --limit 10 --format md
# Export CSV only
python tools/export_nodes_info.py --format csv --output-csv my_nodes.csv
```
**Output files**:
- `data/nodes_detailed.csv` - CSV với Google Maps links
- `data/nodes_detailed.json` - JSON với metadata đầy đủ
- `data/NODES_INFO.md` - Markdown table dễ đọc
### 3. Query OSM API Để Lấy Tên Đường (Chậm)
```bash
# Chỉ nên dùng cho ít nodes
python tools/export_nodes_info.py --use-osm-api --limit 5 --format md
```
[WARNING] **Cảnh báo**: Query OSM API rất chậm (1-2 giây/node), chỉ nên dùng cho số lượng nhỏ.
---
## Thu Thập Lại Data Với Street Names
### Cách 1: Chạy Lại Collector (Khuyến Nghị)
NodeSelector đã được cập nhật để tự động lưu tên đường:
```bash
# Chạy collector mới
conda run -n dsp python -m traffic_forecast.collectors.overpass.collector
# Hoặc dùng script
conda run -n dsp python scripts/collect_and_render.py --once --no-visualize
```
Sau khi chạy, `data/nodes.json` sẽ có thêm:
- `street_names`: `["Đường Võ Văn Tần", "Đường Nam Kỳ Khởi Nghĩa"]`
- `intersection_name`: `"Võ Văn Tần ∩ Nam Kỳ Khởi Nghĩa"`
- `way_ids`: `[123456, 789012]`
- `degree`: `3`
- `importance_score`: `25.5`
### Cách 2: Query OSM Từng Node (Chậm)
```python
import requests
def get_street_names(lat, lon):
url = "https://overpass-api.de/api/interpreter"
query = f"""
[out:json];
way["highway"]["name"](around:20,{lat},{lon});
out body;
"""
response = requests.get(url, params={'data': query})
data = response.json()
names = []
for elem in data.get('elements', []):
name = elem.get('tags', {}).get('name')
if name:
names.append(name)
return names
# Sử dụng
lat, lon = 10.7687542, 106.7032974
streets = get_street_names(lat, lon)
print(streets) # ['Võ Văn Tần', 'Nam Kỳ Khởi Nghĩa', ...]
```
---
## Format Xuất Ra
### CSV Format
```csv
Node ID,Latitude,Longitude,Google Maps Link,Intersection,Street Names,Road Type,Degree
node-10.768754-106.703297,10.768754,106.703297,https://...,Võ Văn Tần ∩ Nam Kỳ...,Võ Văn Tần; Nam Kỳ...,primary,3
```
### JSON Format
```json
{
"node_id": "node-10.768754-106.703297",
"lat": 10.7687542,
"lon": 106.7032974,
"google_maps_link": "https://www.google.com/maps?q=10.7687542,106.7032974",
"street_names": ["Võ Văn Tần", "Nam Kỳ Khởi Nghĩa"],
"intersection_name": "Võ Văn Tần ∩ Nam Kỳ Khởi Nghĩa",
"road_type": "primary",
"degree": 3,
"importance_score": 25.5,
"connected_road_types": ["primary", "secondary"],
"way_ids": [123456, 789012],
"lane_count": "2",
"speed_limit": "50"
}
```
### Markdown Format
```markdown
### 1. node-10.768754-106.703297
- **Coordinates**: 10.768754, 106.703297
- **Google Maps**: [Link](https://www.google.com/maps?q=10.7687542,106.7032974)
- **Intersection**: Võ Văn Tần ∩ Nam Kỳ Khởi Nghĩa
- **Streets**:
- Võ Văn Tần
- Nam Kỳ Khởi Nghĩa
- **Road Type**: primary
- **Degree**: 3 connecting roads
- **Importance Score**: 25.5
```
---
## Use Cases
### 1. Tạo Interactive Map
```python
import json
import folium
# Load nodes
with open('data/nodes.json') as f:
nodes = json.load(f)
# Create map centered on HCMC
m = folium.Map(location=[10.8231, 106.6297], zoom_start=12)
# Add markers
for node in nodes:
lat, lon = node['lat'], node['lon']
popup_text = f"""
<b>{node.get('intersection_name', 'Unknown')}</b><br>
Type: {node.get('road_type', 'N/A')}<br>
<a href='https://www.google.com/maps?q={lat},{lon}' target='_blank'>Google Maps</a>
"""
folium.Marker(
location=[lat, lon],
popup=popup_text,
icon=folium.Icon(color='red' if node.get('importance_score', 0) > 20 else 'blue')
).add_to(m)
m.save('traffic_nodes_map.html')
```
### 2. Find Nearest Node
```python
import math
def haversine(lat1, lon1, lat2, lon2):
R = 6371 # Earth radius in km
dlat = math.radians(lat2 - lat1)
dlon = math.radians(lon2 - lon1)
a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
c = 2 * math.asin(math.sqrt(a))
return R * c
def find_nearest_node(target_lat, target_lon, nodes):
nearest = None
min_dist = float('inf')
for node in nodes:
dist = haversine(target_lat, target_lon, node['lat'], node['lon'])
if dist < min_dist:
min_dist = dist
nearest = node
return nearest, min_dist
# Example: Find node nearest to Landmark 81
landmark81_lat, landmark81_lon = 10.7944, 106.7218
nearest_node, distance = find_nearest_node(landmark81_lat, landmark81_lon, nodes)
print(f"Nearest node: {nearest_node['node_id']}")
print(f"Distance: {distance:.2f} km")
print(f"Google Maps: https://www.google.com/maps?q={nearest_node['lat']},{nearest_node['lon']}")
```
### 3. Export Cho Google My Maps
```python
import csv
with open('data/nodes.json') as f:
nodes = json.load(f)
# Google My Maps format
with open('google_my_maps.csv', 'w', newline='', encoding='utf-8') as f:
writer = csv.writer(f)
writer.writerow(['Name', 'Description', 'Latitude', 'Longitude'])
for node in nodes:
name = node.get('intersection_name', node['node_id'])
desc = f"Type: {node.get('road_type')}, Degree: {node.get('degree', 'N/A')}"
writer.writerow([name, desc, node['lat'], node['lon']])
```
Sau đó import vào [Google My Maps](https://www.google.com/maps/d/).
---
## Quick Reference
### Files Generated
| File | Purpose | Size |
| -------------------------- | ---------------------------- | ------- |
| `data/nodes.json` | Original nodes (277) | ~80 KB |
| `data/nodes_quick.csv` | Quick export với Google Maps | ~20 KB |
| `data/nodes_detailed.csv` | Full export | ~30 KB |
| `data/nodes_detailed.json` | Full JSON | ~100 KB |
| `data/NODES_INFO.md` | Human-readable table | ~50 KB |
### Python One-Liners
```python
# Load và print Google Maps links
import json
with open('data/nodes.json') as f:
nodes = json.load(f)
for n in nodes[:5]:
print(f"https://www.google.com/maps?q={n['lat']},{n['lon']}")
# Tìm nodes có importance score cao nhất (sau khi re-collect)
sorted_nodes = sorted(nodes, key=lambda x: x.get('importance_score', 0), reverse=True)
for n in sorted_nodes[:10]:
print(f"{n['intersection_name']}: {n['importance_score']:.1f}")
# Tìm các primary roads
primary_nodes = [n for n in nodes if n.get('road_type') == 'primary']
print(f"Primary road nodes: {len(primary_nodes)}")
```
---
## Next Steps
1. **Chạy lại collector** để thu thập street names:
```bash
conda run -n dsp python scripts/collect_and_render.py --once --no-visualize
```
2. **Export với thông tin mới**:
```bash
python tools/export_nodes_info.py --format all
```
3. **Tạo visualization**:
```bash
python tools/create_node_map.py # (TODO: Create this tool)
```
---
**Author**: GitHub Copilot 
**Date**: October 25, 2025 
**Version**: 3.0.0
