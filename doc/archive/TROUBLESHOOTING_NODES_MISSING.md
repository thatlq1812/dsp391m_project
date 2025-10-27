# Troubleshooting Guide: nodes.json Missing Error
## Vấn Đề (Problem)
Sau 1 ngày chạy, hệ thống gặp lỗi:
```
FileNotFoundError: Could not find nodes.json in RUN_DIR or data/
KeyError: 'nodes'
```
## Nguyên Nhân (Root Cause)
1. **Overpass collector** dùng cache bị hỏng, thiếu key 'nodes'
2. **Open-Meteo & Google collectors** không tìm thấy file `nodes.json`
3. File `nodes.json` chỉ tồn tại trong thư mục run đầu tiên (`20251025115239`)
4. Code không copy file này sang các run mới
## Giải Pháp Đã Áp Dụng (Applied Solution)
### 1. Copy nodes.json vào thư mục data chính
```bash
cp ~/dsp391m_project/data/node/20251025115239/collectors/overpass/nodes.json \
~/dsp391m_project/data/nodes.json
```
### 2. Xóa cache bị hỏng
```bash
rm -rf ~/dsp391m_project/cache/*
```
### 3. Tắt caching trong config
```yaml
# File: configs/project_config.yaml
cache:
enabled: false # Đã đổi từ true sang false
```
### 4. Script tự động fix
Đã tạo script `fix_nodes_issue.sh` để tự động:
- Copy nodes.json nếu thiếu
- Xóa cache
- Tắt caching
## Kiểm Tra Xem Đã Fix Chưa (Verification)
### Kiểm tra file tồn tại:
```bash
ls -lh ~/dsp391m_project/data/nodes.json
```
**Expected output:**
```
-rw-r--r-- 1 user user 22K Oct 26 13:48 nodes.json
```
### Kiểm tra cache đã tắt:
```bash
grep "enabled:" ~/dsp391m_project/configs/project_config.yaml
```
**Expected output:**
```yaml
enabled: false
```
### Test collectors manual:
```bash
cd ~/dsp391m_project
~/miniconda3/envs/dsp/bin/python -m traffic_forecast.collectors.overpass.collector
~/miniconda3/envs/dsp/bin/python -m traffic_forecast.collectors.open_meteo.collector
~/miniconda3/envs/dsp/bin/python -m traffic_forecast.collectors.google.collector
```
**Expected output:**
- Overpass: "Saved 40 major intersections"
- Open-Meteo: "Collected weather for 40 nodes"
- Google: "Collected traffic for 120 edges"
### Xem log mới:
```bash
# Đợi đến lần chạy tiếp theo (interval 30 phút)
# Service restart lúc 14:01 UTC, lần chạy tiếp theo: ~14:31 UTC
sudo tail -100 ~/dsp391m_project/logs/collector.log | grep -A50 "Run at 2025-10-26 14:"
```
**Expected:** Không còn lỗi "FileNotFoundError" hoặc "KeyError: 'nodes'"
## Monitoring Commands
### Check service status:
```bash
sudo systemctl status traffic-collector --no-pager
```
### Watch logs real-time:
```bash
tail -f ~/dsp391m_project/logs/collector.log
```
### Count successful collections:
```bash
ls -1 ~/dsp391m_project/data/node/ | wc -l
```
## Lịch Chạy Tiếp Theo (Next Run Schedule)
Service sử dụng adaptive scheduler:
- **Peak hours (7-9 AM, 5-7 PM):** Every 30 minutes
- **Off-peak hours:** Every 60 minutes
- **Weekends:** Every 90 minutes
Current time: **14:06 UTC** (21:06 VN time - off-peak)
Last restart: **14:01 UTC**
Next collection: **~14:31 UTC** (30 minutes after restart)
## Tóm Tắt Fix (Summary)
**FIXED:**
1. nodes.json copied to global location
2. Cache cleared
3. Caching disabled
4. Service restarted
⏳ **WAITING:**
- Next collection cycle (~14:31 UTC) to verify fix
**VERIFIED (manual test):**
- All 3 collectors run successfully when tested manually
- Overpass: 40 nodes
- Open-Meteo: 40 weather records
- Google: 120 traffic edges (Mock API)
## Contact Support
If issue persists after next collection cycle, check:
1. Service logs for new errors
2. Disk space: `df -h`
3. File permissions: `ls -la ~/dsp391m_project/data/nodes.json`
**Maintainer:** THAT Le Quang (Xiel) 
**Email:** fxlqthat@gmail.com
---
**Last Updated:** 2025-10-26 14:06 UTC 
**Status:** Fix applied, awaiting verification at 14:31 UTC
