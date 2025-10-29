# Hướng Dẫn Deploy Traffic Forecast v5.0 Lên GCP VM Mới

## Tổng Quan

Tài liệu này hướng dẫn chi tiết cách deploy hệ thống Traffic Forecast v5.0 lên một Google Cloud Platform (GCP) project hoàn toàn mới. Phù hợp cho:

- Khởi tạo project mới từ đầu
- Chuyển sang GCP account khác
- Setup môi trường production độc lập

**Thời gian ước tính:** 30-45 phút
**Chi phí:** $0 (sử dụng free tier e2-micro)

---

## Điều Kiện Tiên Quyết

### 1. Tài Khoản Google Cloud Platform

- [ ] Tài khoản Gmail active
- [ ] Đã đăng ký GCP (https://console.cloud.google.com)
- [ ] Free trial $300 credit (cho người dùng mới) hoặc billing account active

### 2. API Keys

- [ ] **Google Directions API key** đã enable và test thành công
- Hiện tại đang dùng: `AIzaSyA1PM9WoXzuFqobz6UbSLwIJcP9PAz3Zhk`
- Đã verify: 234/234 edges (100% success rate)
- Cost estimate: ~$28/day cho hourly collection

### 3. Môi Trường Local

- [ ] Python 3.8+ với conda environment 'dsp'
- [ ] Google Cloud SDK (gcloud CLI) đã cài đặt
- Download: https://cloud.google.com/sdk/docs/install
- Verify: `gcloud --version`
- [ ] Project code đã update version v5.0
- Cache topology: `cache/overpass_topology.json` (78 nodes)
- Config: `configs/project_config.yaml` (radius=4096m)

---

## Bước 1: Tạo GCP Project Mới

### 1.1. Tạo Project Qua Console

1. Truy cập: https://console.cloud.google.com
2. Click **Select a project** → **NEW PROJECT**
3. Điền thông tin:

- **Project name:** `traffic-forecast-dsp391m`
- **Project ID:**Tự động tạo (ví dụ: `traffic-forecast-dsp391m-12345`)
- **Location:**No organization (hoặc chọn organization nếu có)

4. Click **CREATE**
5. **Lưu lại Project ID** - cần dùng cho các bước sau

### 1.2. Verify Project Qua gcloud CLI

```bash
# Login to Google Cloud
gcloud auth login

# Liệt kê tất cả projects
gcloud projects list

# Set project mặc định
gcloud config set project traffic-forecast-dsp391m-12345

# Verify
gcloud config get-value project
```

**Output mong đợi:**

```
traffic-forecast-dsp391m-12345
```

---

## Bước 2: Enable Billing và APIs

### 2.1. Enable Billing

1. Vào: https://console.cloud.google.com/billing
2. Link billing account với project `traffic-forecast-dsp391m`
3. Nếu dùng Free Trial:

- $300 credit sẽ cover ~10 ngày full collection
- Không bị charge sau khi hết credit (cần enable manually)

### 2.2. Enable Required APIs

```bash
# Enable Compute Engine API (cho VM)
gcloud services enable compute.googleapis.com

# Enable Directions API (cho traffic data)
gcloud services enable directions-backend.googleapis.com

# Verify enabled services
gcloud services list --enabled
```

**Services cần có:**

- `compute.googleapis.com`
- `directions-backend.googleapis.com`

### 2.3. Setup API Key Restrictions (Bảo Mật)

1. Vào: https://console.cloud.google.com/apis/credentials
2. Click API key của bạn (`AIzaSyA1PM9WoXzuFqobz6UbSLwIJcP9PAz3Zhk`)
3. **Application restrictions:**

- Chọn: **IP addresses**
- Add: IP của VM (sẽ add sau khi tạo VM)

4. **API restrictions:**

- Chọn: **Restrict key**
- Select: **Directions API** only

5. Click **SAVE**

---

## Bước 3: Setup Local Environment

### 3.1. Update Project Configuration

Mở file `.env` trong project root:

```bash
# .env
GOOGLE_MAPS_API_KEY=AIzaSyA1PM9WoXzuFqobz6UbSLwIJcP9PAz3Zhk
GCP_PROJECT_ID=traffic-forecast-dsp391m-12345 # ← Update này
GCP_ZONE=asia-southeast1-a
VM_NAME=traffic-collector-v5
VM_MACHINE_TYPE=e2-micro
```

### 3.2. Verify Topology Cache

```bash
# Kiểm tra cache file tồn tại
ls -lh cache/overpass_topology.json

# Verify số nodes
python -c "import json; print(f'{len(json.load(open(\"cache/overpass_topology.json\"))[\"nodes\"])} nodes')"
```

**Output mong đợi:**

```
78 nodes
```

### 3.3. Test Google API Key Locally

Mở notebook `CONTROL_PANEL.ipynb`:

```python
# Cell: Test API Connection
from traffic_forecast.collectors.google.collector import GoogleDirectionsCollector
import json

# Load config
with open('configs/project_config.yaml') as f:
config = yaml.safe_load(f)

# Initialize collector
collector = GoogleDirectionsCollector(config)

# Load topology
with open('cache/overpass_topology.json') as f:
topology = json.load(f)

# Test one edge
nodes = topology['nodes']
result = collector.collect_edge(nodes[0], nodes[1])

print(f"Status: {result.get('status')}")
print(f"Duration: {result.get('duration_in_traffic')}s")
print(f"Distance: {result.get('distance_meters')}m")
```

**Output mong đợi:**

```
Status: success
Duration: 245s
Distance: 1850m
```

---

## Bước 4: Deploy VM Using GCP_DEPLOYMENT Notebook

### 4.1. Mở Jupyter Notebook

```bash
cd /d/UNI/DSP391m/project
conda activate dsp
jupyter notebook notebooks/GCP_DEPLOYMENT.ipynb
```

### 4.2. Configure VM Parameters (Cell 1)

```python
import os
from pathlib import Path

# Project configuration
PROJECT_ROOT = Path('/d/UNI/DSP391m/project')
GCP_PROJECT_ID = 'traffic-forecast-dsp391m-12345' # ← Your project ID
GCP_ZONE = 'asia-southeast1-a' # Gần Vietnam nhất
VM_NAME = 'traffic-collector-v5'
VM_MACHINE_TYPE = 'e2-micro' # Free tier eligible

# VM specifications
VM_CONFIG = {
'boot_disk_size': '30GB',
'boot_disk_type': 'pd-standard',
'image_family': 'ubuntu-2204-lts',
'image_project': 'ubuntu-os-cloud',
'network': 'default',
'subnet': 'default',
'tags': ['http-server', 'https-server'],
}

print(f" VM Configuration:")
print(f" Project: {GCP_PROJECT_ID}")
print(f" Zone: {GCP_ZONE}")
print(f" Machine: {VM_MACHINE_TYPE}")
print(f" Name: {VM_NAME}")
```

### 4.3. Create VM Instance (Cell 2)

```python
def create_vm_instance():
"""
Tạo VM instance mới trên GCP
"""
cmd = f"""
gcloud compute instances create {VM_NAME} \\
--project={GCP_PROJECT_ID} \\
--zone={GCP_ZONE} \\
--machine-type={VM_MACHINE_TYPE} \\
--network-interface=network-tier=PREMIUM,subnet=default \\
--maintenance-policy=MIGRATE \\
--provisioning-model=STANDARD \\
--scopes=https://www.googleapis.com/auth/cloud-platform \\
--tags=http-server,https-server \\
--create-disk=auto-delete=yes,boot=yes,device-name={VM_NAME},image=projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts,mode=rw,size=30,type=projects/{GCP_PROJECT_ID}/zones/{GCP_ZONE}/diskTypes/pd-standard \\
--no-shielded-secure-boot \\
--shielded-vtpm \\
--shielded-integrity-monitoring \\
--labels=env=production,app=traffic-forecast,version=v5 \\
--reservation-affinity=any
"""

print(" Creating VM instance...")
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if result.returncode == 0:
print(" VM created successfully!")
print(result.stdout)

# Get VM external IP
ip_cmd = f"gcloud compute instances describe {VM_NAME} --zone={GCP_ZONE} --format='get(networkInterfaces[0].accessConfigs[0].natIP)'"
ip_result = subprocess.run(ip_cmd, shell=True, capture_output=True, text=True)
vm_ip = ip_result.stdout.strip()

print(f"\n VM External IP: {vm_ip}")
print(f"\n ACTION REQUIRED:")
print(f" 1. Add {vm_ip} to API key IP restrictions")
print(f" 2. Update .env file: VM_EXTERNAL_IP={vm_ip}")

return vm_ip
else:
print(" VM creation failed!")
print(result.stderr)
return None

# Execute
vm_ip = create_vm_instance()
```

**Output mong đợi:**

```
Creating VM instance...
Created [https://www.googleapis.com/compute/v1/projects/traffic-forecast-dsp391m-12345/zones/asia-southeast1-a/instances/traffic-collector-v5].
NAME ZONE MACHINE_TYPE PREEMPTIBLE INTERNAL_IP EXTERNAL_IP STATUS
traffic-collector-v5 asia-southeast1-a e2-micro 10.148.0.2 34.124.XX.XXX RUNNING

VM created successfully!

VM External IP: 34.124.XX.XXX

ACTION REQUIRED:
1. Add 34.124.XX.XXX to API key IP restrictions
2. Update .env file: VM_EXTERNAL_IP=34.124.XX.XXX
```

### 4.4. Add VM IP to API Key Restrictions

1. Copy VM External IP: `34.124.XX.XXX`
2. Vào: https://console.cloud.google.com/apis/credentials
3. Click API key → Edit
4. **Application restrictions:**

- Add IP: `34.124.XX.XXX`

5. **Save**

### 4.5. Update .env File

```bash
# Add to .env
VM_EXTERNAL_IP=34.124.XX.XXX
```

---

## Bước 5: Upload Project Files to VM

### 5.1. Create Deployment Package (Cell 3)

```python
def create_deployment_package():
"""
Tạo tarball chứa toàn bộ project (trừ data/)
"""
print(" Creating deployment package...")

# Files to include
include_patterns = [
'traffic_forecast/',
'configs/',
'cache/',
'scripts/',
'.env',
'requirements.txt',
'setup.py',
]

# Create tarball
cmd = f"tar -czf /tmp/traffic-forecast-deploy.tar.gz {' '.join(include_patterns)}"
result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, capture_output=True, text=True)

if result.returncode == 0:
# Get package size
size = os.path.getsize('/tmp/traffic-forecast-deploy.tar.gz') / 1024 / 1024
print(f" Package created: /tmp/traffic-forecast-deploy.tar.gz ({size:.1f} MB)")
return '/tmp/traffic-forecast-deploy.tar.gz'
else:
print(" Package creation failed!")
print(result.stderr)
return None

# Execute
package_path = create_deployment_package()
```

### 5.2. Upload to VM (Cell 4)

```python
def upload_to_vm(local_file, remote_path='~/'):
"""
Upload file lên VM qua gcloud scp
"""
print(f" Uploading {local_file} to VM...")

cmd = f"gcloud compute scp {local_file} {VM_NAME}:{remote_path} --zone={GCP_ZONE}"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if result.returncode == 0:
print(" Upload successful!")
return True
else:
print(" Upload failed!")
print(result.stderr)
return False

# Execute
upload_to_vm('/tmp/traffic-forecast-deploy.tar.gz', '~/')
```

**Output mong đợi:**

```
Uploading /tmp/traffic-forecast-deploy.tar.gz to VM...
traffic-forecast-deploy.tar.gz 100% 15MB 5.2MB/s 00:03
Upload successful!
```

---

## Bước 6: Deploy on VM

### 6.1. SSH into VM (Cell 5)

```python
def ssh_to_vm(command=None):
"""
SSH vào VM và execute command (hoặc mở interactive shell)
"""
if command:
cmd = f"gcloud compute ssh {VM_NAME} --zone={GCP_ZONE} --command='{command}'"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
print(result.stderr)
return result.returncode == 0
else:
# Interactive shell
cmd = f"gcloud compute ssh {VM_NAME} --zone={GCP_ZONE}"
subprocess.run(cmd, shell=True)

# Test connection
ssh_to_vm('hostname')
```

### 6.2. Run Automated Deployment Script (Cell 6)

```python
def deploy_on_vm():
"""
Chạy automated deployment script trên VM
"""
deployment_script = """
# Extract project files
tar -xzf traffic-forecast-deploy.tar.gz
cd traffic-forecast

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip python3-venv git

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c 'from traffic_forecast import __version__; print(f"Version: {__version__}")'

# Test collection (1 edge only)
python -c '
from traffic_forecast.collectors.google.collector import GoogleDirectionsCollector
import json, yaml

with open("configs/project_config.yaml") as f:
config = yaml.safe_load(f)

collector = GoogleDirectionsCollector(config)

with open("cache/overpass_topology.json") as f:
topology = json.load(f)

nodes = topology["nodes"]
result = collector.collect_edge(nodes[0], nodes[1])
print(f"Test result: {result.get(\\"status\\")}")
'

echo " Deployment complete!"
"""

print(" Running deployment on VM...")

# Upload deployment script
with open('/tmp/deploy.sh', 'w') as f:
f.write(deployment_script)

upload_to_vm('/tmp/deploy.sh', '~/')

# Execute deployment
ssh_to_vm('bash deploy.sh')

# Execute
deploy_on_vm()
```

**Output mong đợi:**

```
Running deployment on VM...
Uploading /tmp/deploy.sh to VM...
Upload successful!

Extracting project files...
Installing system dependencies...
Creating virtual environment...
Installing Python packages...
Successfully installed traffic-forecast-5.0.0
Version: 5.0.0
Test result: success
Deployment complete!
```

---

## Bước 7: Setup Automated Collection

### 7.1. Create Cron Job (Cell 7)

```python
def setup_cron_collection(interval_minutes=60):
"""
Setup cron job để collect data tự động

Args:
interval_minutes: Khoảng thời gian giữa các lần collect (default: 60 phút)
"""
cron_script = f"""
#!/bin/bash
# Traffic Forecast Collection Cron Job
# Runs every {interval_minutes} minutes

cd ~/traffic-forecast
source venv/bin/activate

# Run collection
python -m traffic_forecast.cli collect \\
--config configs/project_config.yaml \\
--output data/downloads/download_$(date +%Y%m%d_%H%M%S)/ \\
--cache cache/overpass_topology.json \\
--log-level INFO \\
>> logs/collection_$(date +%Y%m%d).log 2>&1

# Keep only last 14 days of data
python scripts/data/cleanup_runs.py --days 14 >> logs/cleanup.log 2>&1
"""

# Upload cron script
with open('/tmp/collect_cron.sh', 'w') as f:
f.write(cron_script)

upload_to_vm('/tmp/collect_cron.sh', '~/traffic-forecast/')

# Make executable
ssh_to_vm('chmod +x ~/traffic-forecast/collect_cron.sh')

# Setup cron job
cron_entry = f"*/{interval_minutes} * * * * ~/traffic-forecast/collect_cron.sh"
ssh_to_vm(f'(crontab -l 2>/dev/null; echo "{cron_entry}") | crontab -')

print(f" Cron job configured: Every {interval_minutes} minutes")
print(f" Script: ~/traffic-forecast/collect_cron.sh")
print(f" Logs: ~/traffic-forecast/logs/collection_YYYYMMDD.log")

# Execute - Collect mỗi giờ
setup_cron_collection(interval_minutes=60)
```

### 7.2. Verify Cron Job

```python
# Check cron job
ssh_to_vm('crontab -l')
```

**Output mong đợi:**

```
*/60 * * * * ~/traffic-forecast/collect_cron.sh
```

### 7.3. Manual Test Run

```python
# Run collection manually to verify
ssh_to_vm('~/traffic-forecast/collect_cron.sh')
```

---

## Bước 8: Monitor Collection

### 8.1. Check Collection Logs (Cell 8)

```python
def check_collection_logs(lines=50):
"""
Xem logs collection mới nhất
"""
today = datetime.now().strftime('%Y%m%d')
log_file = f'~/traffic-forecast/logs/collection_{today}.log'

print(f" Last {lines} lines of {log_file}:")
ssh_to_vm(f'tail -n {lines} {log_file}')

# Execute
check_collection_logs(lines=50)
```

### 8.2. Check Data Files (Cell 9)

```python
def list_collected_data():
"""
Liệt kê các file data đã collect
"""
print(" Collected data files:")
ssh_to_vm('ls -lh ~/traffic-forecast/data/downloads/')

# Execute
list_collected_data()
```

### 8.3. Validate Collection Quality (Cell 10)

```python
def validate_vm_collection():
"""
Validate chất lượng data collection trên VM
"""
validation_script = """
cd ~/traffic-forecast
source venv/bin/activate

python -c '
import os, json, glob
from pathlib import Path

data_dir = Path("data/downloads")
runs = sorted(data_dir.glob("download_*"))

if not runs:
print(" No collection runs found!")
exit(1)

latest_run = runs[-1]
print(f" Latest run: {latest_run.name}")

# Count files
traffic_files = list(latest_run.glob("traffic_*.json"))
print(f" Traffic files: {len(traffic_files)}")

# Load and analyze
total_edges = 0
successful_edges = 0

for tf in traffic_files:
with open(tf) as f:
data = json.load(f)
total_edges += len(data.get("edges", []))
successful_edges += sum(1 for e in data.get("edges", []) if e.get("status") == "success")

success_rate = (successful_edges / total_edges * 100) if total_edges > 0 else 0
print(f" Total edges: {total_edges}")
print(f" Successful: {successful_edges}")
print(f" Success rate: {success_rate:.1f}%")

if success_rate >= 95:
print(" Collection quality: EXCELLENT")
elif success_rate >= 80:
print(" Collection quality: GOOD")
else:
print(" Collection quality: POOR")
'
"""

with open('/tmp/validate.sh', 'w') as f:
f.write(validation_script)

upload_to_vm('/tmp/validate.sh', '~/')
ssh_to_vm('bash validate.sh')

# Execute
validate_vm_collection()
```

**Output mong đợi:**

```
Latest run: download_20250128_140502
Traffic files: 3
Total edges: 234
Successful: 234
Success rate: 100.0%
Collection quality: EXCELLENT
```

---

## Bước 9: Download Collected Data

### 9.1. Download Latest Run (Cell 11)

```python
def download_latest_run(local_dir='data/vm_downloads/'):
"""
Download data run mới nhất từ VM về local
"""
# Create local directory
os.makedirs(local_dir, exist_ok=True)

# Get latest run name
result = subprocess.run(
f"gcloud compute ssh {VM_NAME} --zone={GCP_ZONE} --command='ls -t ~/traffic-forecast/data/downloads/ | head -n 1'",
shell=True, capture_output=True, text=True
)
latest_run = result.stdout.strip()

if not latest_run:
print(" No runs found on VM")
return False

print(f" Downloading {latest_run}...")

# Download using gcloud scp
remote_path = f'{VM_NAME}:~/traffic-forecast/data/downloads/{latest_run}'
cmd = f"gcloud compute scp --recurse {remote_path} {local_dir} --zone={GCP_ZONE}"

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if result.returncode == 0:
local_path = os.path.join(local_dir, latest_run)
print(f" Downloaded to: {local_path}")

# Count files
files = list(Path(local_path).glob("*.json"))
print(f" Files: {len(files)}")
return True
else:
print(" Download failed!")
print(result.stderr)
return False

# Execute
download_latest_run()
```

### 9.2. Download All Runs (Cell 12)

```python
def download_all_runs(local_dir='data/vm_downloads_all/'):
"""
Download tất cả data runs từ VM
"""
os.makedirs(local_dir, exist_ok=True)

print(" Downloading all runs from VM...")

remote_path = f'{VM_NAME}:~/traffic-forecast/data/downloads/*'
cmd = f"gcloud compute scp --recurse {remote_path} {local_dir} --zone={GCP_ZONE}"

result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if result.returncode == 0:
# Count runs
runs = list(Path(local_dir).glob("download_*"))
print(f" Downloaded {len(runs)} runs to: {local_dir}")
return True
else:
print(" Download failed!")
print(result.stderr)
return False

# Execute (optional - dùng khi cần download full dataset)
# download_all_runs()
```

---

## Bước 10: Cost Monitoring & VM Management

### 10.1. Estimate Monthly Cost (Cell 13)

```python
def estimate_monthly_cost():
"""
Ước tính chi phí hàng tháng
"""
# VM costs
vm_cost_per_hour = 0 # e2-micro free tier
hours_per_month = 730
vm_monthly = vm_cost_per_hour * hours_per_month

# API costs (from CONTROL_PANEL calculations)
edges_per_run = 234
runs_per_day = 24 # Hourly collection
api_cost_per_1000 = 0.005 # $5 per 1000 requests

api_calls_per_day = edges_per_run * runs_per_day
api_cost_per_day = (api_calls_per_day / 1000) * api_cost_per_1000
api_monthly = api_cost_per_day * 30

# Storage costs
storage_gb = 5 # ~5GB for 30 days of data
storage_cost_per_gb = 0.020 # Standard storage
storage_monthly = storage_gb * storage_cost_per_gb

# Total
total_monthly = vm_monthly + api_monthly + storage_monthly

print(" Monthly Cost Estimate:")
print(f" VM (e2-micro): ${vm_monthly:.2f}")
print(f" Directions API: ${api_monthly:.2f}")
print(f" Storage ({storage_gb}GB): ${storage_monthly:.2f}")
print(f" ─────────────────────────")
print(f" TOTAL: ${total_monthly:.2f}/month")
print(f"\n For 7-day academic project:")
print(f" Total cost: ${total_monthly * 7/30:.2f}")

# Execute
estimate_monthly_cost()
```

**Output mong đợi:**

```
Monthly Cost Estimate:
VM (e2-micro): $0.00
Directions API: $842.40
Storage (5GB): $0.10
─────────────────────────
TOTAL: $842.50/month

For 7-day academic project:
Total cost: $196.58
```

### 10.2. Stop VM (Cell 14)

```python
def stop_vm():
"""
Dừng VM để tiết kiệm chi phí (data vẫn giữ nguyên)
"""
print(f" Stopping VM: {VM_NAME}...")
cmd = f"gcloud compute instances stop {VM_NAME} --zone={GCP_ZONE}"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if result.returncode == 0:
print(" VM stopped successfully!")
print(" To restart: start_vm()")
else:
print(" Stop failed!")
print(result.stderr)

# Execute when needed
# stop_vm()
```

### 10.3. Start VM (Cell 15)

```python
def start_vm():
"""
Khởi động lại VM
"""
print(f"▶ Starting VM: {VM_NAME}...")
cmd = f"gcloud compute instances start {VM_NAME} --zone={GCP_ZONE}"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if result.returncode == 0:
print(" VM started successfully!")

# Get new IP (might change after restart)
ip_cmd = f"gcloud compute instances describe {VM_NAME} --zone={GCP_ZONE} --format='get(networkInterfaces[0].accessConfigs[0].natIP)'"
ip_result = subprocess.run(ip_cmd, shell=True, capture_output=True, text=True)
new_ip = ip_result.stdout.strip()

print(f" External IP: {new_ip}")
print(f" If IP changed, update API key restrictions!")
else:
print(" Start failed!")
print(result.stderr)

# Execute when needed
# start_vm()
```

### 10.4. Delete VM (Cell 16)

```python
def delete_vm():
"""
Xóa VM và tất cả data (KHÔNG THỂ KHÔI PHỤC!)
"""
print(f" WARNING: This will DELETE VM and all data!")
confirm = input(f"Type '{VM_NAME}' to confirm deletion: ")

if confirm != VM_NAME:
print(" Deletion cancelled")
return

print(f"Deleting VM: {VM_NAME}...")
cmd = f"gcloud compute instances delete {VM_NAME} --zone={GCP_ZONE} --quiet"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

if result.returncode == 0:
print(" VM deleted successfully!")
print(" To recreate: create_vm_instance()")
else:
print(" Deletion failed!")
print(result.stderr)

# Execute with caution!
# delete_vm()
```

---

## Troubleshooting

### Issue 1: VM Creation Failed - Quota Exceeded

**Triệu chứng:**

```
ERROR: (gcloud.compute.instances.create) Quota 'CPUS' exceeded. Limit: 8.0 in region asia-southeast1.
```

**Giải pháp:**

1. Vào: https://console.cloud.google.com/iam-admin/quotas
2. Filter: `Region: asia-southeast1`, `Metric: CPUs`
3. Click **EDIT QUOTAS** → Request increase to 12 CPUs
4. Hoặc đổi zone: `GCP_ZONE=us-central1-a`

### Issue 2: API Key Returns 403 Forbidden

**Triệu chứng:**

```
{
"status": "REQUEST_DENIED",
"error_message": "This API project is not authorized to use this API."
}
```

**Giải pháp:**

1. Verify Directions API enabled: `gcloud services list --enabled | grep directions`
2. Check API key restrictions: https://console.cloud.google.com/apis/credentials
3. Test với curl:

```bash
curl "https://maps.googleapis.com/maps/api/directions/json?origin=10.762622,106.660172&destination=10.771553,106.700806&key=YOUR_API_KEY"
```

### Issue 3: Collection Success Rate < 95%

**Triệu chứng:**

```
Latest run: download_20250128_140502
Success rate: 72.3%
Collection quality: POOR
```

**Giải pháp:**

1. Check logs: `ssh_to_vm('tail -n 100 ~/traffic-forecast/logs/collection_*.log')`
2. Common issues:

- Rate limiting: Reduce cron frequency (120 minutes)
- API quota: Check quota usage in GCP console
- Network issues: Verify VM can reach Google APIs

3. Manual test:

```python
ssh_to_vm('cd ~/traffic-forecast && source venv/bin/activate && python -m traffic_forecast.cli test-api')
```

### Issue 4: Cannot SSH to VM

**Triệu chứng:**

```
ERROR: (gcloud.compute.ssh) Could not SSH into the instance.
```

**Giải pháp:**

1. Check VM status: `gcloud compute instances list`
2. If TERMINATED: `start_vm()`
3. Check firewall:

```bash
gcloud compute firewall-rules list
```

4. Add SSH rule if missing:

```bash
gcloud compute firewall-rules create allow-ssh --allow tcp:22 --source-ranges 0.0.0.0/0
```

### Issue 5: Download Data Too Slow

**Triệu chứng:**

```
Downloading download_20250128_140502...
[Still running after 10 minutes]
```

**Giải pháp:**

1. Use compression:

```python
ssh_to_vm('cd ~/traffic-forecast/data && tar -czf downloads.tar.gz downloads/')
download_to_vm('~/traffic-forecast/data/downloads.tar.gz', 'data/')
```

2. Use gsutil (faster for large files):

```bash
# On VM
gsutil cp -r ~/traffic-forecast/data/downloads gs://your-bucket/

# On local
gsutil cp -r gs://your-bucket/downloads data/
```

---

## Checklist: Deployment Hoàn Tất

Sau khi hoàn thành tất cả các bước, verify:

- [ ] GCP project created và billing enabled
- [ ] APIs enabled (Compute Engine, Directions API)
- [ ] API key configured với IP restrictions
- [ ] VM instance running (status: RUNNING)
- [ ] Project files uploaded và extracted
- [ ] Python environment installed và tested
- [ ] Cron job configured (hourly collection)
- [ ] First collection run successful (100% success rate)
- [ ] Logs accessible và monitoring working
- [ ] Data downloadable từ VM về local

**System Status:**PRODUCTION READY

---

## Next Steps

### Immediate (First 24 Hours)

1. **Monitor first 24 hours:**

- Check logs every 2-3 hours: `check_collection_logs()`
- Verify cron job running: `ssh_to_vm('grep CRON /var/log/syslog | tail -n 20')`
- Validate data quality: `validate_vm_collection()`

2. **Setup alerts:**

- Create Cloud Monitoring dashboard
- Alert on API quota usage > 80%
- Alert on collection success rate < 95%

### Short-term (Week 1)

1. **Data analysis:**

- Download first week data: `download_all_runs()`
- Run EDA: Open `DATA_DASHBOARD.ipynb`
- Verify data patterns: Check traffic variations by hour/day

2. **Optimization:**

- Adjust collection frequency based on cost vs data quality
- Fine-tune node selection if needed
- Consider reducing to 15-minute intervals during peak hours only

### Long-term (After Week 1)

1. **Machine Learning:**

- Gather 7+ days of data
- Train models: Open `ML_TRAINING.ipynb`
- Evaluate forecasting accuracy

2. **Production hardening:**

- Setup automated backups to Cloud Storage
- Implement error recovery mechanisms
- Add data quality monitoring

---

## Cost Optimization Tips

### For Academic Projects (7 days)

- **Free tier eligible:** e2-micro VM ($0/month)
- **Main cost:**Directions API (~$197 for 7 days)
- **Optimization:**
- Collect only during daytime (6 AM - 10 PM) → Save 33% ($132)
- Reduce to 2-hour intervals → Save 50% ($98)
- Collect only weekdays → Save 28% ($142)

### For Long-term Production

- **Preemptible VM:**Save 80% on compute (not recommended for 24/7 collection)
- **Committed use discounts:**Save 57% on VM if running > 1 year
- **Nearline storage:**Move old data to cheaper storage ($0.01/GB vs $0.02/GB)
- **BigQuery export:**Analyze data in BigQuery (cheaper than keeping raw JSON)

---

## Support

### Documentation

- Main README: `/doc/v5/README_V5.md`
- Vietnamese guide: `/doc/v5/HUONG_DAN_V5.md`
- Architecture: `/doc/v5/BAO_CAO_CAI_TIEN_V5.md`

### Tools

- Local control: `notebooks/CONTROL_PANEL.ipynb`
- GCP management: `notebooks/GCP_DEPLOYMENT.ipynb` (this file)
- Data analysis: `notebooks/DATA_DASHBOARD.ipynb`

### Contact

- Project: DSP391m - Data Science Project
- Version: 5.0.0
- Platform: Google Cloud Platform

---

**Deployment Complete! Your Traffic Forecast v5.0 system is now running on GCP!**
