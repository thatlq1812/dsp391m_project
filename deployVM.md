## Hướng dẫn deploy và quản lý VM thu thập dữ liệu (tổng hợp)

Mục đích: hướng dẫn đầy đủ (bằng lệnh) để tạo một Google Compute Engine VM chạy quy trình thu thập dữ liệu theo chu kỳ 15 phút trong 6 giờ, cách kiểm tra/troubleshoot, và cách cho người khác kết nối tạm thời. Tài liệu này tổng hợp các lệnh và bước đã thảo luận.

Giả định/tổng quan:
- Repository đã có file `startup.sh` (ở root của workspace). Script `startup.sh` trên repo thực hiện: cài Miniconda nếu cần, clone repo vào `/opt/dsp_project`, tạo env `dsp`, chạy vòng thu thập với `timeout 6h bash -lc "conda activate dsp && bash scripts/run_interval.sh 900 --no-visualize"`, rồi shutdown VM.
- Project, zone, instance mẫu trong ví dụ:
  - PROJECT: `shaped-ship-474607-f7`
  - ZONE: `asia-southeast1-a`
  - INSTANCE: `traffic-collector-vm`
  - EXTERNAL IP ví dụ: `34.158.35.53`

## 1. Chuẩn bị file `startup.sh`
- Nếu bạn đã có `startup.sh` trong repo (đã kiểm tra), không cần thay đổi. Nếu muốn sửa, mở `startup.sh` và kiểm tra các phần chính:
  - cài đặt gói cơ bản (`apt-get install`) và Miniconda
  - clone repo vào `/opt/dsp_project`
  - tạo hoặc cập nhật conda env `dsp` (từ `environment.yml` hoặc `requirements.txt`)
  - chạy vòng thu thập với `timeout 6h` cho interval 900s (15 phút)
  - đồng bộ và `shutdown -h now` khi hoàn tất (bạn có thể bỏ dòng shutdown nếu không muốn VM tắt)

## 2. Tạo VM với startup script (gcloud)
Ví dụ tạo VM mới và truyền `startup.sh`:

```bash
gcloud compute instances create traffic-collector-vm \
  --zone=asia-southeast1-a \
  --machine-type=e2-medium \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --metadata-from-file startup-script=C:/path/to/startup.sh \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --project=shaped-ship-474607-f7
```

Nếu bạn muốn cập nhật `startup.sh` đã lưu cục bộ và để VM chạy script đó khi boot lại, dùng:

```bash
gcloud compute instances add-metadata traffic-collector-vm \
  --zone=asia-southeast1-a \
  --metadata-from-file startup-script=C:/path/to/startup.sh \
  --project=shaped-ship-474607-f7

gcloud compute instances reset traffic-collector-vm --zone=asia-southeast1-a --project=shaped-ship-474607-f7
```

## 3. Kiểm tra trạng thái khởi chạy và logs
- Xem serial output (stdout của startup script):

```bash
gcloud compute instances get-serial-port-output traffic-collector-vm --zone=asia-southeast1-a --project=shaped-ship-474607-f7
```

- SSH vào VM để kiểm tra log file (nếu script ghi ra `/var/log/collector-startup.log` hoặc `collector.log`):

```bash
gcloud compute ssh traffic-collector-vm --zone asia-southeast1-a --project shaped-ship-474607-f7
sudo tail -n 200 /var/log/collector-startup.log
sudo tail -n 200 /opt/dsp_project/collector.log
ls -la /opt/dsp_project
```

## 4. Vấn đề thường gặp: "Run 'conda init' before 'conda activate'"
- Nguyên nhân: `conda activate` yêu cầu shell đã được khởi tạo (conda phải được sourced). `conda init` sửa file cấu hình shell cho các shell mới, nhưng phiên hiện tại chưa nhận thay đổi.
- Cách khắc phục/run an toàn:

1) Source file conda trước khi activate:

```bash
source /home/USERNAME/miniconda3/etc/profile.d/conda.sh
conda activate dsp
```

Ví dụ trong `nohup` hoặc `bash -lc`:

```bash
nohup bash -lc "source /home/fxlqt/miniconda3/etc/profile.d/conda.sh && conda activate dsp && timeout 6h bash /opt/dsp_project/scripts/run_interval.sh 900 --no-visualize" > /opt/dsp_project/collector.log 2>&1 &
```

2) Dùng `conda run` (không cần `conda activate`):

```bash
/home/fxlqt/miniconda3/bin/conda run -n dsp bash -lc "timeout 6h /opt/dsp_project/scripts/run_interval.sh 900 --no-visualize"
```

3) Gọi trực tiếp Python từ env (không dùng conda activate):

```bash
/home/fxlqt/miniconda3/envs/dsp/bin/python /opt/dsp_project/scripts/collect_and_render.py --interval 900 --no-visualize
```

## 5. Chạy manual / restart job
- Kill job cũ (theo PID hoặc theo tên):

```bash
ps aux | grep -E 'run_interval|collect_and_render' | grep -v grep
# kill <PID> hoặc
pkill -f run_interval.sh
```

- Start job an toàn (example):

```bash
# source conda and run in background with timeout
nohup bash -lc "source /home/fxlqt/miniconda3/etc/profile.d/conda.sh && conda activate dsp && timeout 6h bash /opt/dsp_project/scripts/run_interval.sh 900 --no-visualize" > /opt/dsp_project/collector.log 2>&1 &

# or using conda run
nohup /home/fxlqt/miniconda3/bin/conda run -n dsp bash -lc "timeout 6h /opt/dsp_project/scripts/run_interval.sh 900 --no-visualize" > /opt/dsp_project/collector.log 2>&1 &
```

## 6. Kiểm tra tiến trình và dữ liệu
- Kiểm tra tiến trình:

```bash
ps aux | grep -E 'run_interval|collect_and_render' | grep -v grep
pgrep -a -f run_interval.sh
```

- Kiểm tra file log output:

```bash
sudo tail -n 200 /opt/dsp_project/collector.log
```

- Kiểm tra folder dữ liệu, file output mới nhất:

```bash
ls -lt /opt/dsp_project/node /opt/dsp_project/data /opt/dsp_project/images 2>/dev/null | head
```

## 7. Cho người khác truy cập VM — các phương án 1 dòng
Lưu ý: VM có External IP (ví dụ `34.158.35.53`) thì có thể kết nối trực tiếp. Dưới đây là các phương án "1-line" tùy nhu cầu.

- Trực tiếp SSH nếu họ có private key:

```bash
ssh -i /path/to/private_key alice@34.158.35.53
```

- Dùng gcloud SSH (nếu đã bật OS Login và user đã có IAM role):

```bash
gcloud compute ssh alice@traffic-collector-vm --zone asia-southeast1-a --project shaped-ship-474607-f7
```

- SSH qua IAP (không cần External IP):

```bash
gcloud compute ssh alice@traffic-collector-vm --zone asia-southeast1-a --project shaped-ship-474607-f7 --tunnel-through-iap
```

- Thêm public SSH key vào metadata (owner chạy, sau đó user SSH):

```bash
# owner: add key (replace the pubkey content)
gcloud compute instances add-metadata traffic-collector-vm --zone asia-southeast1-a --project shaped-ship-474607-f7 --metadata "ssh-keys=alice:ssh-rsa AAAA... alice@example.com"

# user: then
ssh alice@34.158.35.53
```

- Bật OS Login và cấp IAM (quản lý tốt hơn, có thể revoke):

```bash
# enable OS Login on instance
gcloud compute instances add-metadata traffic-collector-vm --zone asia-southeast1-a --project shaped-ship-474607-f7 --metadata enable-oslogin=TRUE

# add IAM role for user (non-admin)
gcloud projects add-iam-policy-binding shaped-ship-474607-f7 --member="user:alice@example.com" --role="roles/compute.osLogin"

# admin (sudo)
gcloud projects add-iam-policy-binding shaped-ship-474607-f7 --member="user:alice@example.com" --role="roles/compute.osAdminLogin"
```

## 8. (Không an toàn) Cho phép password login tạm thời
Chỉ dùng tạm; phải revert sau khi xong.

- 1-liner (owner chạy từ máy local) — tạo user `alice`, đặt password, bật PasswordAuthentication và restart SSH (thay password bằng mật khẩu bạn chọn):

```bash
gcloud compute ssh fxlqt@traffic-collector-vm --zone asia-southeast1-a --project shaped-ship-474607-f7 --command "\
sudo useradd -m -s /bin/bash alice || true; \
echo 'alice:MyTempPass123!' | sudo chpasswd; \
sudo grep -q '^PasswordAuthentication' /etc/ssh/sshd_config && sudo sed -i 's/^PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config || sudo bash -c \"echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config\"; \
sudo grep -q '^ChallengeResponseAuthentication' /etc/ssh/sshd_config && sudo sed -i 's/^ChallengeResponseAuthentication .*/ChallengeResponseAuthentication no/' /etc/ssh/sshd_config || sudo bash -c \"echo 'ChallengeResponseAuthentication no' >> /etc/ssh/sshd_config\"; \
sudo systemctl restart ssh || sudo systemctl restart sshd; echo DONE"
```

- Sau đó người dùng SSH bằng:

```bash
ssh alice@34.158.35.53
```

- Revert khi xong:

```bash
# disable password auth
gcloud compute ssh fxlqt@traffic-collector-vm --zone asia-southeast1-a --project shaped-ship-474607-f7 --command "sudo sed -i 's/^PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config || true; sudo systemctl restart ssh || sudo systemctl restart sshd"

# remove user
gcloud compute ssh fxlqt@traffic-collector-vm --zone asia-southeast1-a --project shaped-ship-474607-f7 --command "sudo deluser --remove-home alice || sudo userdel -r alice || true"

# (optional) delete firewall rule if created to open SSH to all
gcloud compute firewall-rules delete allow-ssh-all --project shaped-ship-474607-f7 --quiet || true
```

## 9. Firewall & hạn chế truy cập
- Kiểm tra rule SSH hiện có:

```bash
gcloud compute firewall-rules list --filter="allowed:tcp:22" --project shaped-ship-474607-f7
```

- Tạo rule chỉ cho IP cụ thể (thay YOUR_IP):

```bash
gcloud compute firewall-rules create allow-ssh-from-my-ip \
  --direction=INGRESS --priority=1000 --network=default \
  --action=ALLOW --rules=tcp:22 \
  --source-ranges=203.0.113.45/32 \
  --target-tags=ssh-server --project=shaped-ship-474607-f7

gcloud compute instances add-tags traffic-collector-vm --tags=ssh-server --zone=asia-southeast1-a --project=shaped-ship-474607-f7
```

## 10. Sao lưu / lấy artifact
- Copy folder dữ liệu về local bằng `gcloud compute scp`:

```bash
gcloud compute scp --recurse traffic-collector-vm:/opt/dsp_project/node ./node-backup --zone=asia-southeast1-a --project=shaped-ship-474607-f7
```

- Hoặc upload từ VM lên GCS (nếu VM có quyền):

```bash
gsutil cp -r /opt/dsp_project/node gs://your-bucket/path/
```

## 11. Kiểm tra nhanh (quality gates)
- Sau deploy/check: 1) kiểm tra serial log; 2) kiểm tra `collector.log` và `node/` hoặc `images/`; 3) kiểm tra tiến trình chạy; 4) chờ 6 giờ để `timeout` tự dừng hoặc giám sát và kill nếu cần.

## 12. Ghi chú & best-practices
- Nếu quy trình hay env nặng: cân nhắc tạo custom image với conda + env đã sẵn sàng để giảm thời gian khởi tạo.
- Tránh password login lâu dài. Nếu nhiều người cần truy cập, bật OS Login + cấp IAM. Nếu không muốn External IP, dùng IAP.
- Luôn backup dữ liệu quan trọng trước khi xóa `/opt/dsp_project` hoặc recreate VM.

## 13. Tóm tắt lệnh hữu ích (copy/paste)

- Tạo instance (ví dụ):
```bash
gcloud compute instances create traffic-collector-vm --zone=asia-southeast1-a --machine-type=e2-medium --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud --boot-disk-size=50GB --metadata-from-file startup-script=C:/path/to/startup.sh --scopes=https://www.googleapis.com/auth/cloud-platform --project=shaped-ship-474607-f7
```

- SSH vào instance và kiểm tra collector log:
```bash
gcloud compute ssh traffic-collector-vm --zone asia-southeast1-a --project shaped-ship-474607-f7 --command "sudo tail -n 200 /opt/dsp_project/collector.log || sudo tail -n 200 /var/log/collector-startup.log"
```

- Start job bằng nohup (ví dụ):
```bash
gcloud compute ssh traffic-collector-vm --zone asia-southeast1-a --project shaped-ship-474607-f7 --command "nohup bash -lc 'source /home/fxlqt/miniconda3/etc/profile.d/conda.sh && conda activate dsp && timeout 6h bash /opt/dsp_project/scripts/run_interval.sh 900 --no-visualize' > /opt/dsp_project/collector.log 2>&1 &"
```

---

Nếu bạn muốn, tôi có thể:
- tạo file `startup.sh` mẫu trong repo (nếu bạn chưa có),
- chạy lệnh add-metadata để thêm public key nếu bạn paste public key ở đây,
- hoặc tạo lệnh 1-line để bật password login tạm (nếu bạn xác nhận username và password).

Hết.
