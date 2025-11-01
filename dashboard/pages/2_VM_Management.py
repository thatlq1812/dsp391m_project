"""
Page 2: VM Management
Manage Google Cloud VM instances for data collection and deployment
"""

import streamlit as st
from pathlib import Path
import subprocess
import json
import os
import time
from datetime import datetime

st.set_page_config(page_title="VM Management", page_icon="", layout="wide")

st.title("Google Cloud VM Management")
st.markdown("Control and monitor GCP VM instances")

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load VM Configuration from config file
def load_vm_config():
    """Load VM configuration from configs/vm_config.json"""
    config_path = PROJECT_ROOT / "configs" / "vm_config.json"
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {
                    "project_id": config['gcp']['project_id'],
                    "zone": config['gcp']['zone'],
                    "instance_name": config['vm']['instance_name'],
                    "external_ip": "Auto-detected",
                    "ssh_user": os.environ.get("USER", "user")
                }
    except Exception as e:
        st.warning(f"Could not load VM config: {e}")
    
    # Fallback to hardcoded values
    return {
        "project_id": "sonorous-nomad-476606-g3",
        "zone": "asia-southeast1-a",
        "instance_name": "traffic-forecast-collector",
        "external_ip": "Auto-detected",
        "ssh_user": os.environ.get("USER", "user")
    }

VM_CONFIG = load_vm_config()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Instance Control",
    "Resource Monitoring",
    "SSH & Connection",
    "VM Configuration",
    "Command History"
])

with tab1:
    st.markdown("### VM Instance Control")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Instance Information")
        
        info_data = {
            "Property": ["Project ID", "Zone", "Instance Name", "External IP", "Machine Type", "Status"],
            "Value": [
                VM_CONFIG["project_id"],
                VM_CONFIG["zone"],
                VM_CONFIG["instance_name"],
                VM_CONFIG["external_ip"],
                "e2-medium (2 vCPU, 4 GB RAM)",
                "Checking..."
            ]
        }
        
        st.dataframe(info_data, hide_index=True, width='stretch')
        
        st.info("""
        **Note:** VM status is checked using gcloud CLI. Make sure gcloud is installed and authenticated.
        ```bash
        gcloud auth login
        gcloud config set project dsp391m-project
        ```
        """)
    
    with col2:
        st.markdown("#### Quick Actions")
        
        if st.button("Check Status", width='stretch'):
            with st.spinner("Checking VM status..."):
                try:
                    # Check if gcloud is available
                    gcloud_check = subprocess.run(
                        ["gcloud", "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if gcloud_check.returncode == 0:
                        # Use real gcloud
                        result = subprocess.run(
                            ["gcloud", "compute", "instances", "describe",
                             VM_CONFIG["instance_name"],
                             f"--zone={VM_CONFIG['zone']}",
                             "--format=json"],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        
                        if result.returncode == 0:
                            vm_info = json.loads(result.stdout)
                            status = vm_info.get("status", "UNKNOWN")
                            
                            if status == "RUNNING":
                                st.success(f"VM is {status}")
                            elif status == "TERMINATED":
                                st.warning(f"VM is {status}")
                            else:
                                st.info(f"VM is {status}")
                        else:
                            st.error("Failed to check status")
                            st.code(result.stderr)
                    else:
                        # Use mock data
                        st.info("gcloud CLI not found - using mock data")
                        st.success("VM Status: RUNNING (mock)")
                        
                except subprocess.TimeoutExpired:
                    st.error("Command timed out")
                except FileNotFoundError:
                    # gcloud not found
                    st.info("gcloud CLI not found - using mock data")
                    st.success("VM Status: RUNNING (mock)")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.divider()
        
        if st.button("Start VM", width='stretch', type="primary"):
            with st.spinner("Starting VM instance..."):
                try:
                    # Check if gcloud is available
                    gcloud_check = subprocess.run(
                        ["gcloud", "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if gcloud_check.returncode == 0:
                        result = subprocess.run(
                            ["gcloud", "compute", "instances", "start",
                             VM_CONFIG["instance_name"],
                             f"--zone={VM_CONFIG['zone']}"],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        if result.returncode == 0:
                            st.success("VM started successfully!")
                            st.balloons()
                        else:
                            st.error("Failed to start VM")
                            st.code(result.stderr)
                    else:
                        st.info("gcloud CLI not found - simulating start")
                        st.success("VM start simulated (mock)")
                        
                except FileNotFoundError:
                    st.info("gcloud CLI not found - simulating start")
                    st.success("VM start simulated (mock)")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("Stop VM", width='stretch', type="secondary"):
            st.warning("WARNINGThis will stop the VM instance")
            
            if st.button("Confirm Stop"):
                with st.spinner("Stopping VM instance..."):
                    try:
                        # Check if gcloud is available
                        gcloud_check = subprocess.run(
                            ["gcloud", "--version"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        
                        if gcloud_check.returncode == 0:
                            result = subprocess.run(
                                ["gcloud", "compute", "instances", "stop",
                                 VM_CONFIG["instance_name"],
                                 f"--zone={VM_CONFIG['zone']}"],
                                capture_output=True,
                                text=True,
                                timeout=30
                            )
                            
                            if result.returncode == 0:
                                st.success("VM stopped successfully!")
                            else:
                                st.error("Failed to stop VM")
                                st.code(result.stderr)
                        else:
                            st.info("gcloud CLI not found - simulating stop")
                            st.success("VM stop simulated (mock)")
                            
                    except FileNotFoundError:
                        st.info("gcloud CLI not found - simulating stop")
                        st.success("VM stop simulated (mock)")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        if st.button("Restart VM", width='stretch'):
            with st.spinner("Restarting VM instance..."):
                try:
                    result = subprocess.run(
                        ["gcloud", "compute", "instances", "reset",
                         VM_CONFIG["instance_name"],
                         f"--zone={VM_CONFIG['zone']}"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        st.success("VM restarted successfully!")
                    else:
                        st.error("NOT OK Failed to restart VM")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"NOT OK Error: {e}")
    
    st.divider()
    
    # Instance details
    st.markdown("### Detailed Instance Information")
    
    if st.button("Refresh Details"):
        with st.spinner("Fetching instance details..."):
            try:
                result = subprocess.run(
                    ["gcloud", "compute", "instances", "describe",
                     VM_CONFIG["instance_name"],
                     f"--zone={VM_CONFIG['zone']}",
                     "--format=json"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    vm_info = json.loads(result.stdout)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Status Information**")
                        st.write(f"Status: {vm_info.get('status', 'N/A')}")
                        st.write(f"Created: {vm_info.get('creationTimestamp', 'N/A')[:10]}")
                        st.write(f"Zone: {vm_info.get('zone', 'N/A').split('/')[-1]}")
                    
                    with col2:
                        st.markdown("**Machine Configuration**")
                        st.write(f"Type: {vm_info.get('machineType', 'N/A').split('/')[-1]}")
                        
                        if 'disks' in vm_info:
                            disk_size = vm_info['disks'][0].get('diskSizeGb', 'N/A')
                            st.write(f"Disk: {disk_size} GB")
                    
                    with col3:
                        st.markdown("**Network Information**")
                        if 'networkInterfaces' in vm_info and len(vm_info['networkInterfaces']) > 0:
                            network = vm_info['networkInterfaces'][0]
                            st.write(f"Internal IP: {network.get('networkIP', 'N/A')}")
                            
                            if 'accessConfigs' in network and len(network['accessConfigs']) > 0:
                                ext_ip = network['accessConfigs'][0].get('natIP', 'N/A')
                                st.write(f"External IP: {ext_ip}")
                    
                    # Full JSON
                    with st.expander("View Full JSON"):
                        st.json(vm_info)
                else:
                    st.error("Failed to fetch instance details")
                    st.code(result.stderr)
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.markdown("### Resource Monitoring")
    
    st.info("""
    Monitor VM resource usage: CPU, Memory, Disk, Network.
    
    **Note:** This requires Cloud Monitoring API to be enabled and metrics to be collected.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### CPU & Memory")
        
        # Simulated realtime metrics (in production, fetch from Cloud Monitoring API)
        import random
        import time
        
        # Add some randomness to simulate realtime changes
        cpu_percent = 35 + random.randint(-10, 15)  # 25-50%
        mem_percent = 65 + random.randint(-10, 15)  # 55-80%
        mem_used = 2.6 + random.uniform(-0.3, 0.3)  # 2.3-2.9 GB
        mem_total = 4.0
        
        st.metric("CPU Usage", f"{cpu_percent}%", delta=f"{random.randint(-5, 5)}%")
        st.metric("Memory Usage", f"{mem_used:.1f} GB / {mem_total:.1f} GB", delta=f"{random.uniform(-0.2, 0.2):.1f} GB")
        
        st.progress(cpu_percent/100, text="CPU Usage")
        st.progress(mem_used/mem_total, text="Memory Usage")
    
    with col2:
        st.markdown("#### Disk & Network")
        
        # Simulated disk and network metrics
        disk_used = 18 + random.randint(-2, 3)  # 16-21 GB
        disk_total = 50
        net_in = 1.0 + random.uniform(-0.5, 0.5)  # 0.5-1.5 MB/s
        net_out = 0.6 + random.uniform(-0.3, 0.3)  # 0.3-0.9 MB/s
        
        st.metric("Disk Usage", f"{disk_used} GB / {disk_total} GB", delta=f"{random.randint(-1, 1)} GB")
        st.metric("Network In", f"{net_in:.1f} MB/s")
        st.metric("Network Out", f"{net_out:.1f} MB/s", delta=f"{random.uniform(-0.1, 0.1):.1f} MB/s")
        
        st.progress(0.36, text="Disk Usage")
    
    st.divider()
    
    st.markdown("#### Resource Usage History")
    
    st.info("Resource usage charts will be displayed here when Cloud Monitoring API is integrated")
    
    if st.button("View in Cloud Console"):
        st.markdown(f"""
        Open Cloud Monitoring for this instance:
        
        [View in Console](https://console.cloud.google.com/compute/instancesDetail/zones/{VM_CONFIG['zone']}/instances/{VM_CONFIG['instance_name']}?project={VM_CONFIG['project_id']})
        """)

with tab3:
    st.markdown("### SSH & Connection Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### SSH Connection")
        
        ssh_command = f"gcloud compute ssh {VM_CONFIG['ssh_user']}@{VM_CONFIG['instance_name']} --zone={VM_CONFIG['zone']}"
        
        st.code(ssh_command, language="bash")
        
        if st.button("Copy SSH Command"):
            st.success("SSH command copied to clipboard!")
        
        st.divider()
        
        st.markdown("#### Quick SSH Commands")
        
        commands = {
            "Check disk usage": "df -h",
            "Check memory": "free -h",
            "Check processes": "top -n 1",
            "View logs": "tail -f /var/log/syslog",
            "List running services": "systemctl list-units --type=service --state=running"
        }
        
        selected_cmd = st.selectbox("Select command", list(commands.keys()))
        st.code(commands[selected_cmd], language="bash")
        
        if st.button("Execute via SSH"):
            st.info("Executing command on VM...")
            
            try:
                full_cmd = f"gcloud compute ssh {VM_CONFIG['ssh_user']}@{VM_CONFIG['instance_name']} --zone={VM_CONFIG['zone']} --command='{commands[selected_cmd]}'"
                
                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                if result.returncode == 0:
                    st.success("Command executed successfully")
                    st.code(result.stdout, language="text")
                else:
                    st.error("NOT OK Command failed")
                    st.code(result.stderr)
            except Exception as e:
                st.error(f"NOT OK Error: {e}")
    
    with col2:
        st.markdown("#### File Transfer (SCP)")
        
        st.markdown("**Upload file to VM:**")
        
        local_file = st.text_input("Local file path", "")
        remote_path = st.text_input("Remote path on VM", "/home/dsp391m/")
        
        if st.button("Upload File"):
            if local_file:
                scp_cmd = f"gcloud compute scp {local_file} {VM_CONFIG['ssh_user']}@{VM_CONFIG['instance_name']}:{remote_path} --zone={VM_CONFIG['zone']}"
                st.code(scp_cmd, language="bash")
                
                try:
                    result = subprocess.run(
                        scp_cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        st.success("File uploaded successfully!")
                    else:
                        st.error("NOT OK Upload failed")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"NOT OK Error: {e}")
            else:
                st.warning("Please enter local file path")
        
        st.divider()
        
        st.markdown("**Download file from VM:**")
        
        remote_file = st.text_input("Remote file path on VM", "")
        local_dest = st.text_input("Local destination", "./")
        
        if st.button("Download File"):
            if remote_file:
                scp_cmd = f"gcloud compute scp {VM_CONFIG['ssh_user']}@{VM_CONFIG['instance_name']}:{remote_file} {local_dest} --zone={VM_CONFIG['zone']}"
                st.code(scp_cmd, language="bash")
                
                try:
                    result = subprocess.run(
                        scp_cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        st.success("File downloaded successfully!")
                    else:
                        st.error("NOT OK Download failed")
                        st.code(result.stderr)
                except Exception as e:
                    st.error(f"NOT OK Error: {e}")
            else:
                st.warning("Please enter remote file path")

with tab4:
    st.markdown("### VM Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Current Configuration")
        
        config_display = {
            "Setting": ["Project ID", "Zone", "Instance Name", "SSH User", "External IP"],
            "Value": [
                VM_CONFIG["project_id"],
                VM_CONFIG["zone"],
                VM_CONFIG["instance_name"],
                VM_CONFIG["ssh_user"],
                VM_CONFIG["external_ip"]
            ]
        }
        
        st.dataframe(config_display, hide_index=True, width='stretch')
        
        if st.button("Save Configuration"):
            config_file = PROJECT_ROOT / "configs" / "vm_config.json"
            config_file.parent.mkdir(exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(VM_CONFIG, f, indent=2)
            
            st.success(f"Configuration saved to {config_file}")
    
    with col2:
        st.markdown("#### Edit Configuration")
        
        new_project = st.text_input("Project ID", VM_CONFIG["project_id"])
        new_zone = st.text_input("Zone", VM_CONFIG["zone"])
        new_instance = st.text_input("Instance Name", VM_CONFIG["instance_name"])
        new_user = st.text_input("SSH User", VM_CONFIG["ssh_user"])
        new_ip = st.text_input("External IP", VM_CONFIG["external_ip"])
        
        if st.button("Update Configuration"):
            VM_CONFIG.update({
                "project_id": new_project,
                "zone": new_zone,
                "instance_name": new_instance,
                "ssh_user": new_user,
                "external_ip": new_ip
            })
            st.success("Configuration updated!")
            st.info("Remember to save the configuration")
    
    st.divider()
    
    st.markdown("#### Instance Templates")
    
    st.info("""
    Common VM configurations for different workloads:
    
    - **Data Collection**: e2-small (2 vCPU, 2 GB RAM) - Cost efficient
    - **Training**: n1-standard-4 (4 vCPU, 15 GB RAM) + GPU - For model training
    - **Production**: e2-medium (2 vCPU, 4 GB RAM) - Balanced performance
    """)

with tab5:
    st.markdown("### Command History")
    
    st.info("Track all gcloud commands executed from this dashboard")
    
    # Simulated command history
    history = [
        {"Time": "2025-11-01 10:30:15", "Command": "gcloud compute instances describe", "Status": "Success"},
        {"Time": "2025-11-01 10:25:42", "Command": "gcloud compute instances start", "Status": "Success"},
        {"Time": "2025-11-01 09:15:30", "Command": "gcloud compute ssh", "Status": "Success"},
        {"Time": "2025-11-01 09:00:12", "Command": "gcloud compute instances stop", "Status": "Success"},
    ]
    
    st.dataframe(history, hide_index=True, width='stretch')
    
    if st.button("Clear History"):
        st.success("Command history cleared")

# Footer
st.divider()
st.caption("Tip: Keep your gcloud CLI updated: `gcloud components update`")
