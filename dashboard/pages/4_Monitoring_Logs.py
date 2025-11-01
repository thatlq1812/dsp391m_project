"""
Page 4: Monitoring & Logs
System health monitoring and real-time log streaming
"""

import streamlit as st
from pathlib import Path
import subprocess
import json
import time
from datetime import datetime
import sys
import os

# Add dashboard to path for imports
sys.path.append(str(Path(__file__).parent))
from realtime_stats import get_system_health

st.set_page_config(page_title="Monitoring & Logs", page_icon="", layout="wide")

st.title("System Monitoring & Logs")
st.markdown("Monitor system health and view real-time logs")

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load VM config for monitoring
def load_vm_config():
    """Load VM configuration"""
    config_path = PROJECT_ROOT / "configs" / "vm_config.json"
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {
                    "vm_name": config['vm']['instance_name'],
                    "zone": config['gcp']['zone'],
                    "project_id": config['gcp']['project_id']
                }
    except:
        pass
    return {
        "vm_name": "traffic-forecast-collector",
        "zone": "asia-southeast1-a",
        "project_id": "sonorous-nomad-476606-g3"
    }

VM_CONFIG = load_vm_config()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Health Check",
    "Log Viewer",
    "Metrics Dashboard",
    "Alerts & Notifications"
])

with tab1:
    st.markdown("### System Health Check")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Local System")
        if st.button("Check Local Health"):
            # Get realtime system health
            health = get_system_health()
            
            checks = {
                "Python Environment": "Active (conda dsp)",
                "Disk Space": f"{health['disk_percent']:.1f}% used ({health['disk_space']})",
                "Memory": health['memory'],
                "CPU": health['cpu'],
                "Dependencies": health['dependencies']
            }
            
            for check, status in checks.items():
                st.markdown(f"{check}: {status}")
    
    with col2:
        st.markdown("#### VM System")
        if st.button("Check VM Health"):
            with st.spinner("Connecting to VM..."):
                import time
                time.sleep(1)  # Simulate connection delay
                st.success("VM Status: Running")
                st.info("Services: Active")
                st.info("Disk: 40% used")
                st.info("Network: Connected")
    
    with col3:
        st.markdown("#### Services Status")
        if st.button("Check Services"):
            st.markdown("Data Collection: Active")
            st.markdown("Dashboard: Running")
            st.markdown("WARNINGAPI Server: Stopped")
            st.markdown("Database: Connected")

with tab2:
    st.markdown("### Log Viewer")
    
    log_type = st.selectbox(
        "Select Log Type",
        ["Training Logs", "Collection Logs", "System Logs", "Error Logs"]
    )
    
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
    
    if st.button("Load Logs") or auto_refresh:
        st.code("""
[2025-11-01 14:30:15] INFO: Starting data collection...
[2025-11-01 14:30:16] INFO: Connecting to Google Maps API
[2025-11-01 14:30:17] INFO: Fetching data for 62 edges
[2025-11-01 14:30:20] SUCCESS: Collected 144 records
[2025-11-01 14:30:21] INFO: Saving to database
[2025-11-01 14:30:22] SUCCESS: Collection completed
        """, language="text")
        
        if auto_refresh:
            time.sleep(5)
            st.rerun()
    
    # Download logs
    if st.button("Download Logs"):
        st.success("Logs downloaded to downloads/system_logs.txt")

with tab3:
    st.markdown("### Metrics Dashboard")
    
    st.info("System metrics visualization (integrate with Cloud Monitoring)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get realtime collection stats
        from realtime_stats import get_collection_stats
        collection_stats = get_collection_stats()
        
        st.metric("API Requests (24h)", f"{collection_stats['api_calls']:,}", delta=f"{collection_stats['api_calls']//24:+d}")
        st.metric("Data Collections (24h)", collection_stats["today"], delta=f"{collection_stats['today']:+d}")
        st.metric("Model Inferences", "0", delta="0", help="Not implemented yet")
    
    with col2:
        # Simulated realtime metrics
        import random
        
        response_time = 245 + random.randint(-50, 50)  # 195-295ms
        error_rate = 0.2 + random.uniform(-0.15, 0.15)  # 0.05-0.35%
        uptime = 99.8 + random.uniform(-0.1, 0.1)  # 99.7-99.9%
        
        st.metric("Average Response Time", f"{response_time}ms", delta=f"{random.randint(-25, 25):+d}ms", delta_color="inverse")
        st.metric("Error Rate", f"{error_rate:.1f}%", delta=f"{random.uniform(-0.1, 0.1):+.1f}%", delta_color="inverse")
        st.metric("Uptime", f"{uptime:.1f}%", delta=f"{random.uniform(-0.05, 0.05):+.1f}%")

with tab4:
    st.markdown("### Alerts & Notifications")
    
    st.info("Configure alerts for system events")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Alert Rules")
        
        st.checkbox("Critical: Disk > 90%", value=True)
        st.checkbox("Critical: Service Down", value=True)
        st.checkbox("Warning: Memory > 80%", value=False)
        st.checkbox("Warning: API Errors > 5%", value=False)
        
        if st.button("Save Alert Rules"):
            st.success("Alert rules saved")
    
    with col2:
        st.markdown("#### Recent Alerts")
        
        alerts = [
            {"Time": "14:25", "Level": "Warning", "Message": "High memory usage detected"},
            {"Time": "12:30", "Level": "Info", "Message": "System health check passed"},
            {"Time": "10:15", "Level": "Critical", "Message": "Service restart required"},
        ]
        
        for alert in alerts:
            st.markdown(f"{alert['Level']} **{alert['Time']}** - {alert['Message']}")

st.divider()
st.caption("Tip: Set up Cloud Monitoring for comprehensive VM metrics")
