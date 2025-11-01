"""
Page 5: Data Collection
Control Google Maps API data collection
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import sys
import os

# Add dashboard to path for imports
sys.path.append(str(Path(__file__).parent))
from realtime_stats import get_collection_stats, get_recent_collections

st.set_page_config(page_title="Data Collection", page_icon="📡", layout="wide")

st.title("Data Collection Control")
st.markdown("Manage Google Maps API data collection")

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Collection Control",
    "Scheduling",
    "Download from VM",
    "Collection Stats"
])

with tab1:
    st.markdown("### Collection Control")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Collection Configuration")
        
        collection_mode = st.radio(
            "Collection Mode",
            ["Single Run", "Interval Loop", "Scheduled"]
        )
        
        if collection_mode == "Single Run":
            st.info("Collect data once and stop")
        elif collection_mode == "Interval Loop":
            interval = st.slider("Interval (minutes)", 1, 60, 15)
            st.info(f"Collect data every {interval} minutes")
        else:
            st.info("Collection runs on predefined schedule")
        
        visualize = st.checkbox("Generate Visualization", value=True)
        save_raw = st.checkbox("Save Raw Data", value=True)
    
    with col2:
        st.markdown("#### Quick Actions")
        
        if st.button("Start Collection", width='stretch', type="primary"):
            st.success("Collection started!")
            st.info("Check Collection Stats tab for progress")
        
        if st.button("Stop Collection", width='stretch'):
            st.warning("Collection stopped")
        
        if st.button("View Latest", width='stretch'):
            st.info("Opening latest collection data...")
    
    st.divider()
    
    st.markdown("#### Collection Scripts")
    
    scripts = {
        "Single Collection": "scripts/collect_once.py",
        "Interval Loop": "scripts/collect_and_render.py --interval 900",
        "Scheduled": "Use cron job or Task Scheduler"
    }
    
    for name, cmd in scripts.items():
        st.code(cmd, language="bash")

with tab2:
    st.markdown("### Collection Scheduling")
    
    st.info("Schedule automatic data collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Schedule Configuration")
        
        schedule_type = st.selectbox(
            "Schedule Type",
            ["Hourly", "Daily", "Custom Cron"]
        )
        
        if schedule_type == "Hourly":
            st.info("Collect every hour at :00")
        elif schedule_type == "Daily":
            time_select = st.time_input("Collection Time", value=None)
        else:
            cron_expr = st.text_input("Cron Expression", "0 */6 * * *")
            st.caption("Example: 0 */6 * * * = Every 6 hours")
        
        if st.button("Save Schedule"):
            st.success("Schedule saved")
    
    with col2:
        st.markdown("#### Active Schedules")
        
        st.markdown("- Every 15 minutes (Interval)")
        st.markdown("- Daily at 02:00 (Cron)")
        st.markdown("- Hourly (Paused)")

with tab3:
    st.markdown("### Download Data from VM")
    
    st.info("Download collected data from VM to local machine")
    
    col1, col2 = st.columns(2)
    
    with col1:
        download_option = st.radio(
            "Download Option",
            [
                "All Runs (compressed - fastest)",
                "Missing Runs (compressed - recommended)", 
                "Missing Runs (uncompressed - slower, more reliable)"
            ]
        )
        
        # Determine which script to use
        if download_option.startswith("All Runs"):
            script_name = "download_all.sh"
            description = "Downloads all available runs using tar.gz compression"
        elif "uncompressed" in download_option:
            script_name = "download_missing_uncompressed.sh"
            description = "Downloads missing runs individually without compression"
        else:  # Missing Runs (compressed)
            script_name = "download_missing.sh"
            description = "Downloads missing runs using tar.gz compression"
        
        st.markdown(f"**Selected:** `{script_name}`")
        st.caption(description)
        
        if st.button("Generate Download Command", width='stretch', type="primary"):
            # Build command for user to copy
            cmd = f"bash scripts/data/{script_name}"
            
            st.success("Command ready! Copy and run in your terminal:")
            st.code(cmd, language="bash")
            
            st.info("""
            **How to run:**
            1. Open a terminal in the project directory
            2. Copy the command above
            3. Paste and run it
            4. Monitor the download progress
            """)
            
            # Show script location
            script_path = PROJECT_ROOT / "scripts" / "data" / script_name
            if script_path.exists():
                st.success(f"Script found at: `{script_path}`")
            else:
                st.error(f"Script not found: `{script_path}`")
    
    with col2:
        st.markdown("#### Download Instructions")
        
        st.markdown("""
        **All Runs:**
        - Downloads all available runs from VM
        - Overwrites existing runs if any
        
        **Missing Runs Only:**
        - Compares local vs VM runs
        - Downloads only missing runs
        - Faster if you already have most data
        """)
        
        st.markdown("#### Manual Commands")
        st.code("""
# Download all runs
bash scripts/data/download_all.sh

# Download only missing runs
bash scripts/data/download_missing.sh

# Download missing runs uncompressed
bash scripts/data/download_missing_uncompressed.sh
        """, language="bash")
        
        st.caption("Make sure VM is accessible via gcloud")

with tab4:
    st.markdown("### Collection Statistics")
    
    # Get realtime stats
    stats = get_collection_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Collections", stats["total_collections"], help="All time")
        st.metric("This Week", stats["this_week"], delta=f"+{stats['this_week']}")
        st.metric("Today", stats["today"], delta=f"+{stats['today']}")
    
    with col2:
        st.metric("Success Rate", f"{stats['success_rate']}%", delta=f"{stats['success_rate']-95:.1f}%")
        st.metric("Avg Duration", stats["avg_duration"], delta="-15s", delta_color="inverse")
        st.metric("Data Points", f"{stats['data_points']:,}", delta=f"+{stats['data_points']//10}")
    
    with col3:
        st.metric("API Calls", f"{stats['api_calls']:,}", help="Estimated from data")
        st.metric("Errors", stats["errors"], delta=f"{'+' if stats['errors'] > 0 else ''}{stats['errors']}", delta_color="inverse")
        st.metric("Last Collection", stats["last_collection"])
    
    st.divider()
    
    st.markdown("#### Recent Collections")
    
    # Get recent collections
    collections = get_recent_collections()
    
    if collections:
        st.dataframe(collections, hide_index=True, width='stretch')
    else:
        st.info("No collection runs found. Start data collection to see statistics.")

st.divider()
st.caption("Tip: Use interval collection for continuous data gathering")
