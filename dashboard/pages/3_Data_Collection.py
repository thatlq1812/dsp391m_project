"""
Page 3: Data Collection
Guide operators through collection scripts, scheduling, and data sync.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from dashboard.utils.command_blocks import show_command_block, show_command_list
from realtime_stats import get_collection_stats, get_recent_collections

st.set_page_config(page_title="Data Collection", page_icon="📡", layout="wide")

st.title("Data Collection Control")
st.markdown("Prepare collection commands, check schedules, and sync data from the VM.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "data"
CONDA_ENV = os.environ.get("CONDA_ENV", "dsp")

# Tabs
CONTROL_TAB, SCHED_TAB, DOWNLOAD_TAB, STATS_TAB = st.tabs(
    [
        "Collection Control",
        "Scheduling",
        "Download from VM",
        "Collection Stats",
    ]
)

with CONTROL_TAB:
    st.markdown("### Quick Commands")

    st.info(
        "Use the widgets below to prepare Shell commands. All commands assume the terminal is opened "
        "at the project root."
    )

    collection_mode = st.radio(
        "Collection Mode",
        [
            "Single Run (no visualization)",
            "Single Run (with visualization)",
            "Interval Loop",
        ],
        index=0,
    )

    if collection_mode == "Interval Loop":
        minutes = st.slider("Interval (minutes)", min_value=1, max_value=120, value=15)
        interval_args = ["--interval", str(minutes * 60)]
        visualize = st.checkbox("Skip visualization (recommended for cron)", value=True)
        if visualize:
            interval_args.append("--no-visualize")
    else:
        interval_args = []
        visualize = collection_mode.endswith("visualization")

    if st.button("Prepare Command", type="primary", width='stretch'):
        if collection_mode.startswith("Single"):
            script = [
                "python",
                "scripts/collect_and_render.py",
                "--once",
            ]
            if not visualize:
                script.append("--no-visualize")
        else:
            script = [
                "python",
                "scripts/collect_and_render.py",
                *interval_args,
            ]

        show_command_list(
            [
                [
                    "conda",
                    "run",
                    "-n",
                    CONDA_ENV,
                    "--no-capture-output",
                    *script,
                ],
                script,
            ],
            description="Run one of the commands below to start collection",
            cwd=PROJECT_ROOT,
        )
        st.success(
            "Command prepared. Use the Conda variant to ensure dependencies are available. "
            "Press Ctrl+C in the terminal to stop collection."
        )

    st.divider()

    st.markdown("#### Helper Scripts")
    st.code(
        """
# Inspect collection configuration
python scripts/tools/show_node_info.py --sample 5

# Validate Google API quota
python tools/test_google_limited.py
        """.strip(),
        language="bash",
    )

with SCHED_TAB:
    st.markdown("### Scheduling Guidelines")
    st.info(
        "Schedule recurring collection with Windows Task Scheduler or cron on Linux. The command below "
        "activates the Conda environment before launching the interval script."
    )

    interval_minutes = st.slider("Interval (minutes)", 5, 180, 60, 5)
    schedule_command = [
        "bash",
        "-lc",
        "source C:/ProgramData/miniconda3/Scripts/activate dsp && "
        f"python scripts/collect_and_render.py --interval {interval_minutes * 60} --no-visualize",
    ]

    if st.button("Copy Scheduler Command", width='stretch'):
        show_command_block(
            schedule_command,
            cwd=PROJECT_ROOT,
            description="Use this command in Task Scheduler (Windows) or crontab (WSL/Linux).",
        )
        st.success("Command prepared. Paste it into your scheduler configuration.")

    st.divider()

    st.markdown("#### Example Cron Entries")
    st.code(
        """
# Every 15 minutes
*/15 * * * * /usr/bin/env bash -lc 'source ~/miniconda3/bin/activate dsp && \
    python scripts/collect_and_render.py --interval 900 --no-visualize'

# Daily collection at 02:00 with visualization output
0 2 * * * /usr/bin/env bash -lc 'source ~/miniconda3/bin/activate dsp && \
    python scripts/collect_and_render.py --once'
        """.strip(),
        language="bash",
    )

with DOWNLOAD_TAB:
    st.markdown("### Sync Data from VM")

    download_option = st.radio(
        "Download Option",
        [
            "All runs (tar.gz)",
            "Only missing runs (tar.gz)",
            "Only missing runs (raw files)",
        ],
        index=1,
    )

    script_map = {
        "All runs (tar.gz)": "download_all.sh",
        "Only missing runs (tar.gz)": "download_missing.sh",
        "Only missing runs (raw files)": "download_missing_uncompressed.sh",
    }

    target_script = script_map[download_option]
    script_path = SCRIPTS_DIR / target_script

    if st.button("Prepare Download Command", type="primary", width='stretch'):
        if not script_path.exists():
            st.error(f"Script not found: {script_path.relative_to(PROJECT_ROOT)}")
        else:
            show_command_block(
                ["bash", str(script_path.relative_to(PROJECT_ROOT))],
                cwd=PROJECT_ROOT,
                description="Sync run data from the VM (requires gcloud/SSH access).",
                success_hint="Ensure the VM is reachable and SSH keys are configured.",
            )
            st.success("Command prepared. Run it in a terminal to start the download.")

    st.divider()

    st.markdown("#### Manual rsync Template")
    st.code(
        """
rsync -avz \
  --progress \
  --exclude '*.tmp' \
  <gcp-username>@<vm-ip>:/home/<gcp-username>/traffic-forecast/data/runs/ \
  data/runs/
        """.strip(),
        language="bash",
    )

with STATS_TAB:
    st.markdown("### Collection Statistics")

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
        st.metric("Errors", stats["errors"], delta=f"{stats['errors']:+d}", delta_color="inverse")
        st.metric("Last Collection", stats["last_collection"])

    st.divider()
    st.markdown("#### Recent Collections")

    recent = get_recent_collections()
    if recent:
        st.dataframe(recent, hide_index=True, width='stretch')
    else:
        st.info("No collection runs recorded yet. Prepare a command in the first tab to begin collecting.")

st.divider()
st.caption("Tip: keep an interval collection running during peak hours, then switch to scheduled runs overnight.")
