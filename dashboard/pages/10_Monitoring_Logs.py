"""
Page 10: Monitoring & Logs
Summarise local diagnostics and point to external monitoring tooling.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from dashboard.utils.command_blocks import show_command_block
from realtime_stats import get_collection_stats, get_system_health

st.set_page_config(page_title="Monitoring & Logs", page_icon="", layout="wide")

st.title("Monitoring & Logs")
st.markdown("Check local health, inspect logs, and jump to cloud dashboards.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"

TAB_HEALTH, TAB_LOGS, TAB_METRICS, TAB_ALERTS = st.tabs(
    [
        "Health Check",
        "Log Viewer",
        "Metrics",
        "Alerts",
    ]
)

with TAB_HEALTH:
    st.markdown("### Local Environment")
    health = get_system_health()
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU", health.get("cpu", "-"))
    col2.metric("Memory", health.get("memory", "-"))
    col3.metric("Disk", health.get("disk_percent", 0), help=health.get("disk_space"))
    st.caption("Values computed from local machine sensors.")

    st.divider()
    st.markdown("### VM Checklist")
    st.info(
        "Use the VM Management page to control instance state. For manual CLI checks, run the command below."
    )
    if st.button("Prepare gcloud describe", width='stretch'):
        show_command_block(
            [
                "gcloud",
                "compute",
                "instances",
                "describe",
                "traffic-forecast-collector",
                "--zone=asia-southeast1-a",
                "--format=json",
            ],
            cwd=PROJECT_ROOT,
            description="Requires gcloud CLI authentication.",
        )

with TAB_LOGS:
    st.markdown("### Local Logs")
    if not LOGS_DIR.exists():
        st.info("The `logs/` directory does not exist. Configure logging in scripts to populate it.")
    else:
        log_files = [p for p in LOGS_DIR.glob("**/*.log")]
        if not log_files:
            st.info("No `.log` files detected yet.")
        else:
            selected = st.selectbox("Select log file", [f.relative_to(PROJECT_ROOT) for f in log_files])
            log_path = PROJECT_ROOT / selected
            if st.button("Tail log file", width='stretch'):
                show_command_block(
                    ["tail", "-n", "200", str(log_path.relative_to(PROJECT_ROOT))],
                    cwd=PROJECT_ROOT,
                    description="View the latest 200 lines of the selected log.",
                )
            if st.button("Open log (read-only)", width='stretch'):
                try:
                    preview = "".join(log_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)[-200:])
                    st.code(preview or "(file empty)", language="text")
                except OSError as exc:
                    st.error(f"Failed to read {log_path}: {exc}")

with TAB_METRICS:
    st.markdown("### Collection Metrics Snapshot")
    stats = get_collection_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Collections (24h)", stats.get("today", 0))
    col2.metric("Collections (7d)", stats.get("this_week", 0))
    col3.metric("Overall", stats.get("total_collections", 0))

    st.divider()
    st.markdown("### Cloud Monitoring (Manual)")
    if st.button("Open Google Cloud Metrics", width='stretch'):
        st.markdown(
            "[Open console](https://console.cloud.google.com/monitoring)"
        )
        st.info("Use the GCP console to configure dashboards for CPU, memory, and API quotas.")

with TAB_ALERTS:
    st.markdown("### Alerting Guidance")
    st.info(
        "Alerts are not automated inside the dashboard. Configure Cloud Monitoring policies or Slack webhooks via the API page."
    )
    st.code(
        json.dumps(
            {
                "policy": "traffic-collector",
                "threshold": {"metric": "compute.googleapis.com/instance/cpu/utilization", "above": 0.8},
                "notification_channels": ["slack"],
            },
            indent=2,
        ),
        language="json",
    )

st.divider()
st.caption("Tip: export structured logs to Cloud Logging for long-term retention and alerting.")
