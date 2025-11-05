"""
Page 12: VM Management
Guided commands for managing the Google Cloud VM that hosts collectors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from dashboard.utils.command_blocks import show_command_block, show_command_list

st.set_page_config(page_title="VM Management", page_icon="", layout="wide")

st.title("Google Cloud VM Management")
st.markdown("Prepare gcloud commands and review configuration details.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "vm_config.json"

DEFAULT_CONFIG = {
    "project_id": "sonorous-nomad-476606-g3",
    "zone": "asia-southeast1-a",
    "instance_name": "traffic-forecast-collector",
    "ssh_user": os.environ.get("USER", "user"),
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            config = json.loads(CONFIG_PATH.read_text())
            return {
                "project_id": config.get("gcp", {}).get("project_id", DEFAULT_CONFIG["project_id"]),
                "zone": config.get("gcp", {}).get("zone", DEFAULT_CONFIG["zone"]),
                "instance_name": config.get("vm", {}).get("instance_name", DEFAULT_CONFIG["instance_name"]),
                "ssh_user": config.get("vm", {}).get("ssh_user", DEFAULT_CONFIG["ssh_user"]),
            }
        except json.JSONDecodeError:
            st.warning("vm_config.json is invalid JSON. Using default values.")
    return DEFAULT_CONFIG


CONFIG = load_config()

TAB_INSTANCE, TAB_MONITOR, TAB_SSH, TAB_CONFIG = st.tabs(
    [
        "Instance Control",
        "Monitoring",
        "SSH",
        "Configuration",
    ]
)

with TAB_INSTANCE:
    st.markdown("### Instance Control")
    st.info("Ensure gcloud CLI is installed and authenticated (`gcloud auth login`).")

    if st.button("Check status", type="primary", width='stretch'):
        show_command_block(
            [
                "gcloud",
                "compute",
                "instances",
                "describe",
                CONFIG["instance_name"],
                f"--zone={CONFIG['zone']}",
                "--format=json(status,networkInterfaces[0].accessConfigs[0].natIP)",
            ],
            cwd=PROJECT_ROOT,
            description="Retrieve VM status and external IP.",
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start VM", width='stretch'):
            show_command_block(
                [
                    "gcloud",
                    "compute",
                    "instances",
                    "start",
                    CONFIG["instance_name"],
                    f"--zone={CONFIG['zone']}",
                ],
                cwd=PROJECT_ROOT,
                description="Start the VM instance.",
            )
    with col2:
        if st.button("Stop VM", width='stretch'):
            show_command_block(
                [
                    "gcloud",
                    "compute",
                    "instances",
                    "stop",
                    CONFIG["instance_name"],
                    f"--zone={CONFIG['zone']}",
                ],
                cwd=PROJECT_ROOT,
                description="Stop the VM to save costs.",
            )

    st.divider()
    st.markdown("#### Serial Console (debug)")
    if st.button("Open serial console", width='stretch'):
        show_command_block(
            [
                "gcloud",
                "compute",
                "connect-to-serial-port",
                CONFIG["instance_name"],
                f"--zone={CONFIG['zone']}",
            ],
            cwd=PROJECT_ROOT,
            description="Connect to the VM serial console for debugging boot issues.",
        )

with TAB_MONITOR:
    st.markdown("### Monitoring Links")
    st.info("Use Google Cloud Monitoring for CPU, memory, and disk metrics.")
    st.markdown(
        f"- [Compute Instance](https://console.cloud.google.com/compute/instancesDetail/zones/{CONFIG['zone']}/instances/{CONFIG['instance_name']}?project={CONFIG['project_id']})\n"
        f"- [Cloud Monitoring](https://console.cloud.google.com/monitoring/dashboards/resourceList/Compute%20Engine?project={CONFIG['project_id']})"
    )

    if st.button("Prepare metrics export", width='stretch'):
        show_command_block(
            [
                "gcloud",
                "monitoring",
                "time-series",
                "list",
                f"--project={CONFIG['project_id']}",
                "--filter",
                'metric.type="compute.googleapis.com/instance/cpu/utilization"',
                "--limit",
                "5",
            ],
            cwd=PROJECT_ROOT,
            description="Sample command to pull recent CPU metrics from Cloud Monitoring.",
        )

with TAB_SSH:
    st.markdown("### SSH & Remote Commands")
    ssh_target = f"{CONFIG['ssh_user']}@{CONFIG['instance_name']}"
    st.code(f"gcloud compute ssh {ssh_target} --zone={CONFIG['zone']}", language="bash")
    quick_cmd = st.selectbox(
        "Helper command",
        [
            "df -h",
            "free -h",
            "top -b -n 1 | head",
            "journalctl -u traffic-collector -n 100",
            "sudo systemctl status traffic-collector",
        ],
    )
    if st.button("Prepare remote execution", width='stretch'):
        show_command_block(
            [
                "gcloud",
                "compute",
                "ssh",
                ssh_target,
                f"--zone={CONFIG['zone']}",
                "--command",
                quick_cmd,
            ],
            cwd=PROJECT_ROOT,
            description="Execute a single command over SSH (requires gcloud CLI).",
        )

with TAB_CONFIG:
    st.markdown("### Config File")
    if CONFIG_PATH.exists():
        st.code(CONFIG_PATH.read_text(), language="json")
    else:
        st.info("`configs/vm_config.json` not found. Create it to customise VM details.")

    st.markdown("#### Update instructions")
    st.code(
        """
# Example structure for configs/vm_config.json
{
  "gcp": {
    "project_id": "sonorous-nomad-476606-g3",
    "zone": "asia-southeast1-a"
  },
  "vm": {
    "instance_name": "traffic-forecast-collector",
    "ssh_user": "collector"
  },
  "github": {
    "repo": "https://github.com/thatlq1812/dsp391m_project.git",
    "branch": "master",
    "remote_path": "~/dsp391m_project"
  }
}
        """.strip(),
        language="json",
    )

st.divider()
st.caption("Tip: stop the VM when not collecting data to avoid unnecessary cloud spend.")
