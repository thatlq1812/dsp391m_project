"""
Page 11: Deployment
Command-first workflow for pushing changes and updating the VM.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from dashboard.utils.command_blocks import show_command_block, show_command_list

st.set_page_config(page_title="Deployment", page_icon="", layout="wide")

st.title("Deployment & Version Control")
st.markdown("Follow the checklists below to push code and update the VM safely.")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_deploy_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "vm_config.json"
    try:
        if config_path.exists():
            return json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        st.warning(f"Deployment config invalid JSON: {exc}")
    return {
        "gcp": {"project_id": "sonorous-nomad-476606-g3", "zone": "asia-southeast1-a"},
        "vm": {"instance_name": "traffic-forecast-collector"},
        "github": {
            "repo": "https://github.com/thatlq1812/dsp391m_project.git",
            "branch": "master",
            "remote_path": "~/traffic-forecast",
        },
    }


def _ssh_user(config: dict) -> str:
    remote_path = config.get("github", {}).get("remote_path", "")
    return remote_path.split("/")[-1] or "USER"


CONFIG = load_deploy_config()

TAB_DEPLOY, TAB_GIT, TAB_HISTORY, TAB_ROLLBACK = st.tabs(
    [
        "Deploy to VM",
        "Git Operations",
        "History",
        "Rollback",
    ]
)

with TAB_DEPLOY:
    st.markdown("### Deployment Checklist")
    branch = st.selectbox(
        "Branch to deploy",
        [CONFIG["github"].get("branch", "master"), "develop", "feature/dashboard-v4"],
        index=0,
    )
    update_deps = st.checkbox("Update dependencies", value=True)
    restart_services = st.checkbox("Restart services", value=True)
    st.markdown(
        "1. Commit and push local changes\n"
        "2. SSH to the VM\n"
        "3. Pull latest code\n"
        "4. (Optional) Update Python dependencies\n"
        "5. Restart services\n"
        "6. Verify dashboard/API"
    )

    st.divider()
    if st.button("Prepare deployment commands", type="primary", width='stretch'):
        ssh_target = f"{_ssh_user(CONFIG)}@<vm-ip>"
        commands = [
            ["git", "push", "origin", branch],
            ["ssh", ssh_target],
        ]
        show_command_list(
            commands,
            cwd=PROJECT_ROOT,
            description="Run the commands below in order. Replace `<vm-ip>` with the actual IP address.",
        )
        st.info("After connecting via SSH, execute the remote commands listed in the Manual section below.")

    st.markdown("#### Remote commands")
    remote_snippet = [
        "cd ~/dsp391m_project",
        f"git pull origin {branch}",
        "conda activate dsp" if update_deps else "# conda activate dsp",
        "pip install -r requirements.txt" if update_deps else "# pip install -r requirements.txt",
        "systemctl restart traffic-collector" if restart_services else "# systemctl restart traffic-collector",
    ]
    st.code("\n".join(remote_snippet), language="bash")

with TAB_GIT:
    st.markdown("### Local Git Shortcuts")
    commit_msg = st.text_input("Commit message", "deploy: update from dashboard")
    if st.button("Prepare commit", width='stretch'):
        show_command_list(
            [["git", "add", "."], ["git", "commit", "-m", commit_msg]],
            cwd=PROJECT_ROOT,
            description="Stage and commit local changes.",
        )
        st.success("Commands prepared.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Git status", width='stretch'):
            show_command_block(
                ["git", "status", "--short"],
                cwd=PROJECT_ROOT,
                description="Inspect working tree status.",
            )
    with col2:
        if st.button("Git log", width='stretch'):
            show_command_block(
                ["git", "log", "--oneline", "-10"],
                cwd=PROJECT_ROOT,
                description="View the latest commits.",
            )

with TAB_HISTORY:
    st.markdown("### Deployment Log (Manual)")
    st.info(
        "Record deployment events in `docs/CHANGELOG.md`. The table below summarises entries copied from the file when available."
    )
    changelog = PROJECT_ROOT / "docs" / "CHANGELOG.md"
    if changelog.exists():
        lines = changelog.read_text().splitlines()
        recent = [line for line in lines if line.startswith("-")][:10]
        if recent:
            st.code("\n".join(recent), language="markdown")
        else:
            st.caption("No bullet entries detected yet.")
    else:
        st.caption("CHANGELOG.md not found.")

with TAB_ROLLBACK:
    st.markdown("### Rollback Instructions")
    if st.button("List recent commits", width='stretch'):
        show_command_block(
            ["git", "log", "--oneline", "-20"],
            cwd=PROJECT_ROOT,
            description="Identify the commit to roll back to.",
        )
    commit_hash = st.text_input("Commit hash to reset to", "")
    if commit_hash:
        st.code(
            f"git reset --hard {commit_hash}\ngit push --force origin {CONFIG['github'].get('branch', 'master')}",
            language="bash",
        )
        st.warning("Force push rewrites history. Coordinate with collaborators before running.")

st.divider()
st.caption("Tip: After deployment, run smoke tests from the API & Integration page to confirm services are healthy.")
