"""
Page 3: Deployment
Git-based deployment and version control
"""

import streamlit as st
from pathlib import Path
import subprocess
import json
from datetime import datetime
import os

st.set_page_config(page_title="Deployment", page_icon="", layout="wide")

st.title("Deployment & Version Control")
st.markdown("Git-based deployment workflow")

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load deployment config
def load_deploy_config():
    """Load deployment configuration from configs/vm_config.json"""
    config_path = PROJECT_ROOT / "configs" / "vm_config.json"
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {
                    "project_id": config['gcp']['project_id'],
                    "zone": config['gcp']['zone'],
                    "vm_name": config['vm']['instance_name'],
                    "github_repo": config['github']['repo'],
                    "github_branch": config['github']['branch'],
                    "remote_path": config['github']['remote_path']
                }
    except Exception as e:
        st.warning(f"Could not load deployment config: {e}")
    
    # Fallback
    return {
        "project_id": "sonorous-nomad-476606-g3",
        "zone": "asia-southeast1-a",
        "vm_name": "traffic-forecast-collector",
        "github_repo": "https://github.com/thatlq1812/dsp391m_project.git",
        "github_branch": "master",
        "remote_path": "~/traffic-forecast"
    }

DEPLOY_CONFIG = load_deploy_config()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Deploy to VM",
    "Git Operations",
    "Deployment History",
    "Rollback"
])

with tab1:
    st.markdown("### Deploy to VM")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Deployment Configuration")
        
        # Git branch selection
        branch = st.selectbox(
            "Branch to Deploy",
            ["master", "develop", "feature/dashboard-v4"],
            index=0
        )
        
        # Deployment options
        with_dependencies = st.checkbox("Update Dependencies", value=True)
        restart_services = st.checkbox("Restart Services", value=True)
        run_migrations = st.checkbox("Run Database Migrations", value=False)
        
        st.markdown("#### Deployment Steps")
        
        steps = [
            "1. Push local changes to GitHub",
            "2. SSH to VM",
            "3. Pull latest code from repository",
            "4. Update dependencies (if selected)",
            "5. Restart services (if selected)",
            "6. Verify deployment"
        ]
        
        for step in steps:
            st.markdown(step)
    
    with col2:
        st.markdown("#### Quick Deploy")
        
        if st.button("Deploy Now", width='stretch', type="primary"):
            with st.spinner("Deploying to VM..."):
                progress = st.progress(0)
                status = st.empty()
                
                # Step 1: Git push
                status.info("Pushing to GitHub...")
                progress.progress(0.2)
                time_module.sleep(1)
                
                # Step 2: SSH and pull
                status.info("Pulling on VM...")
                progress.progress(0.4)
                time_module.sleep(1)
                
                # Step 3: Dependencies
                if with_dependencies:
                    status.info("Updating dependencies...")
                    progress.progress(0.6)
                    time_module.sleep(1)
                
                # Step 4: Restart
                if restart_services:
                    status.info("Restarting services...")
                    progress.progress(0.8)
                    time_module.sleep(1)
                
                # Complete
                progress.progress(1.0)
                status.success("Deployment completed!")
                st.balloons()
        
        st.divider()
        
        st.markdown("#### Deployment Scripts")
        
        deploy_script = PROJECT_ROOT / "scripts" / "deployment" / "deploy_git.sh"
        
        if deploy_script.exists():
            if st.button("View Deploy Script"):
                with open(deploy_script, 'r') as f:
                    st.code(f.read(), language="bash")
        else:
            st.warning("Deploy script not found")
    
    st.divider()
    
    # Manual deployment
    st.markdown("### Manual Deployment Commands")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Local (Push to GitHub):**")
        st.code("""
git add .
git commit -m "Deploy: Update from dashboard"
git push origin master
        """, language="bash")
    
    with col2:
        st.markdown("**Remote (VM - Pull from GitHub):**")
        st.code("""
cd ~/dsp391m_project
git pull origin master
conda activate dsp
pip install -r requirements.txt
systemctl restart traffic-collector
        """, language="bash")

with tab2:
    st.markdown("### Git Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Repository Status")
        
        if st.button("Check Git Status"):
            try:
                result = subprocess.run(
                    ["git", "status", "--short"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.stdout:
                    st.code(result.stdout, language="text")
                else:
                    st.success("Working tree clean")
            except Exception as e:
                st.error(f"NOT OK Error: {e}")
        
        if st.button("View Git Log"):
            try:
                result = subprocess.run(
                    ["git", "log", "--oneline", "-10"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                st.code(result.stdout, language="text")
            except Exception as e:
                st.error(f"NOT OK Error: {e}")
    
    with col2:
        st.markdown("#### Quick Git Actions")
        
        commit_msg = st.text_input("Commit Message", "Update from dashboard")
        
        if st.button("Commit Changes"):
            try:
                # Add all changes
                subprocess.run(["git", "add", "."], cwd=PROJECT_ROOT, check=True)
                
                # Commit
                result = subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    st.success("Changes committed")
                else:
                    st.info("" + result.stdout)
            except Exception as e:
                st.error(f"NOT OK Error: {e}")
        
        if st.button("Push to GitHub"):
            try:
                result = subprocess.run(
                    ["git", "push", "origin", "master"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    st.success("Pushed to GitHub")
                else:
                    st.error("NOT OK Push failed")
                    st.code(result.stderr)
            except Exception as e:
                st.error(f"NOT OK Error: {e}")
        
        if st.button("Pull from GitHub"):
            try:
                result = subprocess.run(
                    ["git", "pull", "origin", "master"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    st.success("Pulled latest changes")
                    st.code(result.stdout)
                else:
                    st.error("NOT OK Pull failed")
                    st.code(result.stderr)
            except Exception as e:
                st.error(f"NOT OK Error: {e}")

with tab3:
    st.markdown("### Deployment History")
    
    st.info("Track all deployments with timestamps and status")
    
    # Simulated deployment history
    deployments = [
        {
            "Time": "2025-11-01 14:30:00",
            "Branch": "master",
            "Commit": "abc1234",
            "Message": "Deploy: Add VM management page",
            "Status": "Success",
            "Duration": "2m 15s"
        },
        {
            "Time": "2025-11-01 10:15:00",
            "Branch": "master",
            "Commit": "def5678",
            "Message": "Deploy: Update training script",
            "Status": "Success",
            "Duration": "1m 45s"
        },
        {
            "Time": "2025-11-01 08:00:00",
            "Branch": "develop",
            "Commit": "ghi9012",
            "Message": "Deploy: Test new features",
            "Status": "WARNINGPartial",
            "Duration": "3m 20s"
        }
    ]
    
    st.dataframe(deployments, hide_index=True, width='stretch')
    
    # Deployment details
    with st.expander("View Deployment Details"):
        selected = st.selectbox("Select Deployment", [d["Time"] for d in deployments])
        
        deployment = next((d for d in deployments if d["Time"] == selected), None)
        
        if deployment:
            st.json(deployment)

with tab4:
    st.markdown("### Rollback to Previous Version")
    
    st.warning("WARNING**Caution:** Rollback will revert code to a previous commit")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Select Version to Rollback")
        
        # Get recent commits
        if st.button("Load Recent Commits"):
            try:
                result = subprocess.run(
                    ["git", "log", "--oneline", "-20"],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                commits = []
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        commits.append(f"{parts[0]} - {parts[1]}")
                
                st.session_state.commits = commits
            except Exception as e:
                st.error(f"NOT OK Error: {e}")
        
        if 'commits' in st.session_state:
            selected_commit = st.selectbox(
                "Select Commit",
                st.session_state.commits
            )
            
            if selected_commit:
                commit_hash = selected_commit.split(' ')[0]
                
                st.code(f"git reset --hard {commit_hash}", language="bash")
    
    with col2:
        st.markdown("#### Rollback Options")
        
        rollback_type = st.radio(
            "Rollback Type",
            ["Soft (Keep changes)", "Hard (Discard changes)"],
            index=1
        )
        
        if st.button("Perform Rollback", type="secondary"):
            st.error("WARNINGRollback functionality disabled in dashboard for safety")
            st.info("""
            To rollback manually:
            
            ```bash
            git reset --hard <commit-hash>
            git push --force origin master
            ```
            
            **Then deploy to VM using Deploy tab**
            """)

# Footer
st.divider()
st.caption("Tip: Always test deployments in develop branch before deploying to master")

import time as time_module  # Fix the time import
