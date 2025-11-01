#!/usr/bin/env python3
"""
Sync VM Configuration from Deployment Scripts

Reads configuration from deployment shell scripts and syncs to vm_config.json
This ensures consistency between deployment scripts and dashboard configuration.
"""

import json
import re
from pathlib import Path

def extract_from_shell_script(script_path, variables):
    """Extract variable values from shell script"""
    config = {}
    
    if not script_path.exists():
        print(f"WARNING  Script not found: {script_path}")
        return config
    
    content = script_path.read_text(encoding='utf-8')
    
    for var in variables:
        # Match pattern: VAR_NAME="value"
        pattern = rf'{var}="([^"]+)"'
        match = re.search(pattern, content)
        if match:
            config[var] = match.group(1)
        else:
            # Try without quotes: VAR_NAME=value
            pattern = rf'{var}=(\S+)'
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                # Remove ${...} if present
                value = re.sub(r'\$\{[^}]+\}', '', value)
                config[var] = value.strip()
    
    return config

def main():
    print("=" * 70)
    print("VM Configuration Sync Tool")
    print("=" * 70)
    print()
    
    project_root = Path(__file__).parent.parent
    
    # Read from deployment script
    deploy_script = project_root / "scripts" / "deployment" / "deploy_git.sh"
    auto_deploy_script = project_root / "scripts" / "deployment" / "auto_deploy_vm.sh"
    
    print("Reading configuration from deployment scripts...")
    print(f"   - {deploy_script.name}")
    print(f"   - {auto_deploy_script.name}")
    print()
    
    # Extract variables
    deploy_vars = extract_from_shell_script(deploy_script, [
        "PROJECT_ID", "ZONE", "VM_NAME"
    ])
    
    auto_deploy_vars = extract_from_shell_script(auto_deploy_script, [
        "PROJECT_ID", "ZONE", "INSTANCE_NAME", "MACHINE_TYPE", 
        "DISK_SIZE", "GITHUB_REPO", "GITHUB_BRANCH"
    ])
    
    # Merge and build config
    config = {
        "comment": "GCP VM Configuration - Auto-synced from deployment scripts",
        "source": "scripts/deployment/*.sh",
        "last_updated": "2025-11-01",
        "gcp": {
            "project_id": deploy_vars.get("PROJECT_ID", auto_deploy_vars.get("PROJECT_ID", "")),
            "zone": deploy_vars.get("ZONE", auto_deploy_vars.get("ZONE", "")),
            "region": "asia-southeast1"
        },
        "vm": {
            "instance_name": deploy_vars.get("VM_NAME", auto_deploy_vars.get("INSTANCE_NAME", "")),
            "machine_type": auto_deploy_vars.get("MACHINE_TYPE", "e2-micro"),
            "disk_size": auto_deploy_vars.get("DISK_SIZE", "30GB"),
            "os": "ubuntu-2204-lts",
            "tags": ["http-server", "https-server"]
        },
        "ssh": {
            "user": "auto-detect",
            "port": 22,
            "key_path": "~/.ssh/google_compute_engine"
        },
        "github": {
            "repo": auto_deploy_vars.get("GITHUB_REPO", ""),
            "branch": auto_deploy_vars.get("GITHUB_BRANCH", "master"),
            "remote_path": "~/traffic-forecast"
        },
        "services": {
            "data_collection": {
                "script": "scripts/collect_and_render.py",
                "interval": 900,
                "log_file": "~/traffic-forecast/logs/collection.log"
            },
            "api_server": {
                "port": 8000,
                "workers": 4,
                "log_file": "~/traffic-forecast/logs/api.log"
            }
        },
        "conda": {
            "env_name": "dsp",
            "python_version": "3.10"
        }
    }
    
    # Display extracted config
    print("Extracted configuration:")
    print(f"   Project ID:    {config['gcp']['project_id']}")
    print(f"   Zone:          {config['gcp']['zone']}")
    print(f"   Instance:      {config['vm']['instance_name']}")
    print(f"   Machine Type:  {config['vm']['machine_type']}")
    print(f"   GitHub Repo:   {config['github']['repo']}")
    print(f"   Branch:        {config['github']['branch']}")
    print()
    
    # Save to vm_config.json
    config_path = project_root / "configs" / "vm_config.json"
    
    print(f"💾 Saving to: {config_path}")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Configuration synced successfully!")
    print()
    print("Dashboard pages will now use these settings:")
    print("  - Page 2: VM Management")
    print("  - Page 3: Deployment")
    print("  - Page 4: Monitoring")
    print()
    print("To update configuration in the future:")
    print("  1. Edit deployment scripts in scripts/deployment/")
    print("  2. Run this script again: python tools/sync_vm_config.py")
    print()

if __name__ == "__main__":
    main()
