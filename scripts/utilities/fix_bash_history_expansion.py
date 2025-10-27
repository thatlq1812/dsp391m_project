#!/usr/bin/env python3
"""
Fix bash history expansion issue in all shell scripts.
Adds 'set +H' after shebang to prevent '!' from triggering history expansion.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def fix_script(filepath):
    """Add 'set +H' after shebang if not already present."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Cannot read {filepath}: {e}")
        return False

    if not lines:
        return False

    # Check if file starts with bash shebang
    if not lines[0].startswith('#!/bin/bash') and not lines[0].startswith('#!/usr/bin/env bash'):
        return False

    # Check if 'set +H' already exists in first 10 lines
    for i in range(min(10, len(lines))):
        if 'set +H' in lines[i]:
            print(f"✓ Already fixed: {filepath.relative_to(PROJECT_ROOT)}")
            return False

    # Find insertion point (after shebang and any immediate comments)
    insert_idx = 1
    
    # Skip header comments that immediately follow shebang
    while insert_idx < len(lines) and lines[insert_idx].strip().startswith('#'):
        insert_idx += 1

    # Insert 'set +H' with appropriate formatting
    if insert_idx < len(lines) and lines[insert_idx].strip() == '':
        # If there's a blank line, insert before it
        lines.insert(insert_idx, 'set +H  # Disable history expansion to avoid "event ! not found" errors\n')
    else:
        # Insert after comments with blank line
        lines.insert(insert_idx, '\nset +H  # Disable history expansion to avoid "event ! not found" errors\n')

    # Write back
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"✅ Fixed: {filepath.relative_to(PROJECT_ROOT)}")
        return True
    except Exception as e:
        print(f"❌ Cannot write {filepath}: {e}")
        return False


def main():
    """Fix all bash scripts in the project."""
    print("🔍 Scanning for bash scripts...\n")
    
    fixed_count = 0
    total_count = 0
    
    # Find all .sh files in scripts directory
    for sh_file in SCRIPTS_DIR.rglob('*.sh'):
        total_count += 1
        if fix_script(sh_file):
            fixed_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   Total bash scripts: {total_count}")
    print(f"   Fixed: {fixed_count}")
    print(f"   Already OK: {total_count - fixed_count}")
    
    if fixed_count > 0:
        print(f"\n✅ All scripts fixed! No more 'event ! not found' errors.")
    else:
        print(f"\n✓ All scripts already have history expansion disabled.")


if __name__ == '__main__':
    main()
