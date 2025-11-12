#!/usr/bin/env python3
"""Fix NOT OK spacing issues"""

from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).parent.parent.parent

def fix_not_ok_spacing(text):
    """Add space after NOT OK"""
    # Fix "NOT OK" followed immediately by capital letter or text
    text = re.sub(r'NOT OK([A-Z])', r'NOT OK \1', text)
    text = re.sub(r'NOT OK([a-z])', r'NOT OK \1', text)
    return text

def process_file(filepath):
    """Process a single file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    content = fix_not_ok_spacing(content)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {filepath.name}")
        return True
    return False

def main():
    dashboard_dir = PROJECT_ROOT / "dashboard"
    files = list(dashboard_dir.glob("*.py")) + list((dashboard_dir / "pages").glob("*.py"))
    
    fixed = 0
    for f in files:
        if process_file(f):
            fixed += 1
    
    print(f"\nFixed {fixed} files")

if __name__ == "__main__":
    main()
