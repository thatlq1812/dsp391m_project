# Deprecated Scripts
These scripts have been superseded by newer implementations or are no longer needed in the current workflow.
**DO NOT USE** these scripts for new work. They are kept for historical reference only.
---
## Files in this directory
### `add_teammate_access.sh`
- **Status:** Empty placeholder
- **Replaced by:** `scripts/vm_users_setup.sh`
- **Reason:** Never implemented, functionality moved to main setup script
### `check_images.sh`
- **Status:** Not needed
- **Replaced by:** N/A (visualization disabled in production)
- **Reason:** Production runs with `--no-visualize` flag to save resources
### `cleanup.sh`
- **Status:** Deprecated
- **Replaced by:** `scripts/cleanup_runs.py`
- **Reason:** Python version provides better control and dry-run mode
### `cloud_quickref.sh`
- **Status:** Deprecated
- **Replaced by:** `doc/QUICKREF.md`
- **Reason:** Documentation moved to markdown for better formatting
### `deploy_wizard.sh`
- **Status:** Deprecated
- **Replaced by:** `scripts/deploy_week_collection.sh`
- **Reason:** Automated deployment is more reliable than interactive wizard
### `fix_nodes_issue.sh`
- **Status:** Historical fix (Oct 26, 2025)
- **Replaced by:** `scripts/fix_overpass_cache.sh`
- **Reason:** Emergency fix for specific incident, improved version available
---
## Why Keep Deprecated Scripts?
1. **Historical reference** - Document what approaches were tried
2. **Learning resource** - See evolution of deployment methods
3. **Emergency fallback** - Rare cases where old method might be needed
4. **Code archaeology** - Understand decisions made during development
---
## Cleanup Policy
Scripts will be removed from this directory after:
- 6 months of no usage
- Confirmed replacement works in production
- No dependencies from other scripts
---
## If You Need to Use a Deprecated Script
1. Check if there's a replacement (see above)
2. Understand why it was deprecated
3. Consider updating to new method
4. If absolutely necessary, test thoroughly first
---
**Last Updated:** October 27, 2025
