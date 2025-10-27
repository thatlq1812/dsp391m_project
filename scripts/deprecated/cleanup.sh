#!/bin/bash
# Cleanup script for Python project
# Removes cache files, temporary data, and build artifacts
set +H  # Disable history expansion to avoid "event ! not found" errors

echo "Starting cleanup..."

# Remove Python cache
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null

# Remove pytest cache
echo "Removing pytest cache..."
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null

# Remove old data runs (keep last 7 days)
echo "Cleaning old data runs..."
if [ -d "data/node" ]; then
 find data/node -type d -mtime +7 -exec rm -rf {} + 2>/dev/null
fi

# Remove old images (keep last 7 days)
if [ -d "data/images" ]; then
 find data/images -type f -mtime +7 -delete 2>/dev/null
fi

# Remove temporary files
echo "Removing temporary files..."
find . -type f -name "*.tmp" -delete 2>/dev/null
find . -type f -name "*.log" -mtime +7 -delete 2>/dev/null

# Clean MLflow artifacts (optional - keep recent runs)
if [ -d "mlruns" ]; then
 echo "Cleaning old MLflow runs (>30 days)..."
 find mlruns -type d -mtime +30 -exec rm -rf {} + 2>/dev/null
fi

echo "Cleanup completed!"
echo ""
echo "Summary:"
echo "- Python cache removed"
echo "- Old data runs cleaned (>7 days)"
echo "- Temporary files removed"
