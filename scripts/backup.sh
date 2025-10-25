#!/bin/bash
# Backup Script for Traffic Forecast System
# Creates backup of database and configuration

set -e

BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="traffic_forecast_backup_${TIMESTAMP}"

echo "Creating backup: $BACKUP_NAME"
echo ""

# Create backup directory
mkdir -p $BACKUP_DIR/$BACKUP_NAME

# Backup database
if [ -f "traffic_history.db" ]; then
    echo "Backing up database..."
    cp traffic_history.db $BACKUP_DIR/$BACKUP_NAME/
fi

# Backup configurations
echo "Backing up configurations..."
cp -r configs $BACKUP_DIR/$BACKUP_NAME/

# Backup environment
if [ -f ".env" ]; then
    cp .env $BACKUP_DIR/$BACKUP_NAME/
fi

# Backup recent data (last 3 runs)
if [ -d "data/node" ]; then
    echo "Backing up recent data..."
    mkdir -p $BACKUP_DIR/$BACKUP_NAME/data/node
    ls -t data/node | head -3 | while read dir; do
        cp -r "data/node/$dir" "$BACKUP_DIR/$BACKUP_NAME/data/node/"
    done
fi

# Create archive
echo "Creating archive..."
tar -czf "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" -C $BACKUP_DIR $BACKUP_NAME
rm -rf $BACKUP_DIR/$BACKUP_NAME

BACKUP_SIZE=$(du -h "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" | cut -f1)

echo ""
echo "Backup complete!"
echo "Location: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
echo "Size: $BACKUP_SIZE"

# Cleanup old backups (keep last 7)
echo ""
echo "Cleaning old backups (keeping last 7)..."
ls -t $BACKUP_DIR/*.tar.gz | tail -n +8 | xargs -r rm
echo "Done!"
