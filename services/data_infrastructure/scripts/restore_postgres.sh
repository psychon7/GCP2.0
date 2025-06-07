#!/bin/bash
# PostgreSQL Restore Script for GCP Project

# Check if backup file is provided
if [ -z "$1" ]; then
    echo "Error: No backup file specified"
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Set variables
BACKUP_FILE=$1
LOG_DIR="/backups/postgres"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/restore_log_$TIMESTAMP.log"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Error: Backup file '$BACKUP_FILE' not found"
    exit 1
fi

# Log start time
echo "Starting PostgreSQL restore at $(date)" > $LOG_FILE
echo "Restoring from backup file: $BACKUP_FILE" >> $LOG_FILE

# Confirm restore
echo "WARNING: This will overwrite the current PostgreSQL database."
read -p "Are you sure you want to proceed? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Restore cancelled by user"
    exit 0
fi

# Drop and recreate database
echo "Dropping and recreating database..." >> $LOG_FILE
PGPASSWORD=${POSTGRES_PASSWORD:-gcp_password} psql -h postgres -U gcp_user -c "DROP DATABASE IF EXISTS gcp;" postgres
PGPASSWORD=${POSTGRES_PASSWORD:-gcp_password} psql -h postgres -U gcp_user -c "CREATE DATABASE gcp;" postgres

# Restore database
echo "Restoring database from backup..." >> $LOG_FILE
gunzip -c $BACKUP_FILE | pg_restore -h postgres -U gcp_user -d gcp

# Check if restore was successful
if [ $? -eq 0 ]; then
    echo "Restore completed successfully at $(date)" >> $LOG_FILE
    echo "Restore process completed successfully"
    exit 0
else
    echo "Restore failed at $(date)" >> $LOG_FILE
    echo "Restore process failed"
    exit 1
fi
