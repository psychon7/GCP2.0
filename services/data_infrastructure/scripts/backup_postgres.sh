#!/bin/bash
# PostgreSQL Backup Script for GCP Project

# Set variables
BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/postgres_backup_$TIMESTAMP.sql.gz"
LOG_FILE="$BACKUP_DIR/backup_log_$TIMESTAMP.log"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Log start time
echo "Starting PostgreSQL backup at $(date)" > $LOG_FILE

# Perform backup
echo "Creating backup file: $BACKUP_FILE" >> $LOG_FILE
pg_dump -h postgres -U gcp_user -d gcp -F c | gzip > $BACKUP_FILE

# Check if backup was successful
if [ $? -eq 0 ]; then
    echo "Backup completed successfully at $(date)" >> $LOG_FILE
    echo "Backup file size: $(du -h $BACKUP_FILE | cut -f1)" >> $LOG_FILE
    
    # Remove backups older than 7 days
    echo "Removing backups older than 7 days" >> $LOG_FILE
    find $BACKUP_DIR -name "postgres_backup_*.sql.gz" -type f -mtime +7 -delete
    
    # List remaining backups
    echo "Remaining backups:" >> $LOG_FILE
    ls -lh $BACKUP_DIR/postgres_backup_*.sql.gz >> $LOG_FILE
    
    echo "Backup process completed successfully"
    exit 0
else
    echo "Backup failed at $(date)" >> $LOG_FILE
    echo "Backup process failed"
    exit 1
fi
