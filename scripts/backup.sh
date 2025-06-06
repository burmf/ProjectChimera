#!/bin/bash

# Backup script for AI Trading Platform
# Creates backups of database and configuration

set -e

BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="bot_backup_${TIMESTAMP}"

echo "🔄 Starting backup process..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Database backup
echo "💾 Backing up PostgreSQL database..."
docker compose exec -T postgres pg_dump -U botuser trading_bot > "$BACKUP_DIR/${BACKUP_FILE}.sql"

# Configuration backup
echo "⚙️  Backing up configuration..."
tar -czf "$BACKUP_DIR/${BACKUP_FILE}_config.tar.gz" \
    .env \
    docker-compose.yml \
    sql/ \
    --exclude='*.log'

# Clean old backups (keep last 10)
echo "🧹 Cleaning old backups..."
cd "$BACKUP_DIR"
ls -t bot_backup_*.sql | tail -n +11 | xargs -r rm --
ls -t bot_backup_*_config.tar.gz | tail -n +11 | xargs -r rm --

echo "✅ Backup completed: ${BACKUP_FILE}"
echo "📁 Backup location: ${BACKUP_DIR}/"