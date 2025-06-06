#!/bin/bash

# Setup script for AI Trading Platform - The Architect v2
# This script prepares the environment for Docker deployment

set -e  # Exit on any error

echo "🚀 Setting up AI Trading Platform - The Architect v2"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   For Amazon Linux 2023: sudo yum update -y && sudo yum install docker -y"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs sql

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys:"
    echo "   - NEWS_API_KEY: Get from https://newsapi.org/"
    echo "   - OPENAI_API_KEY: Get from https://platform.openai.com/api-keys"
    echo ""
    echo "⚠️  Don't forget to set secure passwords for database!"
else
    echo "✅ .env file already exists"
fi

# Set proper permissions
echo "🔒 Setting permissions..."
chmod +x scripts/*.sh 2>/dev/null || true

# Pull base images to speed up first build
echo "📦 Pulling base Docker images..."
docker pull python:3.11-slim
docker pull timescale/timescaledb:latest-pg16
docker pull redis:7-alpine

# Build the application
echo "🔨 Building application..."
docker compose build

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: docker compose up -d"
echo "3. Access the application at http://localhost:8501"
echo ""
echo "Useful commands:"
echo "  Start:     docker compose up -d"
echo "  Stop:      docker compose down"
echo "  Logs:      docker compose logs -f"
echo "  Rebuild:   docker compose build --no-cache"
echo ""