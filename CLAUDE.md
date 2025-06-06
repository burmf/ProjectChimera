# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Docker Setup (Recommended)

**Initial Setup:**
```bash
# Install Docker (Amazon Linux 2023)
sudo yum update -y && sudo yum install docker -y
sudo usermod -aG docker ec2-user
sudo systemctl start docker && sudo systemctl enable docker

# Setup project
./scripts/setup.sh
```

**Environment Configuration:**
```bash
# Copy environment template and configure API keys
cp .env.example .env
# Edit .env with your NewsAPI and OpenAI API keys
```

**Docker Operations:**
```bash
# Start all services (PostgreSQL + Redis + App + Collectors)
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild after code changes
docker compose build --no-cache && docker compose up -d

# Database backup
./scripts/backup.sh
```

**Service Access:**
- Web UI: http://localhost:8501
- PostgreSQL: localhost:5432
- Redis: localhost:6379

### Legacy Direct Execution

**Database Setup (SQLite - Legacy):**
```bash
python database.py
```

**Application Launch (Direct):**
```bash
streamlit run app.py
```

**Dependencies:**
```bash
pip install -r requirements.txt
```

## Architecture Overview

### Core System Components

**Main Application (app.py)**
- Streamlit web interface with 4 main tabs
- Handles API key management (NewsAPI, OpenAI)
- Orchestrates data collection, AI analysis, and backtesting
- Database path: `data/system_data.db`
- Target pair: USD/JPY

**Core Modules (core/)**
- `ai_manager.py`: OpenAI API integration with parallel model inference, cost tracking, and structured JSON response handling
- `portfolio.py`: Trading portfolio simulation with spread consideration, position management, and P&L tracking  
- `backtester.py`: Strategy backtesting engine supporting technical indicators and AI-driven signals

**Data Layer**
- `database.py`: SQLite schema setup with tables for price data, news articles, AI decisions, trade history, and API usage tracking
- `data_collector.py`: YFinance price data and NewsAPI news data collection with database persistence

**Analysis Modules (modules/)**
- `technical_analyzer.py`: SMA crossover signal generation for technical trading strategies

### Data Flow Architecture

1. **Data Collection**: YFinance (price) + NewsAPI (news) → SQLite database
2. **AI Processing**: News articles → OpenAI models → Trade decisions (JSON format)
3. **Signal Generation**: Technical indicators OR AI decisions → Trading signals
4. **Backtesting**: Historical price data + Signals → Portfolio simulation with realistic trading costs
5. **Visualization**: Plotly charts for price action, trade entries/exits, and equity curves

### Key Configuration

- Database: SQLite at `data/system_data.db`
- Default trading pair: USD/JPY
- Price data: 90-day lookback, 1-hour intervals
- News data: 29-day lookback with forex-relevant keywords
- Spread simulation: Configurable pip-based spread costs

### Database Schema Integration

**PostgreSQL + TimescaleDB (Production):**
- `trading.price_data`: Time-series optimized OHLCV data with hypertables
- `trading.news_articles`: News content with AI processing status
- `trading.ai_trade_decisions`: Structured trading recommendations with JSONB storage
- `trading.openai_api_usage`: API cost tracking with time-series optimization
- `trading.trade_history`: Complete trade simulation records
- Automated data retention policies and materialized views for performance

**SQLite (Legacy/Development):**
- Compatible schema for local development and testing
- Simplified table structure without TimescaleDB features

### AI Model Integration

Supports multiple OpenAI models with parallel processing:
- Cost estimation per model/token usage
- Structured JSON response parsing for trade decisions
- Manual analysis logging for model comparison
- Configurable system prompts for different trading strategies

### Docker Architecture

**Services:**
- `postgres`: TimescaleDB container with automated schema initialization
- `redis`: Message streams and caching layer
- `app`: Main Streamlit web interface
- `price_collector`: Scheduled price data collection (1-minute intervals)
- `news_collector`: Scheduled news collection and AI analysis (5-minute intervals)

**Volumes:**
- `postgres_data`: Persistent database storage
- `redis_data`: Redis persistence
- `./data`: Application data directory
- `./logs`: Application logs

**Networks:**
- Isolated `bot_network` for service communication
- Health checks and automatic restarts for reliability