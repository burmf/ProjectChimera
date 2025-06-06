# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Management Policy

**Automatic Git Operations:**
- Commit changes regularly after completing tasks or making significant modifications
- Use meaningful commit messages that describe the purpose of changes
- Stage relevant files appropriately, excluding temporary/generated files
- Follow conventional commit format with Japanese or English descriptions as appropriate

## System Design Optimization Guidelines

### Current Architecture Analysis

**Strengths:**
- Complete Docker microservices setup (PostgreSQL + TimescaleDB + Redis)
- Advanced AI integration with o3 model support and parallel inference
- Comprehensive technical analysis using ta-lib
- Full Streamlit web interface implementation

**Critical Issues Identified:**

1. **Database Layer Inconsistency** ✅ RESOLVED
   - Removed legacy `database.py` (SQLite-only implementation)
   - Standardized on `database_adapter.py` (unified SQLite/PostgreSQL support)
   - All modules now use consistent database interface

2. **Data Flow Inconsistencies** ✅ RESOLVED
   - Fixed data structure mismatches in data_collector.py
   - News processing pipeline validated and corrected
   - Added proper field mapping (symbol -> pair, added news ID generation)

3. **Module Implementation Status** ✅ RESOLVED
   - **Audit Complete**: All 16 core modules are fully implemented with high quality
   - **Key Implementations**:
     - Portfolio management with P&L tracking
     - Advanced backtesting with temporal constraints (look-ahead bias prevention)  
     - AI integration with o3 model support and parallel inference
     - Real-time stream processing (Redis-based)
     - Technical analysis with TA-Lib integration
     - Machine learning signal fusion with scikit-learn
     - Performance monitoring and system health checks
   - **Status**: Production-ready architecture

### Implementation Priorities

**Phase 1: Core Stability** ✅ COMPLETED
1. ✅ Database layer unification
2. ✅ Data collector fixes
3. ✅ Core module audit and cleanup
4. ⏳ Basic test coverage (in progress)

**Phase 2: Integration & Testing (Current Priority)**
5. ✅ All core modules implemented (higher quality than expected)
6. 🔄 End-to-end integration testing (current focus)
7. 🔄 Performance optimization and monitoring setup

**Phase 3: Production Deployment (Next Priority)**
8. 📋 Docker environment setup documentation
9. 📋 Operational procedures and monitoring
10. 📋 User documentation and guides

### Development Best Practices

- **Single Source of Truth**: Eliminate duplicate implementations
- **Data Flow Validation**: Ensure type consistency across module boundaries
- **Modular Design**: Keep dependencies minimal and clear
- **Test-Driven**: Validate each component before integration
- **Incremental**: Complete one module fully before moving to next

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