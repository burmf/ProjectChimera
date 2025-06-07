# CLAUDE.md - Claude Code Optimization Guide

## Current Project Status & Quick Actions

**Project:** AI Trading Bot (Docker + PostgreSQL + TimescaleDB + Redis)
**Phase:** Integration & Testing (Phase 2/3)
**Current Priority:** End-to-end integration testing

### #quick-setup
```bash
# Essential Commands - Execute in parallel when possible
docker compose up -d                    # Start all services
docker compose logs -f                  # Monitor logs
streamlit run app.py                    # Direct execution (legacy)
```

### #current-issues  
- Phase 2 focus: End-to-end integration testing
- Performance optimization and monitoring setup
- Known: signal_fusion.py look-ahead bias (lines 84-89)

## Claude Code Workflow Optimization

### Exploration Phase - Parallel Context Gathering
**Priority 1 (Execute immediately in parallel):**
```
Glob: "**/*.py" | Bash: "git status" | Bash: "git log --oneline -10" | LS: project root
```

**Priority 2 (Problem-specific):**
```
Grep: error patterns | Read: up to 5 related files | Bash: validation commands
```

### Planning Phase - Strategic Todo Management
- **Use TodoWrite for:** 3+ step tasks, complex implementations, multiple file changes
- **Skip TodoWrite for:** Single file edits, simple fixes, direct commands
- **Task granularity:** One logical change per todo item

### Implementation Phase - Batch Operations
**Parallel file operations:**
```python
# Good: Batch read multiple files
Read: [file1.py, file2.py, file3.py]
MultiEdit: multiple file changes
Bash: [test, type-check, format] in parallel

# Avoid: Sequential operations
Read → Edit → Read → Edit (inefficient)
```

### Commit Phase - Automated Git Operations
```bash
# Execute in parallel for efficiency
git status & git diff & git log --oneline -5
git add relevant_files && git commit -m "descriptive_message"
```

## Project Architecture (Essential Only)

### #architecture-core
```
Trading Bot System:
├── app.py (Streamlit UI - 4 tabs)
├── core/ (16 production-ready modules)
│   ├── ai_manager.py (OpenAI + o3 models)
│   ├── portfolio.py (P&L tracking)
│   ├── backtester.py (temporal constraints)
│   └── database_adapter.py (SQLite/PostgreSQL)
├── modules/ (Analysis engines)
│   ├── technical_analyzer.py (TA-lib)
│   ├── signal_fusion.py (ML fusion)
│   └── feature_builder.py (ML features)
└── workers/ (Background processing)
```

### #data-flow
```
Data → Processing → Decision → Execution
YFinance/NewsAPI → AI Analysis → Trading Signals → Portfolio Simulation
SQLite/PostgreSQL ← Redis Streams ← Docker Services ← Monitoring
```

### #docker-services
```
Services: postgres + redis + app + price_collector + news_collector
Access: UI:8501 | DB:5432 | Redis:6379
Networks: bot_network (isolated)
```

## Development Patterns

### #debug-guide
**Fast problem resolution:**
1. `git status` + `docker compose logs -f` (parallel)
2. `Grep: "error|exception"` in logs directory
3. `Read: problematic_module.py` + error analysis
4. `MultiEdit: fix + test` + `Bash: validation`

### #testing-strategy  
```bash
# Execute validation pipeline
python -m py_compile *.py                    # Syntax check
pytest tests/ -v                            # Unit tests  
docker compose build --no-cache             # Integration
```

### #git-workflow
- **Commit frequency:** After each completed todo item
- **Message format:** `action(scope): description`
- **Staging strategy:** Only relevant files (avoid `git add .`)

## Critical System Knowledge

### #known-bugs
1. **signal_fusion.py:84-89** - Look-ahead bias in ML training
2. **backtester.py** - Same-bar stop loss creates unrealistic results
3. **Risk management** - Static parameters ignore volatility regimes

### #database-schema
- **Production:** PostgreSQL + TimescaleDB (hypertables for time-series)
- **Development:** SQLite compatibility maintained
- **Key tables:** price_data, news_articles, ai_trade_decisions, trade_history

### #api-integrations
- **OpenAI:** Multiple model support (o3, gpt-4), cost tracking
- **NewsAPI:** Forex keywords, 29-day lookback
- **YFinance:** USD/JPY focus, 1-hour intervals, 90-day history

## Quick Reference Links

- **Full Architecture Details:** [Detailed in original CLAUDE.md lines 219-297]
- **Investment Strategy Theory:** [Detailed in original CLAUDE.md lines 117-163]  
- **Docker Operations:** [Detailed in original CLAUDE.md lines 175-217]
- **Database Schemas:** [See sql/init.sql for complete definitions]

---
*This optimized guide focuses on Claude Code workflow efficiency. For complete system documentation, reference the full architecture sections.*