# DEPRECATED: core/ Directory

**⚠️ DEPRECATION NOTICE ⚠️**

The `core/` directory is deprecated as of version 2.0.0 and will be removed in version 3.0.0.

## Migration Path

All functionality has been moved to the new `src/project_chimera/` structure:

### Core Components Migration

| Old Path | New Path | Status |
|----------|----------|--------|
| `core/ai_orchestrator.py` | `src/project_chimera/orchestrator.py` | ✅ Migrated |
| `core/risk_manager.py` | `src/project_chimera/risk/unified_engine.py` | ✅ Migrated |
| `core/bitget_*_client.py` | `src/project_chimera/datafeed/adapters/bitget_enhanced.py` | ✅ Migrated |
| `core/database_adapter.py` | `src/project_chimera/infra/database.py` | ✅ Migrated |
| `core/performance_monitor.py` | `src/project_chimera/monitor/prom_exporter.py` | ✅ Migrated |
| `core/settings.py` | `src/project_chimera/settings.py` | ✅ Migrated |

### Breaking Changes

1. **Import paths changed**: 
   - Old: `from core.ai_orchestrator import AIOrchestrator`
   - New: `from src.project_chimera.orchestrator import TradingOrchestrator`

2. **Class names updated**:
   - `AIOrchestrator` → `TradingOrchestrator`
   - `RiskManager` → `UnifiedRiskEngine`
   - `BitgetClient` → `BitgetEnhancedAdapter`

3. **Configuration structure**:
   - New unified settings with Pydantic validation
   - Environment variable names standardized

### Migration Steps

1. **Update imports**:
   ```python
   # Old
   from core.ai_orchestrator import AIOrchestrator
   from core.risk_manager import RiskManager
   
   # New  
   from src.project_chimera.orchestrator import TradingOrchestrator
   from src.project_chimera.risk.unified_engine import UnifiedRiskEngine
   ```

2. **Update configuration**:
   ```python
   # Old
   from core.settings import Settings
   
   # New
   from src.project_chimera.settings import get_settings
   settings = get_settings()
   ```

3. **Update class instantiation**:
   ```python
   # Old
   orchestrator = AIOrchestrator()
   
   # New
   orchestrator = TradingOrchestrator(symbols=["BTCUSDT"])
   ```

### Affected Files

The following files still import from `core/` and need migration:

#### Production Code
- `systems/master_profit_system.py`
- `systems/ultra_trading_bot.py`
- `launch.py`
- `modules/signal_fusion.py`
- `modules/feature_builder.py`
- `modules/crypto_trader.py`

#### Test Files
- `tests/test_temporal_backtest.py`
- `tests/test_o3_integration.py`
- `tests/test_bitget_websocket.py`
- `tests/test_scalping_system.py`
- `tests/test_ai_department_integration.py`

#### Worker Files
- `workers/risk_monitor.py`
- `workers/realtime_worker.py`
- `workers/openai_worker.py`
- `workers/ai_result_processor.py`

#### Scripts
- `scripts/learning_cycle_runner.py`
- `scripts/scalping_data_collector.py`

### Removal Timeline

- **v2.0.0** (Current): Deprecation warnings added
- **v2.1.0**: Legacy files marked as deprecated
- **v2.2.0**: Migration scripts provided
- **v3.0.0**: core/ directory removed

### Support

For migration assistance:
1. Check the new API documentation in `src/project_chimera/`
2. Review test files for usage examples
3. Use the new unified settings system

**Do not use core/ for new development. All new features should use the src/project_chimera/ structure.**