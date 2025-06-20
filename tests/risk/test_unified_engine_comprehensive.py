"""
Comprehensive tests for UnifiedRiskEngine - targeting 60% coverage
Tests for async risk management, Kelly calculation, ATR sizing, and DD guard
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from project_chimera.risk.unified_engine import (
    UnifiedRiskEngine, 
    UnifiedRiskConfig, 
    UnifiedRiskDecision
)
from project_chimera.domains.market import Signal, SignalType, SignalStrength, OHLCV


class TestUnifiedRiskEngine:
    """Comprehensive tests for UnifiedRiskEngine"""
    
    @pytest.fixture
    def risk_config(self):
        """Standard risk configuration for testing"""
        return UnifiedRiskConfig(
            kelly_base_fraction=0.25,
            kelly_ewma_alpha=0.1,
            kelly_min_trades=10,
            atr_target_daily_vol=0.02,
            atr_periods=14,
            atr_min_position=0.01,
            atr_max_position=0.15,
            dd_caution_threshold=0.05,
            dd_warning_threshold=0.10,
            dd_critical_threshold=0.20,
            dd_warning_cooldown_hours=4.0,
            dd_critical_cooldown_hours=24.0,
            max_leverage=5.0,
            min_confidence=0.3,
            max_portfolio_vol=0.03
        )
    
    @pytest.fixture 
    def risk_engine(self, risk_config):
        """Risk engine instance for testing"""
        return UnifiedRiskEngine(risk_config, initial_equity=100000.0)
    
    @pytest.fixture
    def sample_signal(self):
        """Sample trading signal for testing"""
        return Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=Decimal("50000"),
            timestamp=datetime.now(),
            strategy_name="test_strategy",
            confidence=0.8,
            reasoning="Strong bullish signal"
        )
    
    @pytest.fixture
    def sample_ohlcv(self):
        """Sample OHLCV data for ATR calculation"""
        base_time = datetime.now()
        return [
            OHLCV(
                symbol="BTCUSDT",
                open=Decimal("49000"),
                high=Decimal("51000"),
                low=Decimal("48000"),
                close=Decimal("50000"),
                volume=Decimal("1000"),
                timestamp=base_time - timedelta(minutes=i),
                timeframe="1m"
            ) for i in range(20)
        ]

    def test_initialization(self, risk_config):
        """Test risk engine initialization"""
        engine = UnifiedRiskEngine(risk_config, initial_equity=150000.0)
        
        assert engine.config == risk_config
        assert engine.dd_guard.current_equity == 150000.0
        assert engine.last_decision is None
        assert engine.decision_count == 0
    
    def test_initialization_with_defaults(self):
        """Test initialization with default config"""
        engine = UnifiedRiskEngine(initial_equity=100000.0)
        
        assert engine.config is not None
        assert engine.dd_guard.current_equity == 100000.0
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_basic(self, risk_engine, sample_signal):
        """Test basic position size calculation"""
        with patch.object(risk_engine, '_get_portfolio_value', return_value=100000.0):
            decision = await risk_engine.calculate_position_size(
                signal=sample_signal,
                current_price=50000.0,
                portfolio_value=100000.0
            )
        
        assert isinstance(decision, UnifiedRiskDecision)
        assert decision.can_trade is True
        assert 0.0 <= decision.position_size_pct <= 1.0
        assert decision.leverage >= 1.0
        assert decision.confidence > 0.0
        assert decision.reasoning is not None
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_with_portfolio_value(self, risk_engine, sample_signal):
        """Test position calculation with provided portfolio value"""
        decision = await risk_engine.calculate_position_size(
            signal=sample_signal,
            current_price=50000.0,
            portfolio_value=120000.0
        )
        
        assert decision.can_trade is True
        assert decision.position_size_pct > 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_position_size_await_portfolio_value(self, risk_engine, sample_signal):
        """Test position calculation when awaiting portfolio value"""
        with patch.object(risk_engine, '_get_portfolio_value', return_value=110000.0) as mock_get_value:
            decision = await risk_engine.calculate_position_size(
                signal=sample_signal,
                current_price=50000.0
            )
            
            mock_get_value.assert_called_once()
            assert decision.can_trade is True
    
    @pytest.mark.asyncio
    async def test_dd_guard_trading_halt(self, risk_engine, sample_signal):
        """Test trading halt when DD guard triggers"""
        # Force critical drawdown state
        risk_engine.dd_guard.update_equity(50000.0, datetime.now())  # 50% loss
        
        decision = await risk_engine.calculate_position_size(
            signal=sample_signal,
            current_price=50000.0,
            portfolio_value=50000.0
        )
        
        assert decision.can_trade is False
        assert decision.position_size_pct == 0.0
        assert "DD Guard trading halt" in decision.reasoning
    
    @pytest.mark.asyncio
    async def test_low_confidence_signal(self, risk_engine):
        """Test handling of low confidence signals"""
        low_conf_signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            price=Decimal("50000"),
            timestamp=datetime.now(),
            strategy_name="test_strategy",
            confidence=0.2,  # Below min_confidence
            reasoning="Weak signal"
        )
        
        decision = await risk_engine.calculate_position_size(
            signal=low_conf_signal,
            current_price=50000.0,
            portfolio_value=100000.0
        )
        
        # Should still allow trading but with reduced position
        assert decision.can_trade is True
        assert decision.position_size_pct <= 0.05  # Conservative sizing
    
    @pytest.mark.asyncio
    async def test_volatility_limits(self, risk_engine, sample_signal):
        """Test portfolio volatility limits"""
        # Create high volatility scenario
        risk_engine.config.max_portfolio_vol = 0.01  # Very low limit
        
        decision = await risk_engine.calculate_position_size(
            signal=sample_signal,
            current_price=50000.0,
            portfolio_value=100000.0
        )
        
        assert decision.can_trade is True
        assert "volatility_limit" in decision.primary_constraint or decision.position_size_pct <= 0.05
    
    def test_update_trade_result(self, risk_engine):
        """Test trade result updates"""
        initial_equity = risk_engine.dd_guard.current_equity
        
        # Winning trade
        risk_engine.update_trade_result(0.05, 105000.0)
        
        assert risk_engine.dd_guard.current_equity == 105000.0
        
        # Losing trade
        risk_engine.update_trade_result(-0.03, 101850.0)
        
        assert risk_engine.dd_guard.current_equity == 101850.0
    
    def test_add_price_data(self, risk_engine, sample_ohlcv):
        """Test adding price data for ATR calculations"""
        initial_atr_data_count = len(risk_engine.atr_controller.price_history)
        
        for ohlcv in sample_ohlcv[:5]:
            risk_engine.add_price_data(ohlcv)
        
        final_atr_data_count = len(risk_engine.atr_controller.price_history)
        assert final_atr_data_count > initial_atr_data_count
    
    def test_calculate_base_size_kelly_available(self, risk_engine):
        """Test base size calculation when Kelly data is available"""
        # Add sufficient trade history for Kelly
        for i in range(25):
            return_pct = 0.02 if i % 2 == 0 else -0.01  # 60% win rate
            risk_engine.kelly_calc.add_trade_result(return_pct, datetime.now())
        
        kelly_result = risk_engine.kelly_calc.calculate_dynamic_kelly()
        atr_result = risk_engine.atr_controller.calculate_target_position_size(50000.0)
        
        sample_signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=Decimal("50000"),
            timestamp=datetime.now(),
            strategy_name="test_strategy",
            confidence=0.8
        )
        
        size, method, confidence = risk_engine._calculate_base_size(
            kelly_result, atr_result, sample_signal
        )
        
        assert size > 0.0
        assert method in ['kelly', 'atr', 'conservative']
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_base_size_fallback(self, risk_engine):
        """Test base size calculation fallback to conservative"""
        # Empty Kelly result and invalid ATR
        from project_chimera.risk.dyn_kelly import DynamicKellyResult
        from project_chimera.risk.atr_target import ATRTargetResult
        
        kelly_result = DynamicKellyResult(
            kelly_fraction=0.0, ewma_win_rate=0.5, ewma_avg_win=0.0,
            ewma_avg_loss=0.0, raw_kelly=0.0, vol_adjustment_factor=1.0,
            confidence_score=0.0, sample_size=5, last_updated=datetime.now()
        )
        
        atr_result = ATRTargetResult(
            position_size_pct=0.0, current_atr=0.0, daily_vol_estimate=0.0,
            vol_target_ratio=1.0, regime_adjustment=1.0, confidence_score=0.0,
            target_met=False, last_updated=datetime.now(), price_level=50000.0
        )
        
        sample_signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            price=Decimal("50000"),
            timestamp=datetime.now(),
            strategy_name="test_strategy",
            confidence=0.6
        )
        
        size, method, confidence = risk_engine._calculate_base_size(
            kelly_result, atr_result, sample_signal
        )
        
        assert size == 0.05  # Conservative 5%
        assert method == 'conservative'
    
    def test_apply_volatility_limits_scaling(self, risk_engine):
        """Test volatility limits with scaling"""
        from project_chimera.risk.atr_target import ATRTargetResult
        
        atr_result = ATRTargetResult(
            position_size_pct=0.1, current_atr=1000.0, daily_vol_estimate=0.05,  # High volatility
            vol_target_ratio=1.0, regime_adjustment=1.0, confidence_score=0.8,
            target_met=True, last_updated=datetime.now(), price_level=50000.0
        )
        
        # High position size that would exceed volatility limit
        position_size = 0.8
        risk_engine.config.max_portfolio_vol = 0.02  # 2% max portfolio volatility
        
        limited_size = risk_engine._apply_volatility_limits(
            position_size, atr_result, 100000.0
        )
        
        # Should be scaled down
        assert limited_size < position_size
        assert limited_size * atr_result.daily_vol_estimate <= risk_engine.config.max_portfolio_vol * 1.01
    
    def test_calculate_leverage(self, risk_engine):
        """Test leverage calculation"""
        # High position, high confidence
        leverage = risk_engine._calculate_leverage(0.2, 0.9)
        assert 1.0 <= leverage <= risk_engine.config.max_leverage
        
        # Low position, low confidence
        leverage = risk_engine._calculate_leverage(0.05, 0.3)
        assert 1.0 <= leverage <= 2.0
    
    def test_estimate_risk_metrics(self, risk_engine):
        """Test risk metric calculations"""
        portfolio_vol = risk_engine._estimate_portfolio_volatility(0.1, 0.02)
        assert portfolio_vol == 0.002  # 0.1 * 0.02
        
        risk_adj_return = risk_engine._estimate_risk_adjusted_return(0.1, 0.8, 0.015)
        assert risk_adj_return > 0.0
        
        max_loss = risk_engine._estimate_max_loss(0.1, 0.02)
        assert max_loss == 0.004  # 0.1 * 0.02 * 2
    
    def test_identify_constraint(self, risk_engine):
        """Test constraint identification logic"""
        # Normal sizing
        constraint = risk_engine._identify_constraint(0.1, 0.1, 0.1, 1.0)
        assert constraint == "normal_sizing"
        
        # DD guard constraint
        constraint = risk_engine._identify_constraint(0.1, 0.05, 0.05, 0.5)
        assert "drawdown_guard" in constraint
        
        # Volatility limit
        constraint = risk_engine._identify_constraint(0.1, 0.1, 0.08, 1.0)
        assert constraint == "volatility_limit"
        
        # Low confidence
        constraint = risk_engine._identify_constraint(0.03, 0.03, 0.03, 1.0)
        assert constraint == "insufficient_confidence"
    
    def test_calculate_overall_confidence(self, risk_engine):
        """Test overall confidence calculation"""
        from project_chimera.risk.dyn_kelly import DynamicKellyResult
        from project_chimera.risk.atr_target import ATRTargetResult
        from project_chimera.risk.dd_guard import DDGuardState, DDGuardTier
        
        kelly_result = DynamicKellyResult(
            kelly_fraction=0.1, ewma_win_rate=0.6, ewma_avg_win=0.02,
            ewma_avg_loss=0.01, raw_kelly=0.1, vol_adjustment_factor=1.0,
            confidence_score=0.8, sample_size=25, last_updated=datetime.now()
        )
        
        atr_result = ATRTargetResult(
            position_size_pct=0.1, current_atr=1000.0, daily_vol_estimate=0.02,
            vol_target_ratio=1.0, regime_adjustment=1.0, confidence_score=0.7,
            target_met=True, last_updated=datetime.now(), price_level=50000.0
        )
        
        dd_state = DDGuardState(
            tier=DDGuardTier.NORMAL, drawdown_pct=0.02, position_multiplier=1.0,
            warning_cooldown_until=None, critical_cooldown_until=None, last_updated=datetime.now()
        )
        
        sample_signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=Decimal("50000"),
            timestamp=datetime.now(),
            strategy_name="test_strategy",
            confidence=0.8
        )
        
        confidence = risk_engine._calculate_overall_confidence(
            kelly_result, atr_result, dd_state, sample_signal, 0.75
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high with good inputs
    
    def test_generate_reasoning(self, risk_engine):
        """Test reasoning generation"""
        from project_chimera.risk.dyn_kelly import DynamicKellyResult
        from project_chimera.risk.atr_target import ATRTargetResult  
        from project_chimera.risk.dd_guard import DDGuardState, DDGuardTier
        
        sample_signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=Decimal("50000"),
            timestamp=datetime.now(),
            strategy_name="test_strategy",
            confidence=0.8
        )
        
        kelly_result = DynamicKellyResult(
            kelly_fraction=0.1, ewma_win_rate=0.6, ewma_avg_win=0.02,
            ewma_avg_loss=0.01, raw_kelly=0.1, vol_adjustment_factor=1.0,
            confidence_score=0.8, sample_size=25, last_updated=datetime.now()
        )
        
        atr_result = ATRTargetResult(
            position_size_pct=0.1, current_atr=1000.0, daily_vol_estimate=0.02,
            vol_target_ratio=1.0, regime_adjustment=1.0, confidence_score=0.7,
            target_met=True, last_updated=datetime.now(), price_level=50000.0
        )
        
        dd_state = DDGuardState(
            tier=DDGuardTier.NORMAL, drawdown_pct=0.02, position_multiplier=1.0,
            warning_cooldown_until=None, critical_cooldown_until=None, last_updated=datetime.now()
        )
        
        reasoning = risk_engine._generate_reasoning(
            sample_signal, kelly_result, atr_result, dd_state, 0.1, "kelly", "normal_sizing"
        )
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert "strong" in reasoning.lower()
        assert "kelly" in reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_health_check(self, risk_engine):
        """Test async health check"""
        with patch.object(risk_engine, '_get_portfolio_value', return_value=100000.0):
            with patch.object(risk_engine, '_get_trade_statistics', return_value={
                'win_rate': 0.6, 'avg_win': 0.02, 'avg_loss': 0.01, 'total_trades': 50
            }):
                health = await risk_engine.health_check()
        
        assert health['status'] == 'healthy'
        assert health['portfolio_value'] == 100000.0
        assert health['can_trade'] is True
        assert 'dd_tier' in health
    
    @pytest.mark.asyncio
    async def test_health_check_error(self, risk_engine):
        """Test health check with error"""
        with patch.object(risk_engine, '_get_portfolio_value', side_effect=Exception("Network error")):
            health = await risk_engine.health_check()
        
        assert health['status'] == 'error'
        assert 'error' in health
    
    def test_get_statistics(self, risk_engine):
        """Test comprehensive statistics retrieval"""
        stats = risk_engine.get_statistics()
        
        assert 'engine' in stats
        assert 'kelly' in stats
        assert 'atr' in stats
        assert 'drawdown' in stats
        
        assert stats['engine']['decision_count'] == 0
        assert 'config' in stats['engine']
    
    def test_hot_reload_config(self, risk_engine):
        """Test hot configuration reload"""
        old_kelly_fraction = risk_engine.config.kelly_base_fraction
        
        new_config = UnifiedRiskConfig(
            kelly_base_fraction=0.3,  # Different value
            kelly_ewma_alpha=0.15,
            kelly_min_trades=15,
            atr_target_daily_vol=0.025,
            atr_periods=21,
            atr_min_position=0.005,
            atr_max_position=0.25,
            dd_caution_threshold=0.03,
            dd_warning_threshold=0.08,
            dd_critical_threshold=0.15
        )
        
        risk_engine.hot_reload_config(new_config)
        
        assert risk_engine.config.kelly_base_fraction == 0.3
        assert risk_engine.kelly_calc.config.base_kelly_fraction == 0.3
        assert risk_engine.atr_controller.config.target_daily_vol == 0.025
        assert risk_engine.dd_guard.config.warning_threshold == 0.08
    
    def test_reset_all_state(self, risk_engine):
        """Test complete state reset"""
        # Add some state first
        risk_engine.decision_count = 5
        risk_engine.update_trade_result(0.05, 105000.0)
        
        # Reset
        risk_engine.reset_all_state(150000.0)
        
        assert risk_engine.decision_count == 0
        assert risk_engine.last_decision is None
        assert risk_engine.dd_guard.current_equity == 150000.0
    
    def test_can_trade_sync(self, risk_engine):
        """Test synchronous trading check"""
        can_trade = risk_engine.can_trade_sync()
        assert isinstance(can_trade, bool)
        assert can_trade is True  # Should be true initially
        
        # Force drawdown
        risk_engine.dd_guard.update_equity(70000.0, datetime.now())  # 30% loss
        can_trade = risk_engine.can_trade_sync()
        assert can_trade is False  # Should be false after critical drawdown
    
    @pytest.mark.asyncio
    async def test_evaluate_signal_async(self, risk_engine, sample_signal):
        """Test async signal evaluation wrapper"""
        decision = await risk_engine.evaluate_signal_async(sample_signal, 50000.0)
        
        assert isinstance(decision, UnifiedRiskDecision)
        assert decision.can_trade is True
    
    def test_config_from_settings(self):
        """Test config creation from settings"""
        with patch('project_chimera.risk.unified_engine.get_settings') as mock_settings:
            mock_risk = MagicMock()
            mock_risk.kelly_base_fraction = 0.3
            mock_risk.kelly_ewma_alpha = 0.1
            mock_risk.kelly_min_trades = 20
            mock_risk.atr_target_daily_vol = 0.02
            mock_risk.atr_periods = 14
            mock_risk.atr_min_position = 0.01
            mock_risk.atr_max_position = 0.2
            mock_risk.dd_caution_threshold = 0.05
            mock_risk.dd_warning_threshold = 0.1
            mock_risk.dd_critical_threshold = 0.2
            mock_risk.dd_warning_cooldown_hours = 4.0
            mock_risk.dd_critical_cooldown_hours = 24.0
            mock_risk.max_leverage = 10.0
            mock_risk.min_confidence = 0.3
            mock_risk.max_portfolio_vol = 0.02
            
            mock_settings.return_value.risk = mock_risk
            
            config = UnifiedRiskConfig.from_settings()
            
            assert config.kelly_base_fraction == 0.3
            assert config.atr_target_daily_vol == 0.02
            assert config.dd_warning_threshold == 0.1
    
    def test_decision_is_valid(self):
        """Test UnifiedRiskDecision validation"""
        from project_chimera.risk.dyn_kelly import DynamicKellyResult
        from project_chimera.risk.atr_target import ATRTargetResult
        from project_chimera.risk.dd_guard import DDGuardState, DDGuardTier
        
        # Valid decision
        decision = UnifiedRiskDecision(
            position_size_pct=0.1,
            leverage=2.0,
            can_trade=True,
            confidence=0.7,
            kelly_result=DynamicKellyResult(
                kelly_fraction=0.1, ewma_win_rate=0.6, ewma_avg_win=0.02,
                ewma_avg_loss=0.01, raw_kelly=0.1, vol_adjustment_factor=1.0,
                confidence_score=0.8, sample_size=25, last_updated=datetime.now()
            ),
            atr_result=ATRTargetResult(
                position_size_pct=0.1, current_atr=1000.0, daily_vol_estimate=0.02,
                vol_target_ratio=1.0, regime_adjustment=1.0, confidence_score=0.7,
                target_met=True, last_updated=datetime.now(), price_level=50000.0
            ),
            dd_state=DDGuardState(
                tier=DDGuardTier.NORMAL, drawdown_pct=0.02, position_multiplier=1.0,
                warning_cooldown_until=None, critical_cooldown_until=None, last_updated=datetime.now()
            ),
            estimated_daily_vol=0.002,
            risk_adjusted_return=0.5,
            max_loss_estimate=0.004,
            primary_constraint="normal_sizing",
            sizing_method="kelly",
            reasoning="Test decision"
        )
        
        assert decision.is_valid() is True
        
        # Invalid decision - can't trade
        decision.can_trade = False
        assert decision.is_valid() is False
        
        # Invalid decision - invalid position size
        decision.can_trade = True
        decision.position_size_pct = -0.1
        assert decision.is_valid() is False
        
        decision.position_size_pct = 1.5  # > 1.0
        assert decision.is_valid() is False
        
        # Invalid decision - low confidence
        decision.position_size_pct = 0.1
        decision.confidence = 0.1  # Below 0.3 threshold
        assert decision.is_valid() is False