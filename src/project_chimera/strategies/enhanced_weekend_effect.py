"""
週末効果戦略（パフォーマンス測定機能付き）
Enhanced Weekend Effect Strategy with Performance Monitoring

Design Reference: CLAUDE.md - Strategy Modules Section 5 (WKND_EFF: Fri 23:00 UTC buy → Mon 01:00 sell)
Related Classes:
- StrategyBase: Abstract interface for generate() and on_fill()
- PerformanceMixin: Trade metrics tracking and performance analysis
- MarketFrame -> Signal: Input/output data structures
- UnifiedRiskEngine: Position sizing integration
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

from ..domains.market import MarketFrame, Signal, SignalType
from ..monitor.performance_logger import TradeResult
from .base import StrategyConfig, TechnicalStrategy
from .performance_mixin import PerformanceDecoratorMixin, PerformanceMixin

logger = logging.getLogger(__name__)


class EnhancedWeekendEffectStrategy(
    PerformanceMixin, PerformanceDecoratorMixin, TechnicalStrategy
):
    """
    Enhanced Weekend Effect Strategy with Performance Monitoring

    戦略ロジック:
    1. 金曜日23:00 UTC以降にロングエントリー
    2. 月曜日01:00 UTC以降にクローズ
    3. パフォーマンス測定とPDCA管理を自動実行
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        # 戦略固有パラメータ
        self.entry_day = self.params.get("entry_day", 4)  # 4 = 金曜日
        self.entry_hour = self.params.get("entry_hour", 23)
        self.exit_day = self.params.get("exit_day", 0)  # 0 = 月曜日
        self.exit_hour = self.params.get("exit_hour", 1)
        self.confidence_threshold = self.params.get("confidence_threshold", 0.6)

        # ポジション管理
        self.current_position = None
        self.position_entry_time = None
        self.entry_price = None

        # パフォーマンス追跡
        self._active_trades = {}

        logger.info(
            "Enhanced Weekend Effect Strategy initialized",
            entry_time=f"Fri {self.entry_hour}:00",
            exit_time=f"Mon {self.exit_hour}:00",
            confidence_threshold=self.confidence_threshold,
        )

    @PerformanceDecoratorMixin.measure_signal_generation
    async def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """
        週末効果シグナル生成（パフォーマンス測定付き）
        """
        current_time = datetime.fromtimestamp(market_data.timestamp)

        # 取引コンテキスト設定
        self.set_trade_context(
            {
                "market_time": current_time.isoformat(),
                "weekday": current_time.weekday(),
                "hour": current_time.hour,
                "price": market_data.last,
                "volume": getattr(market_data, "volume", 0),
                "strategy_version": "enhanced_v1.0",
            }
        )

        # エントリーシグナルチェック
        if self._should_enter(current_time) and not self.current_position:
            return await self._create_entry_signal(market_data, current_time)

        # エグジットシグナルチェック
        if self._should_exit(current_time) and self.current_position:
            return await self._create_exit_signal(market_data, current_time)

        return None

    def _should_enter(self, current_time: datetime) -> bool:
        """エントリー条件チェック"""
        return (
            current_time.weekday() == self.entry_day
            and current_time.hour >= self.entry_hour
        )

    def _should_exit(self, current_time: datetime) -> bool:
        """エグジット条件チェック"""
        return (
            current_time.weekday() == self.exit_day
            and current_time.hour >= self.exit_hour
        )

    async def _create_entry_signal(
        self, market_data: MarketFrame, current_time: datetime
    ) -> Signal:
        """エントリーシグナル作成"""
        # 信頼度計算（簡易版）
        confidence = self._calculate_entry_confidence(market_data, current_time)

        if confidence < self.confidence_threshold:
            logger.debug(
                "Confidence below threshold",
                confidence=confidence,
                threshold=self.confidence_threshold,
            )
            return None

        # テクニカル分析実行
        self.update_price_history(market_data)
        indicators = self.calculate_indicators()

        if not indicators.empty:
            latest_indicators = self.get_latest_indicators()
            self.update_trade_context("technical_indicators", latest_indicators)

        # シグナル作成
        signal = Signal(
            signal_type=SignalType.BUY,
            confidence=confidence,
            price=market_data.last,
            size=self._calculate_position_size(market_data, confidence),
            timestamp=current_time,
            strategy_name=self.__class__.__name__,
            metadata={
                "entry_reason": "weekend_effect_friday",
                "technical_context": (
                    self.get_latest_indicators() if not indicators.empty else {}
                ),
                "market_conditions": {
                    "volatility": getattr(market_data, "volatility", None),
                    "volume": getattr(market_data, "volume", None),
                },
            },
        )

        # ポジション記録
        self.current_position = signal
        self.position_entry_time = current_time
        self.entry_price = market_data.last

        logger.info(
            "Weekend effect entry signal generated",
            confidence=confidence,
            price=signal.price,
            size=signal.size,
        )

        return signal

    async def _create_exit_signal(
        self, market_data: MarketFrame, current_time: datetime
    ) -> Signal:
        """エグジットシグナル作成"""
        if not self.current_position:
            return None

        # 保有期間計算
        hold_duration = (current_time - self.position_entry_time).total_seconds()

        # P&L計算
        pnl = (market_data.last - self.entry_price) * self.current_position.size
        pnl_percentage = (
            (market_data.last - self.entry_price) / self.entry_price
        ) * 100

        # 結果判定
        result = (
            TradeResult.WIN
            if pnl > 0
            else TradeResult.LOSS if pnl < 0 else TradeResult.BREAKEVEN
        )

        # エグジットシグナル作成
        signal = Signal(
            signal_type=SignalType.SELL,
            confidence=0.9,  # エグジットは高信頼度
            price=market_data.last,
            size=self.current_position.size,
            timestamp=current_time,
            strategy_name=self.__class__.__name__,
            metadata={
                "exit_reason": "weekend_effect_monday",
                "hold_duration_seconds": hold_duration,
                "entry_price": self.entry_price,
                "pnl": pnl,
                "pnl_percentage": pnl_percentage,
                "result": result.value,
            },
        )

        # 取引完了ログ（非同期）
        if hasattr(self.current_position, "_trade_id"):
            asyncio.create_task(
                self.log_trade_completed(
                    trade_id=self.current_position._trade_id,
                    exit_price=market_data.last,
                    result=result,
                    actual_pnl=pnl,
                    hold_duration_seconds=hold_duration,
                    execution_time_ms=0.0,  # 実際の実行時間は実装時に計測
                    commission=self._calculate_commission(pnl),
                )
            )

        # ポジションクリア
        self.current_position = None
        self.position_entry_time = None
        self.entry_price = None

        logger.info(
            "Weekend effect exit signal generated",
            pnl=pnl,
            pnl_percentage=pnl_percentage,
            result=result.value,
            hold_duration_hours=hold_duration / 3600,
        )

        return signal

    def _calculate_entry_confidence(
        self, market_data: MarketFrame, current_time: datetime
    ) -> float:
        """エントリー信頼度計算"""
        base_confidence = 0.7  # 週末効果基本信頼度

        # 時間調整（夜遅いほど高信頼度）
        time_factor = min(current_time.hour / 24, 0.3)

        # ボラティリティ調整
        volatility_factor = 0.0
        if hasattr(market_data, "volatility") and market_data.volatility:
            # 適度なボラティリティで信頼度向上
            if 0.01 <= market_data.volatility <= 0.03:
                volatility_factor = 0.1
            elif market_data.volatility > 0.05:
                volatility_factor = -0.2  # 高ボラティリティで信頼度低下

        # テクニカル調整
        technical_factor = 0.0
        indicators = self.get_latest_indicators()
        if indicators:
            rsi = indicators.get("rsi", 50)
            if 40 <= rsi <= 60:  # 中性的なRSIで信頼度向上
                technical_factor = 0.1

        confidence = (
            base_confidence + time_factor + volatility_factor + technical_factor
        )
        return max(0.0, min(1.0, confidence))

    def _calculate_position_size(
        self, market_data: MarketFrame, confidence: float
    ) -> float:
        """ポジションサイズ計算"""
        # 基本サイズ（設定値または1.0）
        base_size = self.params.get("base_position_size", 1.0)

        # 信頼度によるサイズ調整
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5-1.0の範囲

        # ボラティリティによるサイズ調整
        volatility_multiplier = 1.0
        if hasattr(market_data, "volatility") and market_data.volatility:
            if market_data.volatility > 0.05:  # 高ボラティリティでサイズ削減
                volatility_multiplier = 0.5
            elif market_data.volatility < 0.01:  # 低ボラティリティでサイズ増加
                volatility_multiplier = 1.2

        size = base_size * confidence_multiplier * volatility_multiplier
        return round(size, 4)

    def _calculate_commission(self, pnl: float) -> float:
        """手数料計算（簡易版）"""
        commission_rate = self.params.get("commission_rate", 0.001)  # 0.1%
        return abs(pnl) * commission_rate

    def validate_config(self) -> None:
        """設定値検証"""
        if not (0 <= self.entry_day <= 6):
            raise ValueError("entry_day must be 0-6 (Monday-Sunday)")
        if not (0 <= self.entry_hour <= 23):
            raise ValueError("entry_hour must be 0-23")
        if not (0 <= self.exit_day <= 6):
            raise ValueError("exit_day must be 0-6 (Monday-Sunday)")
        if not (0 <= self.exit_hour <= 23):
            raise ValueError("exit_hour must be 0-23")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be 0.0-1.0")

    def get_required_data(self) -> dict[str, Any]:
        """必要データ仕様"""
        return {
            "ohlcv_timeframes": ["1h"],
            "orderbook_levels": 1,
            "indicators": ["rsi", "sma_20", "sma_50"],
            "lookback_periods": 50,
            "market_data_fields": ["volatility", "volume"],
        }

    async def run_pdca_cycle_demo(self) -> dict[str, Any]:
        """PDCAサイクルデモ実行"""
        logger.info("Starting PDCA cycle demo for Weekend Effect Strategy")

        # Plan フェーズ
        hypothesis = "信頼度閾値を0.7に上げることで勝率が向上し、リスク調整後リターンが改善される"
        target_metrics = {"win_rate": 0.65, "profit_factor": 2.0, "sharpe_ratio": 1.5}
        parameter_changes = {"confidence_threshold": 0.7, "base_position_size": 0.8}

        session_id = await self.start_pdca_cycle(
            hypothesis, target_metrics, parameter_changes
        )

        # Do フェーズ（シミュレーション）
        execution_summary = {
            "trades_executed": 15,
            "execution_period": "2025-06-01 to 2025-06-15",
            "parameter_adjustments": parameter_changes,
        }
        market_conditions = {
            "avg_volatility": 0.025,
            "market_regime": "trending",
            "news_events": ["Fed meeting", "Employment data"],
        }

        await self.log_pdca_execution(15, market_conditions)

        # Check フェーズ
        analysis = await self.analyze_pdca_performance()

        # Act フェーズ
        improvements = await self.complete_pdca_cycle()

        logger.info("PDCA cycle demo completed", session_id=session_id)

        return {
            "session_id": session_id,
            "analysis": analysis,
            "improvements": improvements,
        }


def create_enhanced_weekend_effect_strategy(
    params: dict[str, Any] = None,
) -> EnhancedWeekendEffectStrategy:
    """Enhanced Weekend Effect Strategy ファクトリー関数"""
    default_params = {
        "entry_day": 4,  # 金曜日
        "entry_hour": 23,  # 23:00 UTC
        "exit_day": 0,  # 月曜日
        "exit_hour": 1,  # 01:00 UTC
        "confidence_threshold": 0.6,
        "base_position_size": 1.0,
        "commission_rate": 0.001,
    }

    if params:
        default_params.update(params)

    config = StrategyConfig(
        name="Enhanced Weekend Effect", enabled=True, params=default_params
    )

    return EnhancedWeekendEffectStrategy(config)
