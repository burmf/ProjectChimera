"""
戦略用パフォーマンス測定ミックスイン
各戦略に自動的にパフォーマンス測定機能を追加
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import asdict
import structlog

from ..monitor.performance_logger import (
    get_performance_logger, TradeExecution, TradeResult, PerformancePhase
)
from ..domains.market import MarketFrame, Signal

logger = structlog.get_logger(__name__)


class PerformanceMixin:
    """
    戦略用パフォーマンス測定ミックスイン
    
    自動的に以下を測定・記録:
    1. シグナル生成時間
    2. 取引実行データ
    3. パフォーマンス指標
    4. PDCAサイクル管理
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_logger = get_performance_logger()
        
        # パフォーマンス測定用変数
        self._signal_start_time = 0.0
        self._execution_start_time = 0.0
        self._current_trade_context = {}
        
        # PDCA管理
        self._pdca_session_id: Optional[str] = None
        self._current_hypothesis = ""
        self._target_metrics = {}
        
        logger.info("Performance monitoring enabled for strategy", 
                   strategy=self.__class__.__name__)
    
    def start_signal_timing(self) -> None:
        """シグナル生成タイミング開始"""
        self._signal_start_time = time.time()
    
    def end_signal_timing(self) -> float:
        """シグナル生成タイミング終了"""
        if self._signal_start_time > 0:
            duration = (time.time() - self._signal_start_time) * 1000  # ms
            self._signal_start_time = 0.0
            return duration
        return 0.0
    
    def start_execution_timing(self) -> None:
        """実行タイミング開始"""
        self._execution_start_time = time.time()
    
    def end_execution_timing(self) -> float:
        """実行タイミング終了"""
        if self._execution_start_time > 0:
            duration = (time.time() - self._execution_start_time) * 1000  # ms
            self._execution_start_time = 0.0
            return duration
        return 0.0
    
    def set_trade_context(self, context: Dict[str, Any]) -> None:
        """取引コンテキスト設定"""
        self._current_trade_context = context.copy()
    
    def update_trade_context(self, key: str, value: Any) -> None:
        """取引コンテキスト更新"""
        self._current_trade_context[key] = value
    
    async def log_signal_generated(self, signal: Signal, market_frame: MarketFrame,
                                 signal_time_ms: float, context: Dict[str, Any] = None) -> str:
        """シグナル生成ログ"""
        trade_id = f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 基本取引データ作成
        trade_execution = TradeExecution(
            trade_id=trade_id,
            strategy_name=self.__class__.__name__,
            timestamp=datetime.now(),
            symbol=market_frame.symbol,
            signal_type=signal.signal_type.value,
            signal_confidence=signal.confidence,
            signal_generation_time_ms=signal_time_ms,
            entry_price=signal.price,
            quantity=signal.size,
            market_volatility=getattr(market_frame, 'volatility', None),
            spread_bps=self._calculate_spread_bps(market_frame),
            volume_at_signal=getattr(market_frame, 'volume', None),
            strategy_context=context or self._current_trade_context.copy()
        )
        
        # 非同期でログ記録
        await self.performance_logger.log_trade_execution(trade_execution)
        
        logger.info("Signal generated and logged", 
                   trade_id=trade_id,
                   strategy=self.__class__.__name__,
                   signal_type=signal.signal_type.value,
                   confidence=signal.confidence)
        
        return trade_id
    
    async def log_trade_completed(self, trade_id: str, exit_price: float, 
                                result: TradeResult, actual_pnl: float,
                                hold_duration_seconds: float,
                                execution_time_ms: float = 0.0,
                                commission: float = 0.0) -> None:
        """取引完了ログ"""
        # 既存の取引記録を更新
        for trade in self.performance_logger.trade_history:
            if trade.trade_id == trade_id:
                trade.exit_price = exit_price
                trade.result = result
                trade.pnl = actual_pnl
                trade.pnl_percentage = (actual_pnl / (trade.entry_price * trade.quantity)) * 100 if trade.quantity > 0 else 0
                trade.hold_duration_seconds = hold_duration_seconds
                trade.total_execution_time_ms = execution_time_ms
                trade.commission = commission
                
                # 更新後の取引をログ
                await self.performance_logger.log_trade_execution(trade)
                
                logger.info("Trade completed and logged", 
                           trade_id=trade_id,
                           result=result.value,
                           pnl=actual_pnl,
                           hold_duration_min=hold_duration_seconds/60)
                break
    
    async def start_pdca_cycle(self, hypothesis: str, target_metrics: Dict[str, float],
                             parameter_changes: Dict[str, Any] = None) -> str:
        """PDCAサイクル開始"""
        self._current_hypothesis = hypothesis
        self._target_metrics = target_metrics.copy()
        
        self._pdca_session_id = await self.performance_logger.start_pdca_cycle(
            strategy_name=self.__class__.__name__,
            hypothesis=hypothesis,
            target_metrics=target_metrics,
            parameter_changes=parameter_changes or {}
        )
        
        logger.info("PDCA cycle started", 
                   strategy=self.__class__.__name__,
                   session_id=self._pdca_session_id,
                   hypothesis=hypothesis)
        
        return self._pdca_session_id
    
    async def log_pdca_execution(self, trades_executed: int, market_conditions: Dict[str, Any]) -> None:
        """PDCA実行フェーズログ"""
        if not self._pdca_session_id:
            logger.warning("PDCA session not started")
            return
        
        execution_summary = {
            "trades_executed": trades_executed,
            "execution_period_start": datetime.now().isoformat(),
            "strategy_parameters": self._get_current_parameters(),
            "execution_notes": f"実行期間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
        
        await self.performance_logger.log_execution_phase(
            strategy_name=self.__class__.__name__,
            execution_summary=execution_summary,
            market_conditions=market_conditions
        )
        
        logger.info("PDCA execution phase logged",
                   trades_executed=trades_executed)
    
    async def analyze_pdca_performance(self) -> Dict[str, Any]:
        """PDCAパフォーマンス分析"""
        if not self._pdca_session_id:
            logger.warning("PDCA session not started")
            return {}
        
        analysis = await self.performance_logger.analyze_performance(self.__class__.__name__)
        
        logger.info("PDCA performance analysis completed",
                   strategy=self.__class__.__name__)
        
        return analysis
    
    async def complete_pdca_cycle(self) -> Dict[str, Any]:
        """PDCAサイクル完了"""
        if not self._pdca_session_id:
            logger.warning("PDCA session not started")
            return {}
        
        # 分析実行
        analysis = await self.analyze_pdca_performance()
        
        # 改善提案生成
        improvements = await self.performance_logger.generate_improvements(
            self.__class__.__name__, analysis
        )
        
        logger.info("PDCA cycle completed",
                   strategy=self.__class__.__name__,
                   session_id=self._pdca_session_id,
                   improvements_count=len(improvements.get("insights", [])))
        
        # セッションリセット
        self._pdca_session_id = None
        self._current_hypothesis = ""
        self._target_metrics = {}
        
        return {
            "analysis": analysis,
            "improvements": improvements
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス要約取得"""
        strategy_metrics = self.performance_logger.strategy_metrics.get(self.__class__.__name__)
        
        if not strategy_metrics:
            return {"status": "no_data"}
        
        return {
            "strategy_name": self.__class__.__name__,
            "total_trades": strategy_metrics.total_trades,
            "win_rate": strategy_metrics.win_rate,
            "total_pnl": strategy_metrics.total_pnl,
            "profit_factor": strategy_metrics.profit_factor,
            "avg_signal_time_ms": strategy_metrics.signal_generation_time_ms,
            "avg_execution_time_ms": strategy_metrics.order_execution_time_ms,
            "last_updated": strategy_metrics.timestamp.isoformat()
        }
    
    def _calculate_spread_bps(self, market_frame: MarketFrame) -> Optional[float]:
        """スプレッドをbpsで計算"""
        try:
            if hasattr(market_frame, 'bid') and hasattr(market_frame, 'ask'):
                spread = market_frame.ask - market_frame.bid
                mid_price = (market_frame.ask + market_frame.bid) / 2
                spread_bps = (spread / mid_price) * 10000  # bps
                return spread_bps
        except:
            pass
        return None
    
    def _get_current_parameters(self) -> Dict[str, Any]:
        """現在のストラテジーパラメータ取得"""
        # 基本パラメータ（サブクラスでオーバーライド推奨）
        params = {
            "strategy_class": self.__class__.__name__,
            "enabled": getattr(self, 'enabled', True)
        }
        
        # config属性からパラメータ抽出
        if hasattr(self, 'config'):
            params.update(getattr(self.config, 'params', {}))
        
        # 動的パラメータ抽出
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, (int, float, str, bool)):
                    params[attr_name] = attr_value
        
        return params


class PerformanceDecoratorMixin:
    """
    デコレーターベースのパフォーマンス測定ミックスイン
    メソッドデコレーターでパフォーマンス測定を自動化
    """
    
    @staticmethod
    def measure_signal_generation(func):
        """シグナル生成メソッド用デコレーター"""
        async def wrapper(self, *args, **kwargs):
            self.start_signal_timing()
            try:
                result = await func(self, *args, **kwargs) if asyncio.iscoroutinefunction(func) else func(self, *args, **kwargs)
                signal_time = self.end_signal_timing()
                
                # シグナルが生成された場合、ログ記録
                if result and hasattr(result, 'signal_type'):
                    market_frame = args[0] if args else None
                    if market_frame:
                        trade_id = await self.log_signal_generated(
                            signal=result,
                            market_frame=market_frame,
                            signal_time_ms=signal_time
                        )
                        # trade_idを結果に追加
                        if hasattr(result, '__dict__'):
                            result.__dict__['_trade_id'] = trade_id
                
                return result
            except Exception as e:
                self.end_signal_timing()
                logger.error("Error in signal generation", 
                           strategy=self.__class__.__name__, 
                           error=str(e))
                raise
        
        return wrapper
    
    @staticmethod
    def measure_execution(func):
        """実行メソッド用デコレーター"""
        async def wrapper(self, *args, **kwargs):
            self.start_execution_timing()
            try:
                result = await func(self, *args, **kwargs) if asyncio.iscoroutinefunction(func) else func(self, *args, **kwargs)
                execution_time = self.end_execution_timing()
                
                # 実行時間をコンテキストに保存
                self.update_trade_context('execution_time_ms', execution_time)
                
                return result
            except Exception as e:
                self.end_execution_timing()
                logger.error("Error in execution", 
                           strategy=self.__class__.__name__, 
                           error=str(e))
                raise
        
        return wrapper