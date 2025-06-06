# core/risk_manager.py
import logging
import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import sys
import os

# Add core modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.redis_manager import redis_manager
from core.database_adapter import db_adapter

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self):
        self.max_daily_loss_pct = 5.0  # Maximum 5% daily loss
        self.max_position_risk_pct = 1.0  # Maximum 1% risk per position
        self.max_open_positions = 5  # Maximum open positions
        self.max_correlation_exposure = 0.7  # Maximum correlation exposure
        self.drawdown_limit_pct = 10.0  # Maximum drawdown before stopping
        
        # Dynamic risk parameters
        self.current_equity = 1_000_000  # Starting equity
        self.peak_equity = 1_000_000
        self.daily_pnl = 0.0
        self.open_positions = []
        
    def evaluate_trade_risk(self, trade_signal: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate if a trade signal meets risk criteria"""
        try:
            risk_assessment = {
                'approved': False,
                'adjusted_lot_size': 0.0,
                'risk_score': 0.0,
                'reasons': [],
                'warnings': []
            }
            
            pair = trade_signal.get('pair')
            direction = trade_signal.get('direction')
            confidence = trade_signal.get('confidence', 0)
            suggested_lot = trade_signal.get('lot_size_factor', 0.1)
            
            # 1. Check daily loss limit
            if not self.check_daily_loss_limit():
                risk_assessment['reasons'].append('Daily loss limit exceeded')
                return False, risk_assessment
            
            # 2. Check drawdown limit
            if not self.check_drawdown_limit():
                risk_assessment['reasons'].append('Maximum drawdown limit exceeded')
                return False, risk_assessment
            
            # 3. Check maximum open positions
            if not self.check_position_limit():
                risk_assessment['reasons'].append('Maximum open positions limit reached')
                return False, risk_assessment
            
            # 4. Check correlation risk
            correlation_risk = self.check_correlation_risk(pair, direction)
            if correlation_risk > self.max_correlation_exposure:
                risk_assessment['reasons'].append(f'Correlation risk too high: {correlation_risk:.2f}')
                return False, risk_assessment
            
            # 5. Calculate position size based on risk
            position_size = self.calculate_position_size(
                pair, 
                trade_signal.get('stop_loss_pips', 50),
                confidence
            )
            
            if position_size <= 0:
                risk_assessment['reasons'].append('Calculated position size too small')
                return False, risk_assessment
            
            # 6. Check minimum confidence threshold
            min_confidence = 0.6
            if confidence < min_confidence:
                risk_assessment['reasons'].append(f'Confidence too low: {confidence:.2f} < {min_confidence}')
                return False, risk_assessment
            
            # Trade approved
            risk_assessment.update({
                'approved': True,
                'adjusted_lot_size': position_size,
                'risk_score': self.calculate_risk_score(trade_signal, position_size),
                'correlation_exposure': correlation_risk,
                'position_risk_pct': (position_size * trade_signal.get('stop_loss_pips', 50)) / self.current_equity * 100
            })
            
            # Add warnings for moderate risks
            if correlation_risk > 0.5:
                risk_assessment['warnings'].append(f'Moderate correlation risk: {correlation_risk:.2f}')
            
            if confidence < 0.75:
                risk_assessment['warnings'].append(f'Moderate confidence: {confidence:.2f}')
            
            logger.info(f"Trade approved for {pair} {direction} - Size: {position_size:.4f}")
            return True, risk_assessment
            
        except Exception as e:
            logger.error(f"Risk evaluation error: {e}")
            return False, {'approved': False, 'reasons': ['Risk evaluation failed']}
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit is exceeded"""
        try:
            daily_loss_pct = abs(self.daily_pnl) / self.current_equity * 100
            return daily_loss_pct < self.max_daily_loss_pct
        except:
            return True  # Allow trade if calculation fails
    
    def check_drawdown_limit(self) -> bool:
        """Check if maximum drawdown limit is exceeded"""
        try:
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity * 100
            return current_drawdown < self.drawdown_limit_pct
        except:
            return True
    
    def check_position_limit(self) -> bool:
        """Check if maximum open positions limit is reached"""
        try:
            current_positions = len(self.get_open_positions())
            return current_positions < self.max_open_positions
        except:
            return True
    
    def check_correlation_risk(self, pair: str, direction: str) -> float:
        """Calculate correlation risk with existing positions"""
        try:
            open_positions = self.get_open_positions()
            
            if not open_positions:
                return 0.0
            
            # Simple correlation check based on currency pairs
            base_currency = pair.split('/')[0] if '/' in pair else pair[:3]
            same_currency_exposure = 0.0
            same_direction_exposure = 0.0
            
            for pos in open_positions:
                pos_pair = pos.get('pair', '')
                pos_direction = pos.get('direction', '')
                pos_size = pos.get('lot_size', 0)
                
                # Check same base currency
                if pos_pair.startswith(base_currency):
                    same_currency_exposure += pos_size
                
                # Check same direction
                if pos_direction == direction:
                    same_direction_exposure += pos_size
            
            # Normalize to total equity
            total_exposure = (same_currency_exposure + same_direction_exposure) / self.current_equity
            return min(total_exposure, 1.0)
            
        except Exception as e:
            logger.error(f"Correlation risk calculation error: {e}")
            return 0.5  # Return moderate risk if calculation fails
    
    def calculate_position_size(self, pair: str, stop_loss_pips: int, confidence: float) -> float:
        """Calculate optimal position size based on risk parameters"""
        try:
            # Base risk amount (1% of equity)
            risk_amount = self.current_equity * (self.max_position_risk_pct / 100)
            
            # Adjust based on confidence
            confidence_multiplier = max(0.1, min(1.0, confidence))
            adjusted_risk = risk_amount * confidence_multiplier
            
            # Calculate position size based on stop loss
            pip_value = self.get_pip_value(pair)
            stop_loss_amount = stop_loss_pips * pip_value
            
            if stop_loss_amount <= 0:
                return 0.0
            
            position_size = adjusted_risk / stop_loss_amount
            
            # Apply minimum and maximum limits
            min_size = 0.001  # Minimum position size
            max_size = self.current_equity * 0.1  # Maximum 10% of equity
            
            return max(min_size, min(position_size, max_size))
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0.001  # Return minimum size if calculation fails
    
    def get_pip_value(self, pair: str) -> float:
        """Get pip value for a currency pair"""
        # Simplified pip value calculation
        if 'JPY' in pair.upper():
            return 100.0  # 1 pip = 0.01 for JPY pairs
        elif 'BTC' in pair.upper():
            return 1.0    # 1 pip = 1 USD for BTC
        else:
            return 10000.0  # 1 pip = 0.0001 for major pairs
    
    def calculate_risk_score(self, trade_signal: Dict, position_size: float) -> float:
        """Calculate overall risk score for the trade"""
        try:
            confidence = trade_signal.get('confidence', 0)
            stop_loss_pips = trade_signal.get('stop_loss_pips', 50)
            
            # Base risk score (lower is better)
            risk_score = 0.0
            
            # Confidence factor (higher confidence = lower risk)
            risk_score += (1 - confidence) * 30
            
            # Stop loss factor (wider stop = higher risk)
            risk_score += min(stop_loss_pips / 100, 0.5) * 20
            
            # Position size factor
            size_risk = position_size / (self.current_equity * 0.01)  # Relative to 1% risk
            risk_score += min(size_risk, 2.0) * 15
            
            # Correlation factor
            correlation = self.check_correlation_risk(
                trade_signal.get('pair', ''), 
                trade_signal.get('direction', '')
            )
            risk_score += correlation * 25
            
            # Current drawdown factor
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            risk_score += drawdown * 10
            
            return min(risk_score, 100.0)  # Cap at 100
            
        except Exception as e:
            logger.error(f"Risk score calculation error: {e}")
            return 50.0  # Return moderate risk if calculation fails
    
    def get_open_positions(self) -> List[Dict]:
        """Get current open positions"""
        try:
            # Try to get from Redis cache first
            cached_positions = redis_manager.get_cache('open_positions')
            if cached_positions:
                return cached_positions
            
            # Fallback to empty list (would integrate with actual trading system)
            return []
            
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []
    
    def update_portfolio_metrics(self):
        """Update current portfolio metrics"""
        try:
            # Get recent trade history
            trades_df = db_adapter.execute_query(
                f"SELECT * FROM {db_adapter.get_table_prefix()}trade_history "
                f"WHERE exit_time >= date('now', '-1 day') "
                f"ORDER BY exit_time DESC LIMIT 100"
            )
            
            if trades_df is not None and not trades_df.empty:
                # Calculate daily P&L
                today_trades = trades_df[trades_df['exit_time'].str.contains(datetime.date.today().isoformat())]
                self.daily_pnl = today_trades['pnl_currency'].sum() if not today_trades.empty else 0.0
                
                # Update current equity
                total_pnl = trades_df['pnl_currency'].sum()
                self.current_equity = 1_000_000 + total_pnl  # Starting equity + P&L
                
                # Update peak equity
                self.peak_equity = max(self.peak_equity, self.current_equity)
            
            # Cache metrics in Redis
            metrics = {
                'current_equity': self.current_equity,
                'peak_equity': self.peak_equity,
                'daily_pnl': self.daily_pnl,
                'current_drawdown': (self.peak_equity - self.current_equity) / self.peak_equity * 100,
                'updated_at': datetime.datetime.now().isoformat()
            }
            
            redis_manager.set_cache('portfolio_metrics', metrics, ttl=300)
            
        except Exception as e:
            logger.error(f"Portfolio metrics update error: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk management summary"""
        try:
            self.update_portfolio_metrics()
            
            open_positions = self.get_open_positions()
            current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity * 100
            daily_loss_pct = abs(self.daily_pnl) / self.current_equity * 100
            
            return {
                'risk_limits': {
                    'max_daily_loss_pct': self.max_daily_loss_pct,
                    'max_position_risk_pct': self.max_position_risk_pct,
                    'max_open_positions': self.max_open_positions,
                    'drawdown_limit_pct': self.drawdown_limit_pct
                },
                'current_status': {
                    'current_equity': self.current_equity,
                    'peak_equity': self.peak_equity,
                    'daily_pnl': self.daily_pnl,
                    'daily_loss_pct': daily_loss_pct,
                    'current_drawdown': current_drawdown,
                    'open_positions_count': len(open_positions),
                    'trading_allowed': (
                        daily_loss_pct < self.max_daily_loss_pct and
                        current_drawdown < self.drawdown_limit_pct and
                        len(open_positions) < self.max_open_positions
                    )
                },
                'open_positions': open_positions,
                'risk_utilization': {
                    'daily_loss_utilization': daily_loss_pct / self.max_daily_loss_pct,
                    'drawdown_utilization': current_drawdown / self.drawdown_limit_pct,
                    'position_utilization': len(open_positions) / self.max_open_positions
                }
            }
            
        except Exception as e:
            logger.error(f"Risk summary generation error: {e}")
            return {
                'error': str(e),
                'current_status': {
                    'trading_allowed': False
                }
            }
    
    def emergency_stop(self, reason: str):
        """Emergency stop all trading activities"""
        try:
            stop_signal = {
                'action': 'emergency_stop',
                'reason': reason,
                'timestamp': datetime.datetime.now().isoformat(),
                'portfolio_state': {
                    'equity': self.current_equity,
                    'drawdown': (self.peak_equity - self.current_equity) / self.peak_equity * 100,
                    'daily_pnl': self.daily_pnl
                }
            }
            
            # Broadcast emergency stop
            redis_manager.publish('emergency_stop', stop_signal)
            redis_manager.add_to_stream('risk_events', stop_signal)
            
            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")

# Global risk manager instance
risk_manager = RiskManager()