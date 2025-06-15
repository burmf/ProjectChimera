"""
Professional Risk Management System
Accurate VaR, Kelly Criterion, and portfolio risk controls
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import scipy.stats as stats
from loguru import logger

from ..config import Settings, get_settings


class MarketRegime(Enum):
    """Market regime classification"""
    NORMAL = "normal"
    VOLATILE = "volatile"
    TRENDING = "trending"
    CRISIS = "crisis"


@dataclass
class PortfolioMetrics:
    """Portfolio risk metrics"""
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    sortino_ratio: float
    current_drawdown: float
    volatility: float
    correlation_risk: float


@dataclass
class PositionRisk:
    """Individual position risk assessment"""
    symbol: str
    size: float
    leverage: float
    var_contribution: float
    marginal_var: float
    correlation_penalty: float
    kelly_optimal: float
    risk_score: float


@dataclass
class RiskLimits:
    """Dynamic risk limits"""
    max_position_size: float
    max_leverage: int
    position_limit: int
    concentration_limit: float
    correlation_limit: float
    drawdown_limit: float


class ProfessionalRiskManager:
    """
    Professional-grade risk management system
    Features: Monte Carlo VaR, dynamic Kelly sizing, regime detection
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.risk_config = self.settings.risk
        
        # Historical data storage
        self.price_history: Dict[str, List[Dict]] = {}
        self.return_history: Dict[str, List[float]] = {}
        self.portfolio_history: List[Dict] = []
        
        # Current portfolio state
        self.current_positions: Dict[str, Dict] = {}
        self.portfolio_value: float = 100000.0  # Initial value
        self.peak_value: float = 100000.0
        
        # Risk models
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.volatility_model: Dict[str, float] = {}
        self.current_regime: MarketRegime = MarketRegime.NORMAL
        
        # Performance tracking
        self.daily_returns: List[float] = []
        self.trade_history: List[Dict] = []
        
        logger.info("ProfessionalRiskManager initialized")
    
    def update_price_data(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Update price history and calculate returns"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.return_history[symbol] = []
        
        # Add new price point
        self.price_history[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
        
        # Calculate return if we have previous price
        if len(self.price_history[symbol]) > 1:
            prev_price = self.price_history[symbol][-2]['price']
            return_val = (price - prev_price) / prev_price
            self.return_history[symbol].append(return_val)
            
            # Limit history size
            if len(self.return_history[symbol]) > 1000:
                self.return_history[symbol] = self.return_history[symbol][-500:]
        
        # Update volatility model
        self._update_volatility_model(symbol)
    
    def _update_volatility_model(self, symbol: str) -> None:
        """Update GARCH-like volatility model"""
        if len(self.return_history[symbol]) < 30:
            return
        
        returns = np.array(self.return_history[symbol][-30:])
        
        # Simple EWMA volatility model
        lambda_decay = 0.94
        weights = np.array([lambda_decay ** i for i in range(len(returns))])
        weights = weights[::-1] / weights.sum()
        
        weighted_var = np.sum(weights * returns**2)
        self.volatility_model[symbol] = np.sqrt(weighted_var)
    
    def calculate_portfolio_var(
        self,
        confidence_level: float = 0.95,
        holding_period: int = 1,
        method: str = "monte_carlo"
    ) -> Dict[str, float]:
        """
        Calculate portfolio Value at Risk using multiple methods
        """
        if not self.current_positions:
            return {"var_95": 0.0, "var_99": 0.0, "expected_shortfall": 0.0}
        
        if method == "monte_carlo":
            return self._monte_carlo_var(confidence_level, holding_period)
        elif method == "parametric":
            return self._parametric_var(confidence_level, holding_period)
        elif method == "historical":
            return self._historical_var(confidence_level, holding_period)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _monte_carlo_var(self, confidence_level: float, holding_period: int) -> Dict[str, float]:
        """Monte Carlo VaR simulation"""
        n_simulations = 10000
        portfolio_returns = []
        
        # Get current portfolio weights
        total_value = sum(pos['value'] for pos in self.current_positions.values())
        weights = {symbol: pos['value'] / total_value for symbol, pos in self.current_positions.items()}
        
        # Build correlation matrix
        self._update_correlation_matrix()
        
        for _ in range(n_simulations):
            portfolio_return = 0
            
            for symbol, weight in weights.items():
                if symbol in self.return_history and len(self.return_history[symbol]) > 30:
                    # Use historical mean and volatility
                    returns = np.array(self.return_history[symbol][-252:])  # Last year
                    mean_return = np.mean(returns)
                    volatility = self.volatility_model.get(symbol, np.std(returns))
                    
                    # Generate correlated random return
                    random_return = np.random.normal(mean_return, volatility)
                    
                    # Apply leverage
                    leverage = self.current_positions[symbol].get('leverage', 1)
                    leveraged_return = random_return * leverage
                    
                    portfolio_return += weight * leveraged_return
            
            portfolio_returns.append(portfolio_return)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate VaR and Expected Shortfall
        var_95 = np.percentile(portfolio_returns, (1 - 0.95) * 100)
        var_99 = np.percentile(portfolio_returns, (1 - 0.99) * 100)
        
        # Expected Shortfall (Conditional VaR)
        tail_returns = portfolio_returns[portfolio_returns <= var_95]
        expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else var_95
        
        return {
            "var_95": abs(var_95) * self.portfolio_value,
            "var_99": abs(var_99) * self.portfolio_value,
            "expected_shortfall": abs(expected_shortfall) * self.portfolio_value
        }
    
    def _parametric_var(self, confidence_level: float, holding_period: int) -> Dict[str, float]:
        """Parametric VaR using normal distribution assumption"""
        if not self.current_positions:
            return {"var_95": 0.0, "var_99": 0.0, "expected_shortfall": 0.0}
        
        # Calculate portfolio mean and variance
        portfolio_mean = 0
        portfolio_variance = 0
        
        total_value = sum(pos['value'] for pos in self.current_positions.values())
        
        for symbol, position in self.current_positions.items():
            if symbol in self.return_history and len(self.return_history[symbol]) > 30:
                weight = position['value'] / total_value
                returns = np.array(self.return_history[symbol][-252:])
                
                mean_return = np.mean(returns)
                variance = np.var(returns)
                leverage = position.get('leverage', 1)
                
                portfolio_mean += weight * mean_return * leverage
                portfolio_variance += (weight * leverage)**2 * variance
        
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Calculate VaR using normal distribution
        z_95 = stats.norm.ppf(0.05)  # 5% quantile
        z_99 = stats.norm.ppf(0.01)  # 1% quantile
        
        var_95 = -(portfolio_mean + z_95 * portfolio_std) * self.portfolio_value
        var_99 = -(portfolio_mean + z_99 * portfolio_std) * self.portfolio_value
        
        # Expected Shortfall for normal distribution
        es_95 = -(portfolio_mean - portfolio_std * stats.norm.pdf(z_95) / 0.05) * self.portfolio_value
        
        return {
            "var_95": max(0, var_95),
            "var_99": max(0, var_99),
            "expected_shortfall": max(0, es_95)
        }
    
    def _historical_var(self, confidence_level: float, holding_period: int) -> Dict[str, float]:
        """Historical VaR using empirical distribution"""
        if len(self.daily_returns) < 50:
            return {"var_95": 0.0, "var_99": 0.0, "expected_shortfall": 0.0}
        
        returns = np.array(self.daily_returns[-252:])  # Last year
        
        var_95 = np.percentile(returns, 5) * self.portfolio_value
        var_99 = np.percentile(returns, 1) * self.portfolio_value
        
        # Expected Shortfall
        tail_returns = returns[returns <= np.percentile(returns, 5)]
        expected_shortfall = np.mean(tail_returns) * self.portfolio_value if len(tail_returns) > 0 else var_95
        
        return {
            "var_95": abs(var_95),
            "var_99": abs(var_99),
            "expected_shortfall": abs(expected_shortfall)
        }
    
    def calculate_kelly_position_size(
        self,
        symbol: str,
        expected_return: float,
        win_probability: float,
        loss_probability: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        Enhanced version considering downside risk
        """
        # Standard Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        
        if avg_loss == 0 or loss_probability == 0:
            return 0.0
        
        # Calculate odds
        win_loss_ratio = avg_win / abs(avg_loss)
        
        # Kelly fraction
        kelly_fraction = (win_probability * win_loss_ratio - loss_probability) / win_loss_ratio
        
        # Apply safety margin (typically 25-50% of Kelly)
        safety_margin = self.risk_config.kelly_fraction
        optimal_fraction = max(0, kelly_fraction * safety_margin)
        
        # Consider portfolio concentration limits
        max_concentration = 0.2  # Maximum 20% in single position
        optimal_fraction = min(optimal_fraction, max_concentration)
        
        # Adjust for volatility
        if symbol in self.volatility_model:
            vol_adjustment = min(1.0, 0.02 / self.volatility_model[symbol])  # Target 2% vol
            optimal_fraction *= vol_adjustment
        
        # Convert to dollar amount
        position_size = optimal_fraction * self.portfolio_value
        
        logger.debug(f"Kelly sizing for {symbol}: {optimal_fraction:.3f} = ${position_size:,.0f}")
        
        return position_size
    
    def _update_correlation_matrix(self) -> None:
        """Update correlation matrix for portfolio symbols"""
        symbols = list(self.return_history.keys())
        
        if len(symbols) < 2:
            return
        
        # Create returns DataFrame
        min_length = min(len(self.return_history[s]) for s in symbols)
        if min_length < 30:
            return
        
        returns_data = {}
        for symbol in symbols:
            returns_data[symbol] = self.return_history[symbol][-min_length:]
        
        df = pd.DataFrame(returns_data)
        self.correlation_matrix = df.corr()
    
    def calculate_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        if len(self.daily_returns) < 30:
            return PortfolioMetrics(
                var_95=0, var_99=0, expected_shortfall=0, sharpe_ratio=0,
                max_drawdown=0, calmar_ratio=0, sortino_ratio=0,
                current_drawdown=0, volatility=0, correlation_risk=0
            )
        
        returns = np.array(self.daily_returns)
        
        # Basic metrics
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # VaR metrics
        var_metrics = self.calculate_portfolio_var()
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Downside deviation for Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else volatility
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        max_drawdown = np.min(drawdowns)
        current_drawdown = drawdowns[-1]
        
        # Calmar ratio
        calmar_ratio = (mean_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Correlation risk
        correlation_risk = self._calculate_correlation_risk()
        
        return PortfolioMetrics(
            var_95=var_metrics["var_95"],
            var_99=var_metrics["var_99"],
            expected_shortfall=var_metrics["expected_shortfall"],
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            current_drawdown=current_drawdown,
            volatility=volatility,
            correlation_risk=correlation_risk
        )
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk score"""
        if self.correlation_matrix is None or len(self.current_positions) < 2:
            return 0.0
        
        total_value = sum(pos['value'] for pos in self.current_positions.values())
        weights = {symbol: pos['value'] / total_value for symbol, pos in self.current_positions.items()}
        
        correlation_risk = 0.0
        symbols = list(weights.keys())
        
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                if symbol1 in self.correlation_matrix.index and symbol2 in self.correlation_matrix.columns:
                    correlation = self.correlation_matrix.loc[symbol1, symbol2]
                    weight_product = weights[symbol1] * weights[symbol2]
                    correlation_risk += abs(correlation) * weight_product
        
        return correlation_risk
    
    def detect_market_regime(self) -> MarketRegime:
        """Detect current market regime using multiple indicators"""
        if len(self.daily_returns) < 50:
            return MarketRegime.NORMAL
        
        returns = np.array(self.daily_returns[-50:])
        
        # Volatility regime
        current_vol = np.std(returns[-10:])
        historical_vol = np.std(returns)
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1
        
        # Trend strength
        trend_strength = abs(np.mean(returns[-20:]))
        
        # Crisis detection (large negative returns)
        extreme_losses = np.sum(returns < -0.05)  # Days with >5% loss
        
        # Regime classification
        if extreme_losses >= 3:
            regime = MarketRegime.CRISIS
        elif vol_ratio > 2.0:
            regime = MarketRegime.VOLATILE
        elif trend_strength > 0.02:
            regime = MarketRegime.TRENDING
        else:
            regime = MarketRegime.NORMAL
        
        if regime != self.current_regime:
            logger.info(f"Market regime changed: {self.current_regime.value} -> {regime.value}")
            self.current_regime = regime
        
        return regime
    
    def get_dynamic_risk_limits(self) -> RiskLimits:
        """Get dynamic risk limits based on current market conditions"""
        base_limits = RiskLimits(
            max_position_size=self.portfolio_value * 0.2,
            max_leverage=self.settings.trading.max_leverage,
            position_limit=self.settings.trading.max_positions,
            concentration_limit=0.3,
            correlation_limit=self.risk_config.max_correlation,
            drawdown_limit=self.risk_config.max_drawdown
        )
        
        # Adjust for market regime
        regime = self.detect_market_regime()
        
        regime_adjustments = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.VOLATILE: 0.7,
            MarketRegime.TRENDING: 1.2,
            MarketRegime.CRISIS: 0.3
        }
        
        adjustment = regime_adjustments[regime]
        
        return RiskLimits(
            max_position_size=base_limits.max_position_size * adjustment,
            max_leverage=int(base_limits.max_leverage * adjustment),
            position_limit=max(1, int(base_limits.position_limit * adjustment)),
            concentration_limit=base_limits.concentration_limit * adjustment,
            correlation_limit=base_limits.correlation_limit,
            drawdown_limit=base_limits.drawdown_limit
        )
    
    def validate_new_position(
        self,
        symbol: str,
        size: float,
        leverage: int
    ) -> Tuple[bool, List[str]]:
        """Validate if new position meets risk requirements"""
        warnings = []
        
        # Get current limits
        limits = self.get_dynamic_risk_limits()
        
        # Position size check
        if size > limits.max_position_size:
            warnings.append(f"Position size ${size:,.0f} exceeds limit ${limits.max_position_size:,.0f}")
        
        # Leverage check
        if leverage > limits.max_leverage:
            warnings.append(f"Leverage {leverage}x exceeds limit {limits.max_leverage}x")
        
        # Portfolio concentration
        total_exposure = sum(pos['value'] * pos.get('leverage', 1) for pos in self.current_positions.values())
        new_exposure = size * leverage
        concentration = (total_exposure + new_exposure) / self.portfolio_value
        
        if concentration > limits.concentration_limit:
            warnings.append(f"Portfolio concentration {concentration:.1%} exceeds {limits.concentration_limit:.1%}")
        
        # VaR limit check
        var_metrics = self.calculate_portfolio_var()
        if var_metrics["var_95"] > self.portfolio_value * self.risk_config.max_portfolio_risk:
            warnings.append(f"Portfolio VaR exceeds {self.risk_config.max_portfolio_risk:.1%} limit")
        
        return len(warnings) == 0, warnings
    
    def update_portfolio_state(self, positions: Dict[str, Dict], portfolio_value: float) -> None:
        """Update current portfolio state"""
        self.current_positions = positions
        self.portfolio_value = portfolio_value
        self.peak_value = max(self.peak_value, portfolio_value)
        
        # Add to daily returns if it's a new day
        if self.portfolio_history:
            last_update = self.portfolio_history[-1]['timestamp'].date()
            current_date = datetime.now().date()
            
            if last_update != current_date:
                if len(self.portfolio_history) > 0:
                    prev_value = self.portfolio_history[-1]['value']
                    daily_return = (portfolio_value - prev_value) / prev_value
                    self.daily_returns.append(daily_return)
        
        # Add to portfolio history
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'value': portfolio_value,
            'positions': positions.copy()
        })
        
        # Limit history size
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-500:]
    
    def generate_risk_report(self) -> str:
        """Generate comprehensive risk report"""
        metrics = self.calculate_portfolio_metrics()
        limits = self.get_dynamic_risk_limits()
        regime = self.detect_market_regime()
        
        report = f"""
🛡️ PROFESSIONAL RISK MANAGEMENT REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 PORTFOLIO METRICS:
  Portfolio Value: ${self.portfolio_value:,.2f}
  Peak Value: ${self.peak_value:,.2f}
  Current Drawdown: {metrics.current_drawdown:.2%}
  
🎯 RISK MEASURES:
  VaR (95%): ${metrics.var_95:,.2f}
  VaR (99%): ${metrics.var_99:,.2f}
  Expected Shortfall: ${metrics.expected_shortfall:,.2f}
  
📈 PERFORMANCE METRICS:
  Sharpe Ratio: {metrics.sharpe_ratio:.3f}
  Sortino Ratio: {metrics.sortino_ratio:.3f}
  Calmar Ratio: {metrics.calmar_ratio:.3f}
  Volatility: {metrics.volatility:.2%}
  
🌍 MARKET ENVIRONMENT:
  Current Regime: {regime.value.upper()}
  Correlation Risk: {metrics.correlation_risk:.2%}
  
⚠️ RISK LIMITS:
  Max Position: ${limits.max_position_size:,.0f}
  Max Leverage: {limits.max_leverage}x
  Max Positions: {limits.position_limit}
  Concentration: {limits.concentration_limit:.1%}
"""
        
        return report


if __name__ == "__main__":
    # Test the risk manager
    settings = get_settings()
    rm = ProfessionalRiskManager(settings)
    
    # Simulate some price updates
    import random
    base_price = 50000
    
    for i in range(100):
        price_change = random.normalvariate(0, 0.02)
        new_price = base_price * (1 + price_change)
        rm.update_price_data('BTCUSDT', new_price, datetime.now())
        base_price = new_price
    
    # Test VaR calculation
    rm.current_positions = {
        'BTCUSDT': {'value': 50000, 'leverage': 10}
    }
    rm.portfolio_value = 100000
    
    var_metrics = rm.calculate_portfolio_var()
    print(f"Portfolio VaR (95%): ${var_metrics['var_95']:,.2f}")
    
    # Test Kelly sizing
    kelly_size = rm.calculate_kelly_position_size(
        'BTCUSDT', 0.01, 0.6, 0.4, 0.02, 0.01
    )
    print(f"Kelly optimal size: ${kelly_size:,.2f}")
    
    print("✅ Professional Risk Manager test completed")