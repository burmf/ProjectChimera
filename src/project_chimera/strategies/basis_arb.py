"""
Basis Arbitrage Strategy (BASIS_ARB)
Exploits pricing differences: Spot ↔ Perp premium > 0.5% → arbitrage opportunity
"""

import statistics
from typing import Any

from ..domains.market import MarketFrame, Signal, SignalType
from .base import Strategy, StrategyConfig


class BasisArbitrageStrategy(Strategy):
    """
    Basis Arbitrage Strategy

    Core trigger: Spot ↔ Perp premium > 0.5% → arbitrage opportunity
    Exploits pricing inefficiencies between spot and perpetual futures
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def validate_config(self) -> None:
        """Validate strategy configuration"""
        # Set default parameters
        self.params.setdefault("min_basis_pct", 0.5)  # Minimum basis threshold
        self.params.setdefault(
            "max_basis_pct", 5.0
        )  # Maximum basis (avoid extreme events)
        self.params.setdefault("funding_rate_factor", 3.0)  # Funding rate consideration
        self.params.setdefault("min_liquidity_ratio", 0.1)  # Min liquidity vs avg
        self.params.setdefault(
            "convergence_timeframe", 24
        )  # Expected convergence hours
        self.params.setdefault("stop_loss_pct", 1.0)  # Stop loss %
        self.params.setdefault("take_profit_pct", 0.8)  # Take profit % (80% of basis)
        self.params.setdefault("max_position_hours", 48)  # Max hold time
        self.params.setdefault("volume_confirmation", True)  # Require volume
        self.params.setdefault("min_notional_value", 10000)  # Min trade size USD

        # Validate ranges
        if self.params["min_basis_pct"] <= 0:
            raise ValueError("min_basis_pct must be positive")
        if self.params["max_basis_pct"] <= self.params["min_basis_pct"]:
            raise ValueError("max_basis_pct must be > min_basis_pct")
        if self.params["funding_rate_factor"] <= 0:
            raise ValueError("funding_rate_factor must be positive")
        if self.params["convergence_timeframe"] <= 0:
            raise ValueError("convergence_timeframe must be positive")

    def get_required_data(self) -> dict[str, Any]:
        """Specify required market data"""
        return {
            "ohlcv_timeframes": ["5m"],
            "orderbook_levels": 5,  # Need orderbook for liquidity
            "indicators": [],
            "spot_price_data": True,  # Need spot prices
            "perpetual_price_data": True,  # Need perp prices
            "funding_rate_data": True,  # Need funding rates
            "lookback_periods": 100,
        }

    async def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """Generate basis arbitrage signal"""
        # Check if we have required spot and perp data
        if not hasattr(market_data, "spot_price") or not hasattr(
            market_data, "perpetual_price"
        ):
            return None

        spot_price = getattr(market_data, "spot_price", None)
        perp_price = getattr(market_data, "perpetual_price", None)
        funding_rate = getattr(market_data, "funding_rate", 0.0)

        if spot_price is None or perp_price is None:
            return None

        spot_price = float(spot_price)
        perp_price = float(perp_price)

        if spot_price == 0:
            return None

        # Calculate basis (premium/discount)
        basis_pct = ((perp_price - spot_price) / spot_price) * 100

        # Check if basis is significant enough
        if abs(basis_pct) < self.params["min_basis_pct"]:
            return None

        if abs(basis_pct) > self.params["max_basis_pct"]:
            return None  # Too extreme, likely due to market stress

        # Check funding rate implications
        funding_adjusted_basis = self._adjust_basis_for_funding(basis_pct, funding_rate)
        if abs(funding_adjusted_basis) < self.params["min_basis_pct"]:
            return None

        # Check liquidity in both markets
        liquidity_ok = self._check_liquidity(market_data)
        if not liquidity_ok:
            return None

        # Volume confirmation
        if self.params["volume_confirmation"]:
            volume_ok = self._check_volume_confirmation(market_data)
            if not volume_ok:
                return None

        # Check if trade size meets minimum notional
        position_value = spot_price * 0.02  # Assume 2% position
        if position_value < self.params["min_notional_value"]:
            return None

        # Determine arbitrage direction
        if basis_pct > 0:
            # Perp premium → sell perp, buy spot
            signal_type = SignalType.SELL  # Short the overpriced perp
            reasoning = f"Basis arb SHORT perp: {basis_pct:.2f}% premium, funding {funding_rate*100:.3f}%"
            target_convergence = spot_price
            stop_loss = perp_price * (1 + self.params["stop_loss_pct"] / 100)
        else:
            # Perp discount → buy perp, sell spot
            signal_type = SignalType.BUY  # Long the underpriced perp
            reasoning = f"Basis arb LONG perp: {abs(basis_pct):.2f}% discount, funding {funding_rate*100:.3f}%"
            target_convergence = spot_price
            stop_loss = perp_price * (1 - self.params["stop_loss_pct"] / 100)

        # Calculate take profit (partial convergence)
        convergence_target = spot_price + (perp_price - spot_price) * (
            1 - self.params["take_profit_pct"] / 100
        )

        # Calculate confidence based on basis size and funding rate alignment
        basis_strength = min(1.0, abs(basis_pct) / 2.0)
        funding_alignment = self._calculate_funding_alignment(basis_pct, funding_rate)
        confidence = min(0.8, 0.5 + basis_strength * 0.2 + funding_alignment * 0.1)

        return Signal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            confidence=confidence,
            target_size=0.02,  # 2% position size
            entry_price=perp_price,  # Trade the perp
            stop_loss=stop_loss,
            take_profit=convergence_target,
            timeframe="5m",
            strategy_id="BASIS_ARB",
            reasoning=reasoning,
            metadata={
                "basis_percentage": basis_pct,
                "spot_price": spot_price,
                "perp_price": perp_price,
                "funding_rate": funding_rate,
                "funding_adjusted_basis": funding_adjusted_basis,
                "expected_convergence_hours": self.params["convergence_timeframe"],
                "pattern_type": "basis_arbitrage",
            },
            timestamp=market_data.timestamp,
        )

    def _adjust_basis_for_funding(self, basis_pct: float, funding_rate: float) -> float:
        """Adjust basis for expected funding costs"""
        # Estimate funding cost over convergence period
        funding_periods = (
            self.params["convergence_timeframe"] / 8
        )  # Funding every 8 hours
        total_funding_cost = (
            funding_rate * funding_periods * 100
        )  # Convert to percentage

        # Adjust basis by funding cost
        if basis_pct > 0:
            # If perp is at premium, funding cost reduces effective basis
            return basis_pct - total_funding_cost
        else:
            # If perp is at discount, funding revenue improves effective basis
            return basis_pct + total_funding_cost

    def _check_liquidity(self, market_data: MarketFrame) -> bool:
        """Check if there's sufficient liquidity for arbitrage"""
        if not hasattr(market_data, "orderbook") or not market_data.orderbook:
            return True  # Assume OK if no orderbook data

        orderbook = market_data.orderbook

        # Check bid/ask spread
        if not orderbook.bids or not orderbook.asks:
            return False

        best_bid = float(orderbook.bids[0].price)
        best_ask = float(orderbook.asks[0].price)

        if best_bid == 0 or best_ask == 0:
            return False

        spread_pct = ((best_ask - best_bid) / best_bid) * 100

        # Spread should be reasonable for arbitrage
        return spread_pct < 0.1  # 0.1% max spread

    def _check_volume_confirmation(self, market_data: MarketFrame) -> bool:
        """Check for adequate volume in both markets"""
        if not market_data.ohlcv_5m or len(market_data.ohlcv_5m) < 20:
            return True  # Assume OK if not enough data

        current_volume = float(market_data.ohlcv_5m[-1].volume)
        recent_volumes = [float(c.volume) for c in market_data.ohlcv_5m[-20:-1]]

        if not recent_volumes or current_volume == 0:
            return True

        avg_volume = statistics.mean(recent_volumes)
        if avg_volume == 0:
            return True

        volume_ratio = current_volume / avg_volume
        return volume_ratio >= self.params["min_liquidity_ratio"]

    def _calculate_funding_alignment(
        self, basis_pct: float, funding_rate: float
    ) -> float:
        """Calculate how well funding rate aligns with basis direction"""
        # Positive funding (longs pay shorts) should align with perp premium
        # Negative funding (shorts pay longs) should align with perp discount

        if basis_pct > 0 and funding_rate > 0:
            # Perp premium with positive funding → aligned
            return min(1.0, funding_rate * 1000)  # Scale funding rate
        elif basis_pct < 0 and funding_rate < 0:
            # Perp discount with negative funding → aligned
            return min(1.0, abs(funding_rate) * 1000)
        else:
            # Misaligned → lower confidence
            return 0.0

    def _calculate_convergence_probability(
        self, basis_pct: float, funding_rate: float
    ) -> float:
        """Calculate probability of basis convergence"""
        # Higher basis magnitude generally means higher convergence probability
        # But extreme basis might indicate market stress

        basis_magnitude = abs(basis_pct)

        if basis_magnitude < 0.5:
            return 0.3  # Low probability for small basis
        elif basis_magnitude < 2.0:
            return 0.7  # Good probability for moderate basis
        elif basis_magnitude < 5.0:
            return 0.5  # Medium probability for large basis
        else:
            return 0.2  # Low probability for extreme basis

    def _estimate_convergence_time(self, basis_pct: float, funding_rate: float) -> int:
        """Estimate time for basis convergence in hours"""
        basis_magnitude = abs(basis_pct)

        # Generally, larger basis takes longer to converge
        if basis_magnitude < 1.0:
            return 8  # 8 hours for small basis
        elif basis_magnitude < 2.0:
            return 24  # 1 day for moderate basis
        elif basis_magnitude < 3.0:
            return 48  # 2 days for large basis
        else:
            return 72  # 3 days for very large basis

    def _get_historical_basis_stats(self, symbol: str) -> dict[str, float]:
        """Get historical basis statistics for the symbol"""
        # This would analyze historical basis data
        # For now, return default statistics
        return {
            "avg_basis": 0.0,
            "std_basis": 1.0,
            "max_basis": 5.0,
            "min_basis": -5.0,
            "convergence_rate": 0.7,  # 70% of basis converges
        }


def create_basis_arbitrage_strategy(
    config: dict[str, Any] | None = None,
) -> BasisArbitrageStrategy:
    """Factory function to create BasisArbitrageStrategy"""
    if config is None:
        config = {}

    strategy_config = StrategyConfig(
        name="basis_arbitrage", enabled=True, params=config
    )

    return BasisArbitrageStrategy(strategy_config)
