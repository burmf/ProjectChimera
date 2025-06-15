"""
Position Management System
Professional position tracking and management
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from loguru import logger

from ..config import Settings, get_settings


class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"
    CANCELLED = "cancelled"


class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """Individual position record"""
    position_id: str
    symbol: str
    side: PositionSide
    entry_price: float
    size: float  # In USD
    leverage: int = 1
    entry_time: datetime = field(default_factory=datetime.now)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    strategy: str = "default"
    notes: str = ""


@dataclass
class PortfolioSummary:
    """Portfolio summary metrics"""
    total_positions: int = 0
    open_positions: int = 0
    total_exposure: float = 0.0  # Total leveraged exposure
    net_exposure: float = 0.0    # Net long/short exposure
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin_used: float = 0.0
    free_margin: float = 0.0
    margin_ratio: float = 0.0
    largest_position: float = 0.0
    position_concentration: float = 0.0


class PositionManager:
    """
    Professional position management system
    Tracks positions, calculates P&L, manages risk
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        
        # Position storage
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Portfolio tracking
        self.account_balance = 100000.0  # Starting balance
        self.available_balance = 100000.0
        self.total_equity = 100000.0
        
        # Risk limits
        self.max_positions = self.settings.trading.max_positions
        self.max_leverage = self.settings.trading.max_leverage
        self.max_exposure_pct = 0.95  # 95% max portfolio exposure
        
        # Performance tracking
        self.position_history: List[Dict] = []
        
        logger.info("PositionManager initialized")
    
    async def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        leverage: int = 1,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        strategy: str = "default",
        notes: str = ""
    ) -> Optional[str]:
        """
        Open a new position
        Returns position ID if successful
        """
        try:
            # Validate position
            if not self._validate_new_position(size, leverage):
                return None
            
            # Generate position ID
            position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create position
            position = Position(
                position_id=position_id,
                symbol=symbol,
                side=PositionSide.LONG if side.lower() == 'long' else PositionSide.SHORT,
                entry_price=entry_price,
                size=size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit,
                current_price=entry_price,
                strategy=strategy,
                notes=notes
            )
            
            # Calculate commission
            position.commission = self._calculate_commission(size, leverage)
            
            # Store position
            self.positions[position_id] = position
            
            # Update account balance
            margin_required = size / leverage
            self.available_balance -= margin_required
            
            logger.info(f"Opened position: {position_id} {side} {size} {symbol} @ {entry_price}")
            
            return position_id
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return None
    
    async def close_position(
        self,
        position_id: str,
        exit_price: Optional[float] = None,
        partial_size: Optional[float] = None
    ) -> bool:
        """
        Close position (fully or partially)
        """
        try:
            if position_id not in self.positions:
                logger.error(f"Position {position_id} not found")
                return False
            
            position = self.positions[position_id]
            
            if position.status != PositionStatus.OPEN:
                logger.error(f"Position {position_id} is not open")
                return False
            
            # Use current price if no exit price provided
            if exit_price is None:
                exit_price = position.current_price
            
            # Calculate P&L
            pnl = self._calculate_pnl(position, exit_price)
            
            # Handle partial close
            if partial_size and partial_size < position.size:
                # Create new position for remaining size
                remaining_size = position.size - partial_size
                
                # Calculate partial P&L
                partial_pnl = pnl * (partial_size / position.size)
                
                # Update original position
                position.size = remaining_size
                position.realized_pnl += partial_pnl
                
                # Update account balance
                margin_freed = partial_size / position.leverage
                self.available_balance += margin_freed + partial_pnl
                
                logger.info(f"Partially closed {position_id}: {partial_size} @ {exit_price} P&L: ${partial_pnl:.2f}")
                
            else:
                # Full close
                position.status = PositionStatus.CLOSED
                position.realized_pnl = pnl
                
                # Update account balance
                margin_freed = position.size / position.leverage
                self.available_balance += margin_freed + pnl
                
                # Move to closed positions
                self.closed_positions.append(position)
                del self.positions[position_id]
                
                logger.info(f"Closed position {position_id} @ {exit_price} P&L: ${pnl:.2f}")
            
            # Update portfolio metrics
            await self._update_portfolio_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return False
    
    async def update_position(self, symbol: str, size: float, side: str) -> None:
        """
        Update position from external trade execution
        """
        try:
            # Find existing position for symbol
            existing_position = None
            for pos in self.positions.values():
                if pos.symbol == symbol and pos.side.value == side:
                    existing_position = pos
                    break
            
            if existing_position:
                # Update existing position
                existing_position.size += size
                logger.debug(f"Updated position {existing_position.position_id}: new size {existing_position.size}")
            else:
                # This would be handled by open_position in real trading
                logger.debug(f"No existing position found for {symbol} {side}")
                
        except Exception as e:
            logger.error(f"Error updating position: {e}")
    
    async def update_prices(self, price_data: Dict[str, float]) -> None:
        """
        Update current prices for all positions
        """
        try:
            for position in self.positions.values():
                if position.symbol in price_data:
                    position.current_price = price_data[position.symbol]
                    position.unrealized_pnl = self._calculate_pnl(position, position.current_price)
                    
                    # Check stop loss and take profit
                    await self._check_exit_conditions(position)
            
            # Update portfolio metrics
            await self._update_portfolio_metrics()
            
        except Exception as e:
            logger.error(f"Error updating prices: {e}")
    
    async def _check_exit_conditions(self, position: Position) -> None:
        """
        Check if position should be closed due to stop loss or take profit
        """
        if not position.current_price:
            return
        
        current_price = position.current_price
        should_close = False
        exit_reason = ""
        
        if position.side == PositionSide.LONG:
            # Long position
            if position.stop_loss and current_price <= position.stop_loss:
                should_close = True
                exit_reason = "Stop Loss"
            elif position.take_profit and current_price >= position.take_profit:
                should_close = True
                exit_reason = "Take Profit"
        else:
            # Short position
            if position.stop_loss and current_price >= position.stop_loss:
                should_close = True
                exit_reason = "Stop Loss"
            elif position.take_profit and current_price <= position.take_profit:
                should_close = True
                exit_reason = "Take Profit"
        
        if should_close:
            logger.info(f"Auto-closing position {position.position_id} due to {exit_reason}")
            await self.close_position(position.position_id, current_price)
    
    def _calculate_pnl(self, position: Position, current_price: float) -> float:
        """
        Calculate position P&L
        """
        if position.side == PositionSide.LONG:
            price_diff = current_price - position.entry_price
        else:
            price_diff = position.entry_price - current_price
        
        # Calculate P&L considering leverage
        pnl_pct = price_diff / position.entry_price
        pnl = position.size * pnl_pct * position.leverage
        
        # Subtract commission
        return pnl - position.commission
    
    def _calculate_commission(self, size: float, leverage: int) -> float:
        """
        Calculate trading commission
        """
        # Typical futures commission: 0.04% for maker, 0.06% for taker
        commission_rate = 0.0006  # 0.06%
        return size * commission_rate
    
    def _validate_new_position(self, size: float, leverage: int) -> bool:
        """
        Validate if new position can be opened
        """
        # Check position count limit
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Cannot open position: max positions ({self.max_positions}) reached")
            return False
        
        # Check leverage limit
        if leverage > self.max_leverage:
            logger.warning(f"Cannot open position: leverage {leverage} exceeds max {self.max_leverage}")
            return False
        
        # Check margin requirements
        margin_required = size / leverage
        if margin_required > self.available_balance:
            logger.warning(f"Cannot open position: insufficient margin (${margin_required:.2f} required, ${self.available_balance:.2f} available)")
            return False
        
        # Check total exposure
        current_exposure = sum(pos.size * pos.leverage for pos in self.positions.values())
        new_total_exposure = current_exposure + (size * leverage)
        max_exposure = self.total_equity * self.max_exposure_pct
        
        if new_total_exposure > max_exposure:
            logger.warning(f"Cannot open position: would exceed max exposure (${new_total_exposure:.2f} > ${max_exposure:.2f})")
            return False
        
        return True
    
    async def _update_portfolio_metrics(self) -> None:
        """
        Update portfolio-level metrics
        """
        try:
            # Calculate total unrealized P&L
            total_unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized = sum(pos.realized_pnl for pos in self.closed_positions)
            
            # Update total equity
            self.total_equity = self.account_balance + total_unrealized + total_realized
            
            # Calculate margin usage
            margin_used = sum(pos.size / pos.leverage for pos in self.positions.values())
            margin_ratio = margin_used / self.total_equity if self.total_equity > 0 else 0
            
            # Log if margin ratio is high
            if margin_ratio > 0.8:
                logger.warning(f"High margin usage: {margin_ratio:.2%}")
            
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def get_portfolio_summary(self) -> PortfolioSummary:
        """
        Get current portfolio summary
        """
        open_positions = list(self.positions.values())
        
        # Calculate exposures
        total_exposure = sum(pos.size * pos.leverage for pos in open_positions)
        long_exposure = sum(pos.size * pos.leverage for pos in open_positions if pos.side == PositionSide.LONG)
        short_exposure = sum(pos.size * pos.leverage for pos in open_positions if pos.side == PositionSide.SHORT)
        net_exposure = long_exposure - short_exposure
        
        # Calculate P&L
        unrealized_pnl = sum(pos.unrealized_pnl for pos in open_positions)
        realized_pnl = sum(pos.realized_pnl for pos in self.closed_positions)
        
        # Calculate margins
        margin_used = sum(pos.size / pos.leverage for pos in open_positions)
        free_margin = self.available_balance
        margin_ratio = margin_used / self.total_equity if self.total_equity > 0 else 0
        
        # Find largest position
        largest_position = max((pos.size for pos in open_positions), default=0)
        
        # Calculate concentration
        if total_exposure > 0:
            position_concentration = largest_position / total_exposure
        else:
            position_concentration = 0
        
        return PortfolioSummary(
            total_positions=len(self.positions) + len(self.closed_positions),
            open_positions=len(self.positions),
            total_exposure=total_exposure,
            net_exposure=net_exposure,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            margin_used=margin_used,
            free_margin=free_margin,
            margin_ratio=margin_ratio,
            largest_position=largest_position,
            position_concentration=position_concentration
        )
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """
        Get specific position by ID
        """
        return self.positions.get(position_id)
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """
        Get all positions for a specific symbol
        """
        return [pos for pos in self.positions.values() if pos.symbol == symbol]
    
    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions
        """
        return list(self.positions.values())
    
    def get_position_risk(self, position_id: str) -> Optional[Dict[str, float]]:
        """
        Calculate risk metrics for specific position
        """
        position = self.positions.get(position_id)
        if not position or not position.current_price:
            return None
        
        # Calculate risk metrics
        unrealized_pnl_pct = position.unrealized_pnl / position.size if position.size > 0 else 0
        
        # Calculate distance to stop loss
        if position.stop_loss:
            if position.side == PositionSide.LONG:
                stop_distance_pct = (position.current_price - position.stop_loss) / position.current_price
            else:
                stop_distance_pct = (position.stop_loss - position.current_price) / position.current_price
        else:
            stop_distance_pct = 0
        
        # Calculate max loss if stopped out
        max_loss = position.size * stop_distance_pct * position.leverage if position.stop_loss else 0
        
        return {
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'stop_distance_pct': stop_distance_pct,
            'max_potential_loss': max_loss,
            'margin_ratio': (position.size / position.leverage) / self.total_equity,
            'leverage_ratio': position.leverage
        }
    
    def generate_position_report(self) -> str:
        """
        Generate comprehensive position report
        """
        summary = self.get_portfolio_summary()
        
        report = f"""
📊 PORTFOLIO POSITION REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💼 PORTFOLIO SUMMARY:
  Total Equity: ${self.total_equity:,.2f}
  Available Balance: ${self.available_balance:,.2f}
  Unrealized P&L: ${summary.unrealized_pnl:,.2f}
  Realized P&L: ${summary.realized_pnl:,.2f}
  
📍 POSITION OVERVIEW:
  Open Positions: {summary.open_positions}
  Total Positions: {summary.total_positions}
  Total Exposure: ${summary.total_exposure:,.2f}
  Net Exposure: ${summary.net_exposure:,.2f}
  
⚡ RISK METRICS:
  Margin Used: ${summary.margin_used:,.2f}
  Margin Ratio: {summary.margin_ratio:.2%}
  Largest Position: ${summary.largest_position:,.2f}
  Position Concentration: {summary.position_concentration:.2%}

🎯 OPEN POSITIONS:
"""
        
        for position in self.positions.values():
            pnl_pct = (position.unrealized_pnl / position.size * 100) if position.size > 0 else 0
            
            report += f"""
  {position.symbol} ({position.side.value.upper()}):
    Size: ${position.size:,.2f} | Leverage: {position.leverage}x
    Entry: ${position.entry_price:.2f} | Current: ${position.current_price or 0:.2f}
    P&L: ${position.unrealized_pnl:.2f} ({pnl_pct:+.2f}%)
    Stop Loss: ${position.stop_loss or 0:.2f} | Take Profit: ${position.take_profit or 0:.2f}
"""
        
        report += f"\n{'='*60}"
        
        return report
    
    async def emergency_close_all(self) -> int:
        """
        Emergency close all positions
        Returns number of positions closed
        """
        logger.warning("🚨 EMERGENCY: Closing all positions")
        
        positions_closed = 0
        for position_id in list(self.positions.keys()):
            if await self.close_position(position_id):
                positions_closed += 1
        
        logger.info(f"Emergency close completed: {positions_closed} positions closed")
        return positions_closed
    
    def reset(self) -> None:
        """
        Reset position manager (for testing)
        """
        self.positions.clear()
        self.closed_positions.clear()
        self.account_balance = 100000.0
        self.available_balance = 100000.0
        self.total_equity = 100000.0
        
        logger.info("PositionManager reset")


if __name__ == "__main__":
    # Test position manager
    async def test_position_manager():
        pm = PositionManager()
        
        # Test opening positions
        pos1 = await pm.open_position(
            symbol="BTCUSDT",
            side="long",
            size=10000,
            entry_price=50000,
            leverage=10,
            stop_loss=49000,
            take_profit=52000,
            strategy="test"
        )
        
        pos2 = await pm.open_position(
            symbol="ETHUSDT",
            side="short",
            size=5000,
            entry_price=3000,
            leverage=5,
            stop_loss=3100,
            take_profit=2900,
            strategy="test"
        )
        
        print(f"Opened positions: {pos1}, {pos2}")
        
        # Test price updates
        await pm.update_prices({
            "BTCUSDT": 51000,  # Profit for long
            "ETHUSDT": 2950    # Profit for short
        })
        
        # Generate report
        print(pm.generate_position_report())
        
        # Test closing position
        if pos1:
            await pm.close_position(pos1, 51500)
        
        print("\nAfter closing BTC position:")
        print(pm.generate_position_report())
    
    asyncio.run(test_position_manager())