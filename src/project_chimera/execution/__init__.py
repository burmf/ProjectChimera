"""
Execution module for Project Chimera
Contains exchange-specific execution engines
"""

from .bitget import BitgetExecutionEngine, BitgetConfig, Order, Fill, OrderStatus, OrderSide, OrderType

__all__ = [
    'BitgetExecutionEngine',
    'BitgetConfig', 
    'Order',
    'Fill',
    'OrderStatus',
    'OrderSide',
    'OrderType'
]