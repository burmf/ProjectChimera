"""
Advanced Analysis Module for ProjectChimera
Provides technical analysis, market regime detection, and signal generation
"""

from .technical_indicators import (
    TechnicalAnalyzer,
    TechnicalSignal,
    demo_technical_analysis,
    quick_bbands,
    quick_macd,
    quick_rsi,
)

__all__ = [
    "TechnicalAnalyzer",
    "TechnicalSignal",
    "quick_rsi",
    "quick_macd",
    "quick_bbands",
    "demo_technical_analysis",
]
