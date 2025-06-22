"""
Advanced Analysis Module for ProjectChimera
Provides technical analysis, market regime detection, and signal generation
"""

from .technical_indicators import (
    TechnicalAnalyzer,
    TechnicalSignal,
    quick_rsi,
    quick_macd,
    quick_bbands,
    demo_technical_analysis
)

__all__ = [
    'TechnicalAnalyzer',
    'TechnicalSignal', 
    'quick_rsi',
    'quick_macd',
    'quick_bbands',
    'demo_technical_analysis'
]