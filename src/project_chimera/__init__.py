"""
Project Chimera - Professional High-Frequency Trading System
Production-grade async trading platform with advanced risk management
"""

__version__ = "2.0.0"
__author__ = "ProjectChimera Dev"
__email__ = "dev@projectchimera.local"

from .settings import Settings
from .container import Container

__all__ = ["Settings", "Container", "__version__"]