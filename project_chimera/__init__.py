"""
Project Chimera - Professional High-Frequency Trading System
Professional Clean Architecture Implementation
"""

__version__ = "2.1.0"
__author__ = "ProjectChimera Dev"
__email__ = "dev@projectchimera.local"

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]