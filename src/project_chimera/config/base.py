"""
Base configuration classes and mixins
Common patterns for all ProjectChimera configurations
"""

from abc import ABC
from typing import Any

from pydantic import BaseModel, Field


class BaseConfig(BaseModel, ABC):
    """
    Base configuration class with common validation patterns

    Design Reference: CLAUDE.md - Configuration consolidation
    Related Classes:
    - All *Config classes inherit from this base
    - Provides common validation and serialization patterns
    - Reduces duplication across configuration classes
    """

    # Common fields for all configs
    enabled: bool = Field(default=True, description="Enable/disable this component")

    # Common validation methods
    model_config = {"extra": "forbid", "validate_assignment": True}

    def validate_positive(self, field_name: str, value: float) -> None:
        """Validate that a field is positive"""
        if value <= 0:
            raise ValueError(f"{field_name} must be positive, got {value}")

    def validate_range(
        self, field_name: str, value: float, min_val: float, max_val: float
    ) -> None:
        """Validate that a field is within range"""
        if not min_val <= value <= max_val:
            raise ValueError(
                f"{field_name} must be between {min_val} and {max_val}, got {value}"
            )

    def validate_percentage(self, field_name: str, value: float) -> None:
        """Validate that a field is a valid percentage (0-1)"""
        self.validate_range(field_name, value, 0.0, 1.0)


class ConfigMixin:
    """
    Mixin providing common configuration patterns

    Design Reference: CLAUDE.md - DRY principle for config classes
    Related Classes:
    - Used by concrete config classes to share common methods
    - Reduces code duplication in validation logic
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""
        if hasattr(self, "model_dump"):
            return self.model_dump()
        elif hasattr(self, "dict"):
            return self.dict()
        else:
            return vars(self)

    def update(self, **kwargs) -> None:
        """Update configuration fields"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration field: {key}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfigMixin":
        """Create configuration from dictionary"""
        return cls(**data)
