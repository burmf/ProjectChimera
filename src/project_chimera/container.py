"""
Simple dependency injection container for ProjectChimera
Provides basic configuration and service management
"""

from typing import Any

from .settings import get_settings


class Container:
    """Simple dependency injection container"""

    def __init__(self):
        """Initialize container with configuration"""
        self.config = get_settings()
        self._providers: dict[str, Any] = {}
        self._singletons: dict[str, Any] = {}

    def register(self, name: str, provider: Any) -> None:
        """Register a service provider"""
        self._providers[name] = provider

    def get(self, name: str) -> Any:
        """Get a service instance"""
        if name in self._singletons:
            return self._singletons[name]

        if name in self._providers:
            instance = self._providers[name]()
            self._singletons[name] = instance
            return instance

        raise KeyError(f"Service '{name}' not found in container")

    def wire(self, modules: list = None) -> None:
        """Wire up dependencies (placeholder for dependency-injector compatibility)"""
        # Basic implementation for compatibility with tests
        pass

    def reset(self) -> None:
        """Reset container state"""
        self._singletons.clear()
        self._providers.clear()
