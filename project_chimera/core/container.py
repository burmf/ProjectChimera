"""
Dependency Injection Container for ProjectChimera
Professional IoC container using dependency-injector patterns
"""

import asyncio
from typing import Dict, Any, Optional, Type, TypeVar, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import threading
from contextlib import asynccontextmanager

from loguru import logger
from ..config import Settings, get_settings
from .api_client import AsyncBitgetClient
from .risk_manager import RiskManager


T = TypeVar('T')


class ServiceScope(Enum):
    """Service lifetime scopes"""
    SINGLETON = "singleton"
    TRANSIENT = "transient" 
    SCOPED = "scoped"


@runtime_checkable
class IAsyncDisposable(Protocol):
    """Protocol for async disposable services"""
    async def dispose(self) -> None:
        """Dispose of the service asynchronously"""
        ...


@runtime_checkable  
class IDisposable(Protocol):
    """Protocol for disposable services"""
    def dispose(self) -> None:
        """Dispose of the service"""
        ...


@dataclass
class ServiceDescriptor:
    """Service registration descriptor"""
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[callable] = None
    instance: Optional[Any] = None
    scope: ServiceScope = ServiceScope.TRANSIENT
    dependencies: Optional[Dict[str, Type]] = None


class ServiceRegistrationError(Exception):
    """Service registration error"""
    pass


class ServiceResolutionError(Exception):
    """Service resolution error"""
    pass


class DIContainer:
    """
    Professional Dependency Injection Container
    Thread-safe with async support and proper lifecycle management
    """
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[Type, Any] = {}
        self._lock = threading.RLock()
        self._is_disposed = False
        
        # Register core services
        self._register_core_services()
    
    def _register_core_services(self) -> None:
        """Register core system services"""
        self.register_singleton(Settings, factory=get_settings)
        self.register_transient(AsyncBitgetClient, dependencies={'settings': Settings})
        self.register_transient(RiskManager, dependencies={'settings': Settings})
    
    def register_singleton(
        self, 
        service_type: Type[T], 
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[callable] = None,
        instance: Optional[T] = None,
        dependencies: Optional[Dict[str, Type]] = None
    ) -> 'DIContainer':
        """Register a singleton service"""
        return self._register_service(
            service_type, implementation_type, factory, instance,
            ServiceScope.SINGLETON, dependencies
        )
    
    def register_transient(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[callable] = None,
        dependencies: Optional[Dict[str, Type]] = None
    ) -> 'DIContainer':
        """Register a transient service"""
        return self._register_service(
            service_type, implementation_type, factory, None,
            ServiceScope.TRANSIENT, dependencies
        )
    
    def register_scoped(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[callable] = None,
        dependencies: Optional[Dict[str, Type]] = None
    ) -> 'DIContainer':
        """Register a scoped service"""
        return self._register_service(
            service_type, implementation_type, factory, None,
            ServiceScope.SCOPED, dependencies
        )
    
    def _register_service(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]],
        factory: Optional[callable],
        instance: Optional[T],
        scope: ServiceScope,
        dependencies: Optional[Dict[str, Type]]
    ) -> 'DIContainer':
        """Internal service registration"""
        with self._lock:
            if self._is_disposed:
                raise ServiceRegistrationError("Container is disposed")
            
            # Validation
            if not service_type:
                raise ServiceRegistrationError("Service type is required")
            
            if instance and scope != ServiceScope.SINGLETON:
                raise ServiceRegistrationError("Instance can only be registered as singleton")
            
            if not any([implementation_type, factory, instance]):
                implementation_type = service_type
            
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type,
                factory=factory,
                instance=instance,
                scope=scope,
                dependencies=dependencies or {}
            )
            
            self._services[service_type] = descriptor
            
            logger.debug(f"Registered {service_type.__name__} as {scope.value}")
            return self
    
    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service synchronously"""
        with self._lock:
            if self._is_disposed:
                raise ServiceResolutionError("Container is disposed")
            
            return self._resolve_service(service_type)
    
    async def resolve_async(self, service_type: Type[T]) -> T:
        """Resolve a service asynchronously"""
        # For async resolution, we run the sync resolution in a thread
        # This ensures thread safety while allowing async operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.resolve, service_type)
    
    def _resolve_service(self, service_type: Type[T]) -> T:
        """Internal service resolution"""
        if service_type not in self._services:
            raise ServiceResolutionError(f"Service {service_type.__name__} not registered")
        
        descriptor = self._services[service_type]
        
        # Handle singleton scope
        if descriptor.scope == ServiceScope.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]
            
            instance = self._create_instance(descriptor)
            self._singletons[service_type] = instance
            return instance
        
        # Handle scoped scope
        elif descriptor.scope == ServiceScope.SCOPED:
            if service_type in self._scoped_instances:
                return self._scoped_instances[service_type]
            
            instance = self._create_instance(descriptor)
            self._scoped_instances[service_type] = instance
            return instance
        
        # Handle transient scope
        else:
            return self._create_instance(descriptor)
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create service instance"""
        # Use existing instance
        if descriptor.instance:
            return descriptor.instance
        
        # Use factory
        if descriptor.factory:
            dependencies = self._resolve_dependencies(descriptor.dependencies)
            return descriptor.factory(**dependencies)
        
        # Use implementation type
        if descriptor.implementation_type:
            dependencies = self._resolve_dependencies(descriptor.dependencies)
            return descriptor.implementation_type(**dependencies)
        
        raise ServiceResolutionError(f"Cannot create instance for {descriptor.service_type.__name__}")
    
    def _resolve_dependencies(self, dependencies: Dict[str, Type]) -> Dict[str, Any]:
        """Resolve service dependencies"""
        resolved = {}
        
        for param_name, dependency_type in dependencies.items():
            try:
                resolved[param_name] = self._resolve_service(dependency_type)
            except ServiceResolutionError as e:
                raise ServiceResolutionError(
                    f"Failed to resolve dependency {dependency_type.__name__}: {e}"
                )
        
        return resolved
    
    def clear_scope(self) -> None:
        """Clear scoped instances"""
        with self._lock:
            # Dispose scoped instances
            for instance in self._scoped_instances.values():
                self._dispose_instance(instance)
            
            self._scoped_instances.clear()
            logger.debug("Cleared scoped instances")
    
    def _dispose_instance(self, instance: Any) -> None:
        """Dispose of an instance"""
        try:
            if isinstance(instance, IAsyncDisposable):
                # For async disposable, we need to run in async context
                # This is a limitation - ideally we'd have async dispose
                logger.warning(f"Async disposable {type(instance).__name__} requires async context")
            elif isinstance(instance, IDisposable):
                instance.dispose()
            elif hasattr(instance, 'close'):
                instance.close()
        except Exception as e:
            logger.error(f"Error disposing {type(instance).__name__}: {e}")
    
    async def dispose_async(self) -> None:
        """Dispose container asynchronously"""
        with self._lock:
            if self._is_disposed:
                return
            
            self._is_disposed = True
            
            # Dispose scoped instances
            for instance in self._scoped_instances.values():
                if isinstance(instance, IAsyncDisposable):
                    try:
                        await instance.dispose()
                    except Exception as e:
                        logger.error(f"Error disposing {type(instance).__name__}: {e}")
                else:
                    self._dispose_instance(instance)
            
            # Dispose singletons
            for instance in self._singletons.values():
                if isinstance(instance, IAsyncDisposable):
                    try:
                        await instance.dispose()
                    except Exception as e:
                        logger.error(f"Error disposing {type(instance).__name__}: {e}")
                else:
                    self._dispose_instance(instance)
            
            self._scoped_instances.clear()
            self._singletons.clear()
            
            logger.info("DIContainer disposed")
    
    def dispose(self) -> None:
        """Dispose container synchronously"""
        asyncio.run(self.dispose_async())
    
    @asynccontextmanager
    async def scope(self):
        """Create a scoped context"""
        try:
            yield self
        finally:
            self.clear_scope()
    
    def __enter__(self):
        """Sync context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit"""
        self.dispose()
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.dispose_async()


# Global container instance
_container: Optional[DIContainer] = None
_container_lock = threading.Lock()


def get_container() -> DIContainer:
    """Get the global container instance (thread-safe singleton)"""
    global _container
    
    if _container is None:
        with _container_lock:
            if _container is None:
                _container = DIContainer()
    
    return _container


def configure_services(container: DIContainer) -> None:
    """Configure application services"""
    # Core services are already registered in DIContainer.__init__
    
    # Register additional services here
    logger.info("Services configured successfully")


# Service locator pattern for easy access
class ServiceLocator:
    """Service locator for easy dependency access"""
    
    @staticmethod
    def get_settings() -> Settings:
        return get_container().resolve(Settings)
    
    @staticmethod
    def get_api_client() -> AsyncBitgetClient:
        return get_container().resolve(AsyncBitgetClient)
    
    @staticmethod
    def get_risk_manager() -> RiskManager:
        return get_container().resolve(RiskManager)
    
    @staticmethod
    async def get_api_client_async() -> AsyncBitgetClient:
        return await get_container().resolve_async(AsyncBitgetClient)


# Example usage and testing
async def test_container():
    """Test the DI container"""
    container = DIContainer()
    
    # Test service resolution
    settings = container.resolve(Settings)
    logger.info(f"Settings environment: {settings.environment}")
    
    # Test async resolution
    async with container:
        api_client = await container.resolve_async(AsyncBitgetClient)
        logger.info(f"API client initialized: {type(api_client).__name__}")
        
        # Test scoped context
        async with container.scope():
            risk_manager = container.resolve(RiskManager)
            logger.info(f"Risk manager initialized: {type(risk_manager).__name__}")


if __name__ == "__main__":
    asyncio.run(test_container())