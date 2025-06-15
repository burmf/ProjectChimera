"""
Dependency injection container using dependency-injector
Centralizes all service dependencies and configuration
"""

from dependency_injector import containers, providers
import httpx
from .settings import Settings, get_settings


class Container(containers.DeclarativeContainer):
    """Main dependency injection container"""
    
    # Configuration
    config = providers.Singleton(get_settings)
    
    # HTTP Client
    http_client = providers.Singleton(
        httpx.AsyncClient,
        timeout=providers.Callable(lambda cfg: cfg.api.timeout_seconds, config)
    )
    
    # API Clients (to be implemented)
    # bitget = providers.Factory(BitgetClient, cfg=config, client=http_client)
    
    # Core Services (to be implemented)  
    # risk_engine = providers.Factory(RiskEngine, cfg=config)
    # portfolio = providers.Factory(Portfolio, cfg=config)
    
    # Infrastructure (to be implemented)
    # redis_repo = providers.Factory(RedisRepository, cfg=config)
    # postgres_repo = providers.Factory(PostgresRepository, cfg=config)
    # prometheus_exporter = providers.Factory(PrometheusExporter, cfg=config)


# Global container instance
container = Container()