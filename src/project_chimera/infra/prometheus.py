"""
Prometheus metrics exporter for data feed monitoring
Tracks latency, throughput, and reliability metrics
"""

import logging
import time

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from ..datafeed.base import AsyncDataFeed

logger = logging.getLogger(__name__)


class DataFeedMetricsExporter:
    """
    Prometheus metrics exporter for data feed monitoring

    Metrics exported:
    - Feed status and health
    - Message throughput
    - Latency percentiles
    - Error rates
    - Connection status
    """

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()

        # Feed status metrics
        self.feed_status = Gauge(
            "datafeed_status",
            "Data feed operational status (0=stopped, 1=degraded, 2=running)",
            ["exchange", "feed_name"],
            registry=self.registry,
        )

        self.feed_symbols_subscribed = Gauge(
            "datafeed_symbols_subscribed_total",
            "Number of symbols subscribed",
            ["exchange", "feed_name"],
            registry=self.registry,
        )

        # Throughput metrics
        self.messages_received_total = Counter(
            "datafeed_messages_received_total",
            "Total messages received from exchange",
            ["exchange", "feed_name", "message_type"],
            registry=self.registry,
        )

        self.feed_errors_total = Counter(
            "datafeed_errors_total",
            "Total feed errors",
            ["exchange", "feed_name", "error_type"],
            registry=self.registry,
        )

        self.feed_reconnections_total = Counter(
            "datafeed_reconnections_total",
            "Total feed reconnections",
            ["exchange", "feed_name"],
            registry=self.registry,
        )

        # Latency metrics
        self.message_latency_seconds = Histogram(
            "datafeed_message_latency_seconds",
            "Message processing latency in seconds",
            ["exchange", "feed_name"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry,
        )

        self.snapshot_latency_seconds = Histogram(
            "datafeed_snapshot_latency_seconds",
            "Snapshot generation latency in seconds",
            ["exchange", "feed_name", "symbol"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=self.registry,
        )

        # Connection health
        self.connection_status = Gauge(
            "datafeed_connection_status",
            "Connection status (0=disconnected, 1=connected)",
            ["exchange", "feed_name"],
            registry=self.registry,
        )

        self.last_message_timestamp = Gauge(
            "datafeed_last_message_timestamp_seconds",
            "Timestamp of last received message",
            ["exchange", "feed_name"],
            registry=self.registry,
        )

        # Performance metrics
        self.data_freshness_seconds = Gauge(
            "datafeed_data_freshness_seconds",
            "Age of most recent data in seconds",
            ["exchange", "feed_name", "symbol", "data_type"],
            registry=self.registry,
        )

        # Registered feeds
        self.feeds: dict[str, AsyncDataFeed] = {}

    def register_feed(self, name: str, feed: AsyncDataFeed) -> None:
        """Register a data feed for monitoring"""
        self.feeds[name] = feed
        logger.info(f"Registered feed '{name}' for monitoring")

    def unregister_feed(self, name: str) -> None:
        """Unregister a data feed"""
        if name in self.feeds:
            del self.feeds[name]
            logger.info(f"Unregistered feed '{name}' from monitoring")

    def update_metrics(self) -> None:
        """Update all metrics from registered feeds"""
        for feed_name, feed in self.feeds.items():
            try:
                self._update_feed_metrics(feed_name, feed)
            except Exception as e:
                logger.error(f"Error updating metrics for feed '{feed_name}': {e}")

    def _update_feed_metrics(self, feed_name: str, feed: AsyncDataFeed) -> None:
        """Update metrics for a specific feed"""
        exchange = feed.adapter.name
        labels = {"exchange": exchange, "feed_name": feed_name}

        # Get feed metrics
        metrics = feed.get_metrics()

        # Feed status
        status_value = self._status_to_numeric(metrics.get("status", "stopped"))
        self.feed_status.labels(**labels).set(status_value)

        # Symbols subscribed
        symbols_count = metrics.get("symbols_subscribed", 0)
        self.feed_symbols_subscribed.labels(**labels).set(symbols_count)

        # Connection status
        connection_value = 1 if feed.adapter.is_connected() else 0
        self.connection_status.labels(**labels).set(connection_value)

        # Messages received (increment counter)
        messages_received = metrics.get("messages_received", 0)
        current_value = self.messages_received_total.labels(
            **labels, message_type="all"
        )._value._value
        if messages_received > current_value:
            self.messages_received_total.labels(
                **labels, message_type="all"
            )._value.inc(messages_received - current_value)

        # Errors (increment counter)
        errors = metrics.get("errors", 0)
        current_errors = self.feed_errors_total.labels(
            **labels, error_type="general"
        )._value._value
        if errors > current_errors:
            self.feed_errors_total.labels(**labels, error_type="general")._value.inc(
                errors - current_errors
            )

        # Reconnections (increment counter)
        reconnections = metrics.get("reconnections", 0)
        current_reconnections = self.feed_reconnections_total.labels(
            **labels
        )._value._value
        if reconnections > current_reconnections:
            self.feed_reconnections_total.labels(**labels)._value.inc(
                reconnections - current_reconnections
            )

        # Last message timestamp
        last_update = metrics.get("last_update")
        if last_update:
            self.last_message_timestamp.labels(**labels).set(last_update.timestamp())

        # Latency metrics
        latency_median = metrics.get("latency_median_ms")
        if latency_median is not None:
            # Convert to seconds and observe
            self.message_latency_seconds.labels(**labels).observe(
                latency_median / 1000.0
            )

        # Data freshness for each symbol
        current_time = time.time()
        for symbol in feed.symbols:
            # This would require access to feed's internal cache
            # For now, we'll estimate based on last_update
            if last_update:
                freshness = current_time - last_update.timestamp()
                self.data_freshness_seconds.labels(
                    **labels, symbol=symbol, data_type="ticker"
                ).set(freshness)

    def _status_to_numeric(self, status: str) -> int:
        """Convert feed status to numeric value"""
        status_map = {"stopped": 0, "error": 0, "degraded": 1, "running": 2}
        return status_map.get(status.lower(), 0)

    async def measure_snapshot_latency(self, feed_name: str, symbol: str) -> None:
        """Measure and record snapshot generation latency"""
        if feed_name not in self.feeds:
            return

        feed = self.feeds[feed_name]
        exchange = feed.adapter.name

        start_time = time.time()
        try:
            await feed.snapshot(symbol)
            latency = time.time() - start_time

            self.snapshot_latency_seconds.labels(
                exchange=exchange, feed_name=feed_name, symbol=symbol
            ).observe(latency)

        except Exception as e:
            logger.error(f"Error measuring snapshot latency: {e}")

    def start_server(self, port: int = 9100) -> None:
        """Start Prometheus metrics HTTP server"""
        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise

    def get_registry(self) -> CollectorRegistry:
        """Get the Prometheus registry"""
        return self.registry


# Global metrics exporter instance
_metrics_exporter: DataFeedMetricsExporter | None = None


def get_metrics_exporter() -> DataFeedMetricsExporter:
    """Get or create the global metrics exporter"""
    global _metrics_exporter
    if _metrics_exporter is None:
        _metrics_exporter = DataFeedMetricsExporter()
    return _metrics_exporter


def start_metrics_server(port: int = 9100) -> DataFeedMetricsExporter:
    """Start the global metrics server"""
    exporter = get_metrics_exporter()
    exporter.start_server(port)
    return exporter
