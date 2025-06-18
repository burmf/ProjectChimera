"""
Simple health check server for Phase F validation
Demonstrates /health endpoint functionality
"""

import asyncio
import json
import logging
import signal
import time
from datetime import datetime
from typing import Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus:
    """System health status tracker"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.components = {
            'orchestrator': {'status': 'running', 'last_check': datetime.now().isoformat()},
            'data_feed': {'status': 'running', 'last_check': datetime.now().isoformat()},
            'strategy_hub': {'status': 'running', 'last_check': datetime.now().isoformat()},
            'risk_engine': {'status': 'running', 'last_check': datetime.now().isoformat()},
            'execution_engine': {'status': 'running', 'last_check': datetime.now().isoformat()},
            'circuit_breaker': {'status': 'closed', 'failures': 0}
        }
        self.metrics = {
            'uptime_seconds': 0,
            'requests_processed': 0,
            'orders_placed': 0,
            'orders_filled': 0,
            'errors': 0
        }
    
    def update_metrics(self):
        """Update system metrics"""
        self.metrics['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()
        self.metrics['requests_processed'] += 1
        
        # Simulate some activity
        import random
        if random.random() < 0.1:  # 10% chance
            self.metrics['orders_placed'] += 1
        if random.random() < 0.05:  # 5% chance
            self.metrics['orders_filled'] += 1
    
    def get_health(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        self.update_metrics()
        
        all_healthy = all(comp['status'] in ['running', 'closed'] for comp in self.components.values())
        
        return {
            'status': 'ok' if all_healthy else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': self.metrics['uptime_seconds'],
            'components': self.components,
            'metrics': self.metrics,
            'version': '1.0.0-phase-f',
            'environment': 'sandbox'
        }


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP handler for health endpoint"""
    
    def __init__(self, health_status: HealthStatus, *args, **kwargs):
        self.health_status = health_status
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/health':
            health = self.health_status.get_health()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = json.dumps(health, indent=2)
            self.wfile.write(response.encode('utf-8'))
            
            logger.info(f"Health check: {health['status']}")
        
        elif self.path == '/':
            # Simple status page
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Project Chimera - Phase F</title>
                <style>
                    body { font-family: Arial; margin: 40px; }
                    .status { padding: 20px; border-radius: 5px; margin: 10px 0; }
                    .ok { background-color: #d4edda; border: 1px solid #c3e6cb; }
                    .degraded { background-color: #fff3cd; border: 1px solid #ffeaa7; }
                    .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
                </style>
                <script>
                    function refreshHealth() {
                        fetch('/health')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('status').innerText = data.status;
                                document.getElementById('uptime').innerText = Math.floor(data.uptime_seconds) + 's';
                                document.getElementById('requests').innerText = data.metrics.requests_processed;
                                document.getElementById('orders').innerText = data.metrics.orders_placed;
                                
                                const statusDiv = document.getElementById('status-div');
                                statusDiv.className = 'status ' + data.status;
                            });
                    }
                    
                    setInterval(refreshHealth, 2000);  // Refresh every 2 seconds
                    window.onload = refreshHealth;
                </script>
            </head>
            <body>
                <h1>🚀 Project Chimera - Phase F Status</h1>
                <div id="status-div" class="status ok">
                    <h2>System Status: <span id="status">Loading...</span></h2>
                    <p><strong>Uptime:</strong> <span id="uptime">0s</span></p>
                    <p><strong>Requests:</strong> <span id="requests">0</span></p>
                    <p><strong>Orders:</strong> <span id="orders">0</span></p>
                </div>
                
                <h3>Pipeline Components:</h3>
                <ul>
                    <li>✅ Data Feed</li>
                    <li>✅ Strategy Hub</li>
                    <li>✅ Risk Engine</li>
                    <li>✅ Execution Engine</li>
                    <li>✅ Circuit Breaker</li>
                </ul>
                
                <h3>Endpoints:</h3>
                <ul>
                    <li><a href="/health">/health</a> - JSON health status</li>
                    <li><a href="/">/</a> - This status page</li>
                </ul>
                
                <p><em>Phase F Implementation: async pipeline Feed→StrategyHub→RiskEngine→Execution</em></p>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode('utf-8'))
        
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        """Override to reduce log noise"""
        pass


def create_handler(health_status):
    """Create handler with health status"""
    def handler(*args, **kwargs):
        return HealthHandler(health_status, *args, **kwargs)
    return handler


class HealthServer:
    """Simple health check server"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.health_status = HealthStatus()
        self.server = None
        self.server_thread = None
        self.running = False
    
    def start(self):
        """Start the health server"""
        logger.info(f"Starting health server on port {self.port}")
        
        handler = create_handler(self.health_status)
        self.server = HTTPServer(('0.0.0.0', self.port), handler)
        
        # Run server in separate thread
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.running = True
        logger.info(f"Health server running on http://localhost:{self.port}")
        logger.info(f"Health endpoint: http://localhost:{self.port}/health")
    
    def stop(self):
        """Stop the health server"""
        if self.server:
            logger.info("Stopping health server")
            self.server.shutdown()
            self.server.server_close()
            self.running = False
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.running
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        return {
            'running': self.running,
            'port': self.port,
            'health_url': f'http://localhost:{self.port}/health'
        }


async def simulate_trading_activity(health_server: HealthServer):
    """Simulate trading system activity"""
    logger.info("Starting trading activity simulation")
    
    while health_server.is_running():
        try:
            # Simulate various system events
            import random
            
            # Simulate order placement
            if random.random() < 0.2:  # 20% chance
                health_server.health_status.metrics['orders_placed'] += 1
                logger.info("Simulated order placement")
            
            # Simulate order fill
            if random.random() < 0.1:  # 10% chance
                health_server.health_status.metrics['orders_filled'] += 1
                logger.info("Simulated order fill")
            
            # Simulate occasional error
            if random.random() < 0.02:  # 2% chance
                health_server.health_status.metrics['errors'] += 1
                logger.warning("Simulated system error")
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in simulation: {e}")
            await asyncio.sleep(5)


def setup_signal_handlers(health_server: HealthServer):
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        health_server.stop()
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health Check Server")
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--simulate', action='store_true', help='Simulate trading activity')
    
    args = parser.parse_args()
    
    # Create and start health server
    health_server = HealthServer(args.port)
    setup_signal_handlers(health_server)
    
    try:
        health_server.start()
        
        print(f"""
🚀 Project Chimera Phase F Health Server Started!

📊 Status Page: http://localhost:{args.port}/
🔍 Health Check: http://localhost:{args.port}/health

Press Ctrl+C to stop the server
        """)
        
        # Start simulation if requested
        simulation_task = None
        if args.simulate:
            simulation_task = asyncio.create_task(simulate_trading_activity(health_server))
        
        # Keep running until stopped
        while health_server.is_running():
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        if simulation_task:
            simulation_task.cancel()
        health_server.stop()
    
    return 0


if __name__ == '__main__':
    exit(asyncio.run(main()))