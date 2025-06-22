"""
Control Server for Trading System
HTTP API for start/stop/health operations
"""

import asyncio
import logging
import signal
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from ..orchestrator import TradingOrchestrator
from ..settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSystemController:
    """Controller for managing trading system lifecycle"""
    
    def __init__(self):
        self.orchestrator: Optional[TradingOrchestrator] = None
        self.settings = get_settings()
        self.is_running = False
        self.start_time = None
        
    async def start_system(self) -> Dict[str, Any]:
        """Start the trading system"""
        if self.is_running:
            return {"success": False, "message": "System is already running"}
        
        try:
            # Initialize orchestrator with default symbols
            symbols = ['BTCUSDT_SPBL', 'ETHUSDT_SPBL']  # Bitget futures format
            self.orchestrator = TradingOrchestrator(symbols)
            
            # Start orchestrator
            await self.orchestrator.start()
            
            self.is_running = True
            self.start_time = datetime.now()
            
            logger.info("Trading system started successfully")
            return {
                "success": True, 
                "message": "Trading system started successfully",
                "start_time": self.start_time.isoformat(),
                "symbols": symbols
            }
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            self.is_running = False
            return {"success": False, "message": f"Failed to start system: {str(e)}"}
    
    async def stop_system(self) -> Dict[str, Any]:
        """Stop the trading system"""
        if not self.is_running:
            return {"success": False, "message": "System is not running"}
        
        try:
            if self.orchestrator:
                await self.orchestrator.stop()
                self.orchestrator = None
            
            self.is_running = False
            stop_time = datetime.now()
            
            uptime_seconds = (stop_time - self.start_time).total_seconds() if self.start_time else 0
            
            logger.info("Trading system stopped successfully")
            return {
                "success": True, 
                "message": "Trading system stopped successfully",
                "stop_time": stop_time.isoformat(),
                "uptime_seconds": uptime_seconds
            }
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")
            return {"success": False, "message": f"Error stopping system: {str(e)}"}
    
    async def get_health(self) -> Dict[str, Any]:
        """Get system health status"""
        if not self.is_running or not self.orchestrator:
            return {
                "status": "stopped",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": 0,
                "components": {},
                "message": "System is not running"
            }
        
        try:
            health_data = self.orchestrator.get_health()
            return health_data
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "message": "Error retrieving health status"
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        if not self.is_running or not self.orchestrator:
            return {
                "chimera_pnl_total_usd": 0.0,
                "chimera_equity_value_usd": 150000.0,
                "chimera_orders_total": 0,
                "chimera_orders_filled_total": 0,
                "chimera_system_uptime_seconds": 0,
                "chimera_drawdown_percent": 0.0,
                "chimera_websocket_latency_ms": 0.0,
                "chimera_slippage_milliseconds": 0.0
            }
        
        try:
            stats = self.orchestrator.stats
            uptime_seconds = (datetime.now() - stats['start_time']).total_seconds() if stats['start_time'] else 0
            
            # Calculate basic metrics from orchestrator stats
            fill_rate = (stats['signals_executed'] / stats['signals_generated']) if stats['signals_generated'] > 0 else 1.0
            
            return {
                "chimera_pnl_total_usd": 0.0,  # TODO: Get from portfolio manager
                "chimera_equity_value_usd": 150000.0,  # TODO: Get from portfolio manager
                "chimera_orders_total": stats['signals_generated'],
                "chimera_orders_filled_total": stats['signals_executed'],
                "chimera_system_uptime_seconds": uptime_seconds,
                "chimera_drawdown_percent": 0.0,  # TODO: Get from risk engine
                "chimera_websocket_latency_ms": 0.0,  # TODO: Get from data feed
                "chimera_slippage_milliseconds": 0.0,  # TODO: Get from execution engine
                "chimera_fill_rate_percent": fill_rate * 100,
                "chimera_error_count": stats['errors'],
                "chimera_market_frames_processed": stats['market_frames_processed']
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {
                "chimera_pnl_total_usd": 0.0,
                "chimera_equity_value_usd": 150000.0,
                "chimera_orders_total": 0,
                "chimera_orders_filled_total": 0,
                "chimera_system_uptime_seconds": 0,
                "chimera_drawdown_percent": 0.0,
                "chimera_websocket_latency_ms": 0.0,
                "chimera_slippage_milliseconds": 0.0
            }

# Global controller instance
controller = TradingSystemController()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle app lifespan events"""
    # Startup
    logger.info("Control server starting up")
    yield
    # Shutdown
    logger.info("Control server shutting down")
    if controller.is_running:
        await controller.stop_system()

# Create FastAPI app
app = FastAPI(
    title="ProjectChimera Control API",
    description="HTTP API for controlling the trading system",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_endpoint():
    """Health check endpoint"""
    try:
        health_data = await controller.get_health()
        return JSONResponse(content=health_data)
    except Exception as e:
        logger.error(f"Health endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "message": str(e)}
        )

@app.get("/metrics")
async def metrics_endpoint():
    """Metrics endpoint"""
    try:
        metrics_data = await controller.get_metrics()
        return JSONResponse(content=metrics_data)
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "message": str(e)}
        )

@app.post("/start")
async def start_endpoint():
    """Start trading system"""
    try:
        result = await controller.start_system()
        if result["success"]:
            return JSONResponse(content=result)
        else:
            return JSONResponse(status_code=400, content=result)
    except Exception as e:
        logger.error(f"Start endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error", "message": str(e)}
        )

@app.post("/stop")
async def stop_endpoint():
    """Stop trading system"""
    try:
        result = await controller.stop_system()
        if result["success"]:
            return JSONResponse(content=result)
        else:
            return JSONResponse(status_code=400, content=result)
    except Exception as e:
        logger.error(f"Stop endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error", "message": str(e)}
        )

@app.get("/status")
async def status_endpoint():
    """Get system status"""
    try:
        return JSONResponse(content={
            "is_running": controller.is_running,
            "start_time": controller.start_time.isoformat() if controller.start_time else None,
            "uptime_seconds": (datetime.now() - controller.start_time).total_seconds() if controller.start_time else 0
        })
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "message": str(e)}
        )

def run_control_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the control server"""
    uvicorn.run(
        "src.project_chimera.api.control_server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False
    )

if __name__ == "__main__":
    run_control_server()