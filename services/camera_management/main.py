"""
Main entry point for the Camera Management service.

This module provides the main entry point for starting the Camera Management service,
which includes the API server, health monitor, and other background services.
"""
import asyncio
import logging
import os
import sys
import signal
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .api.routes import router as api_router
from .db import init_db
from .health.health_checker import HealthMonitorService
from .config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Get logger
logger = logging.getLogger(__name__)

# Global services
health_monitor_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    
    This function is called when the FastAPI application starts up and shuts down.
    It's used to initialize and clean up resources.
    
    Args:
        app: FastAPI application
    """
    # Startup
    logger.info("Initializing Camera Management service")
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        sys.exit(1)
    
    # Start health monitor service
    global health_monitor_service
    health_monitor_service = HealthMonitorService()
    asyncio.create_task(health_monitor_service.start())
    logger.info("Health monitor service started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Camera Management service")
    
    # Stop health monitor service
    if health_monitor_service:
        health_monitor_service.stop()
        logger.info("Health monitor service stopped")


# Create FastAPI app
app = FastAPI(
    title="Camera Management Service",
    description="API for camera stream discovery, validation, and health monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


# Signal handlers
def handle_sigterm(signum, frame):
    """Handle SIGTERM signal."""
    logger.info("Received SIGTERM signal, shutting down")
    sys.exit(0)


def handle_sigint(signum, frame):
    """Handle SIGINT signal."""
    logger.info("Received SIGINT signal, shutting down")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, handle_sigterm)
signal.signal(signal.SIGINT, handle_sigint)


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "services.camera_management.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        workers=config.api.workers
    )


if __name__ == "__main__":
    run_server()
