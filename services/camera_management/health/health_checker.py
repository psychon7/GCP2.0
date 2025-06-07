"""
Health monitoring module for camera streams.

This module provides functionality for monitoring camera stream health,
implementing exponential backoff for failed streams, and maintaining a
healthy pool of working streams.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
import random
from typing import Dict, Any, List, Optional, Set
import aiohttp
from sqlalchemy.orm import Session

from ..models import Camera, HealthCheck
from ..db import camera_repo, health_repo, geo_repo, get_db_session
from ..config import config

# Configure logger
logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Health monitor for camera streams.
    
    This class monitors camera stream health and maintains a healthy pool.
    """
    
    def __init__(self, session: Session):
        """
        Initialize the health monitor.
        
        Args:
            session: SQLAlchemy session
        """
        self.session = session
        self.check_interval = config.health.check_interval
        self.min_pool_size = config.health.min_pool_size
        self.availability_threshold = config.health.availability_threshold
        self.max_consecutive_failures = config.health.max_consecutive_failures
        self.backoff_base = config.health.backoff_base
        self.backoff_max = config.health.backoff_max
    
    async def check_camera_health(self, camera_id: int) -> Dict[str, Any]:
        """
        Check the health of a camera stream.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Get camera from database
            camera = camera_repo.get_by_id(self.session, camera_id)
            if not camera:
                logger.error(f"Camera not found: {camera_id}")
                return {"success": False, "error": "Camera not found"}
            
            logger.info(f"Checking health for camera: {camera.id} ({camera.url})")
            
            # Check camera health
            start_time = time.time()
            is_available, error_type, error_message = await self._check_stream_availability(camera.url)
            response_time = time.time() - start_time
            
            # Create health check record
            health_data = {
                "camera_id": camera.id,
                "checked_at": datetime.utcnow(),
                "is_available": is_available,
                "response_time_ms": int(response_time * 1000)
            }
            
            if not is_available:
                health_data["error_type"] = error_type
                health_data["error_message"] = error_message
            
            health_check = health_repo.create(self.session, health_data)
            
            # Update camera status based on health check
            self._update_camera_status(camera, is_available)
            
            self.session.commit()
            
            logger.info(f"Health check completed for camera: {camera.id}, available: {is_available}")
            
            return {
                "camera_id": camera.id,
                "is_available": is_available,
                "response_time_ms": int(response_time * 1000),
                "error_type": error_type,
                "error_message": error_message
            }
        except Exception as e:
            logger.error(f"Health check failed for camera: {camera_id}, error: {str(e)}")
            return {"success": False, "error_type": "Exception", "error_message": str(e)}
    
    async def _check_stream_availability(self, url: str) -> tuple:
        """
        Check if a camera stream is available.
        
        Args:
            url: Camera URL
            
        Returns:
            tuple: (is_available, error_type, error_message)
        """
        try:
            # For HTTP/HTTPS URLs, just check if the URL is accessible
            if url.startswith(("http://", "https://")):
                timeout = aiohttp.ClientTimeout(total=config.validation.connection_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    }
                    async with session.head(url, headers=headers) as response:
                        if response.status == 200:
                            return True, None, None
                        else:
                            return False, "HTTPError", f"HTTP status: {response.status}"
            
            # For RTSP URLs, we need to use a different approach
            # For now, just assume it's available (validation will check it properly)
            if url.startswith("rtsp://"):
                return True, None, None
            
            # For other URLs, assume they're not available
            return False, "UnsupportedProtocol", f"Unsupported URL protocol: {url}"
        except aiohttp.ClientError as e:
            return False, "ConnectionError", str(e)
        except asyncio.TimeoutError:
            return False, "Timeout", "Connection timed out"
        except Exception as e:
            return False, type(e).__name__, str(e)
    
    def _update_camera_status(self, camera: Camera, is_available: bool):
        """
        Update camera status based on health check.
        
        Args:
            camera: Camera object
            is_available: Whether the camera is available
        """
        # Get recent health checks
        recent_checks = health_repo.get_by_camera_id(self.session, camera.id, limit=self.max_consecutive_failures)
        
        # Calculate consecutive failures
        consecutive_failures = 0
        for check in recent_checks:
            if not check.is_available:
                consecutive_failures += 1
            else:
                break
        
        # Update camera status
        if is_available:
            # If camera was offline, mark it for validation
            if camera.status == "offline":
                camera_repo.update_camera_status(self.session, camera.id, "discovered")
            # Otherwise, just update the last checked time
            else:
                camera_repo.update(self.session, camera.id, {
                    "last_checked_at": datetime.utcnow()
                })
        else:
            # If too many consecutive failures, mark as offline
            if consecutive_failures >= self.max_consecutive_failures:
                camera_repo.update_camera_status(self.session, camera.id, "offline")
            # Otherwise, just update the last checked time
            else:
                camera_repo.update(self.session, camera.id, {
                    "last_checked_at": datetime.utcnow()
                })
    
    async def check_cameras_needing_health_check(self, limit: int = None) -> Dict[int, Dict[str, Any]]:
        """
        Check health of cameras that need health check.
        
        Args:
            limit: Maximum number of cameras to check
            
        Returns:
            Dict[int, Dict[str, Any]]: Health check results by camera ID
        """
        # Get cameras needing health check
        with self.session.begin():
            cameras = camera_repo.get_cameras_needing_health_check(
                self.session, self.check_interval, limit or self.max_workers * 2
            )
        
        camera_ids = [camera.id for camera in cameras]
        
        if not camera_ids:
            logger.info("No cameras need health check")
            return {}
        
        logger.info(f"Checking health for {len(camera_ids)} cameras")
        
        # Check camera health
        results = await self.check_cameras(camera_ids)
        
        return results
    
    async def check_cameras(self, camera_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Check health of multiple cameras concurrently.
        
        Args:
            camera_ids: List of camera IDs
            
        Returns:
            Dict[int, Dict[str, Any]]: Health check results by camera ID
        """
        tasks = []
        for camera_id in camera_ids:
            tasks.append(self._check_camera_with_semaphore(camera_id))
        
        results = await asyncio.gather(*tasks)
        
        return {camera_id: result for camera_id, result in zip(camera_ids, results)}
    
    async def _check_camera_with_semaphore(self, camera_id: int) -> Dict[str, Any]:
        """
        Check camera health with semaphore for concurrency control.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Dict[str, Any]: Health check result
        """
        async with self.semaphore:
            return await self.check_camera_health(camera_id)
    
    @property
    def max_workers(self) -> int:
        """Get the maximum number of concurrent workers."""
        return config.health.max_workers if hasattr(config.health, "max_workers") else 10
    
    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Get the semaphore for concurrency control."""
        if not hasattr(self, "_semaphore"):
            self._semaphore = asyncio.Semaphore(self.max_workers)
        return self._semaphore


class MaintenanceScheduler:
    """
    Maintenance scheduler for camera pool.
    
    This class schedules maintenance tasks for the camera pool.
    """
    
    def __init__(self, session: Session):
        """
        Initialize the maintenance scheduler.
        
        Args:
            session: SQLAlchemy session
        """
        self.session = session
        self.min_pool_size = config.health.min_pool_size
        self.availability_threshold = config.health.availability_threshold
    
    async def run_maintenance(self):
        """Run maintenance tasks for the camera pool."""
        try:
            logger.info("Running camera pool maintenance")
            
            # Update geographic distribution statistics
            geo_repo.update_distribution_stats(self.session)
            
            # Check if we have enough active cameras
            active_count = self.session.query(Camera).filter(Camera.status == "active").count()
            
            if active_count < self.min_pool_size:
                logger.warning(f"Not enough active cameras: {active_count} < {self.min_pool_size}")
                
                # Schedule discovery jobs to find more cameras
                await self._schedule_discovery_jobs()
            
            # Prune consistently failing cameras
            await self._prune_failing_cameras()
            
            # Balance geographic distribution
            await self._balance_geographic_distribution()
            
            self.session.commit()
            
            logger.info("Camera pool maintenance completed")
        except Exception as e:
            logger.error(f"Maintenance failed: {str(e)}")
            self.session.rollback()
    
    async def _schedule_discovery_jobs(self):
        """Schedule discovery jobs to find more cameras."""
        # This would typically involve scheduling discovery jobs
        # For now, just log that we need more cameras
        logger.info("Scheduling discovery jobs to find more cameras")
    
    async def _prune_failing_cameras(self):
        """Prune consistently failing cameras."""
        # Find cameras with low availability
        one_week_ago = datetime.utcnow() - timedelta(days=7)
        
        # Get cameras with health checks in the last week
        cameras_with_health = self.session.query(Camera).join(
            HealthCheck, Camera.id == HealthCheck.camera_id
        ).filter(
            HealthCheck.checked_at >= one_week_ago
        ).distinct().all()
        
        for camera in cameras_with_health:
            # Get health stats for the camera
            health_stats = health_repo.get_health_stats(self.session, camera.id)
            
            # If availability is below threshold and we have enough checks, blacklist the camera
            if (health_stats["availability_rate"] < self.availability_threshold * 100 and
                    health_stats["total_checks"] >= 10):
                logger.info(f"Blacklisting camera with low availability: {camera.id} ({camera.url})")
                camera_repo.update_camera_status(self.session, camera.id, "blacklisted")
    
    async def _balance_geographic_distribution(self):
        """Balance geographic distribution of cameras."""
        # Get current distribution
        continent_distribution = geo_repo.get_by_region_type(self.session, "continent")
        
        # Check if any continent has too many cameras
        total_cameras = self.session.query(Camera).count()
        if total_cameras == 0:
            return
        
        max_continent_percentage = config.geographic.max_continent_percentage
        
        for dist in continent_distribution:
            if dist.percentage > max_continent_percentage * 100:
                logger.info(f"Continent {dist.region_name} has too many cameras: {dist.percentage}% > {max_continent_percentage * 100}%")
                
                # Get excess cameras from this continent
                excess_count = int((dist.percentage / 100 - max_continent_percentage) * total_cameras)
                
                if excess_count > 0:
                    # Find lowest quality cameras from this continent
                    excess_cameras = self.session.query(Camera).filter(
                        Camera.geo_continent == dist.region_name,
                        Camera.status.in_(["active", "degraded"])
                    ).order_by(Camera.entropy_quality).limit(excess_count).all()
                    
                    # Deactivate excess cameras
                    for camera in excess_cameras:
                        logger.info(f"Deactivating camera for geographic balance: {camera.id} ({camera.url})")
                        camera_repo.update_camera_status(self.session, camera.id, "deferred")


class HealthMonitorService:
    """
    Health monitor service.
    
    This class provides a service for monitoring camera stream health.
    """
    
    def __init__(self):
        """Initialize the health monitor service."""
        self.running = False
        self.check_interval = config.health.check_interval
    
    async def start(self):
        """Start the health monitor service."""
        self.running = True
        
        logger.info("Starting health monitor service")
        
        while self.running:
            try:
                # Create a new session for each iteration
                with get_db_session() as session:
                    # Check camera health
                    monitor = HealthMonitor(session)
                    await monitor.check_cameras_needing_health_check()
                    
                    # Run maintenance tasks
                    scheduler = MaintenanceScheduler(session)
                    await scheduler.run_maintenance()
            except Exception as e:
                logger.error(f"Health monitor service error: {str(e)}")
            
            # Wait for next check interval
            await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop the health monitor service."""
        self.running = False
        logger.info("Stopping health monitor service")
