"""
Discovery module for the Camera Management service.

This module provides functionality for discovering public camera streams
from various sources.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
import asyncio
import logging
import time
from datetime import datetime
import aiohttp
from sqlalchemy.orm import Session

from ..models import Camera, DiscoveryJob
from ..db import camera_repo, discovery_repo
from ..config import config

# Configure logger
logger = logging.getLogger(__name__)


class DiscoveryWorker(ABC):
    """
    Abstract base class for camera discovery workers.
    
    This class defines the interface that all discovery workers must implement.
    """
    
    def __init__(self, session: Session, source_name: str):
        """
        Initialize the discovery worker.
        
        Args:
            session: SQLAlchemy session
            source_name: Name of the camera source
        """
        self.session = session
        self.source_name = source_name
        self.job = None
        self.stats = {
            "total_discovered": 0,
            "new_cameras": 0,
            "updated_cameras": 0,
            "failed_urls": 0
        }
        self.rate_limiter = RateLimiter(
            config.discovery.rate_limit_requests,
            config.discovery.rate_limit_window
        )
    
    async def discover(self) -> Dict[str, Any]:
        """
        Run the discovery process.
        
        Returns:
            Dict[str, Any]: Discovery statistics
        """
        try:
            # Create discovery job
            self.job = discovery_repo.create(self.session, {
                "source": self.source_name,
                "status": "running"
            })
            self.session.commit()
            
            logger.info(f"Starting discovery for source: {self.source_name}")
            
            # Run the actual discovery
            await self._discover()
            
            # Complete the job
            discovery_repo.complete_job(self.session, self.job.id, self.stats)
            self.session.commit()
            
            logger.info(f"Discovery completed for source: {self.source_name}, stats: {self.stats}")
            
            return self.stats
        except Exception as e:
            logger.error(f"Discovery failed for source: {self.source_name}, error: {str(e)}")
            if self.job:
                discovery_repo.fail_job(self.session, self.job.id, str(e))
                self.session.commit()
            raise
    
    @abstractmethod
    async def _discover(self):
        """
        Implement the actual discovery logic.
        
        This method must be implemented by subclasses.
        """
        pass
    
    async def process_camera_url(self, url: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Process a discovered camera URL.
        
        Args:
            url: Camera URL
            metadata: Additional metadata for the camera
            
        Returns:
            bool: True if the camera was processed successfully, False otherwise
        """
        try:
            self.stats["total_discovered"] += 1
            
            # Check if camera already exists
            existing_camera = camera_repo.get_by_url(self.session, url)
            
            if existing_camera:
                # Update existing camera
                camera_repo.update(self.session, existing_camera.id, {
                    "last_checked_at": datetime.utcnow(),
                    "metadata": {**(existing_camera.metadata or {}), **(metadata or {})}
                })
                self.stats["updated_cameras"] += 1
                return True
            else:
                # Create new camera
                camera_data = {
                    "url": url,
                    "source": self.source_name,
                    "status": "discovered",
                    "metadata": metadata or {}
                }
                
                # Extract geo information if available in metadata
                if metadata:
                    if "geo_lat" in metadata:
                        camera_data["geo_lat"] = metadata["geo_lat"]
                    if "geo_lon" in metadata:
                        camera_data["geo_lon"] = metadata["geo_lon"]
                    if "geo_city" in metadata:
                        camera_data["geo_city"] = metadata["geo_city"]
                    if "geo_country" in metadata:
                        camera_data["geo_country"] = metadata["geo_country"]
                    if "geo_continent" in metadata:
                        camera_data["geo_continent"] = metadata["geo_continent"]
                
                camera_repo.create(self.session, camera_data)
                self.stats["new_cameras"] += 1
                return True
        except Exception as e:
            logger.error(f"Failed to process camera URL: {url}, error: {str(e)}")
            self.stats["failed_urls"] += 1
            return False
    
    async def rate_limited_request(self, url: str, **kwargs) -> Optional[aiohttp.ClientResponse]:
        """
        Make a rate-limited HTTP request.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments for aiohttp.ClientSession.request
            
        Returns:
            Optional[aiohttp.ClientResponse]: Response if successful, None otherwise
        """
        await self.rate_limiter.acquire()
        
        try:
            timeout = aiohttp.ClientTimeout(total=config.discovery.request_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = kwargs.pop("headers", {})
                headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Pragma": "no-cache",
                    "Cache-Control": "no-cache",
                })
                
                method = kwargs.pop("method", "GET")
                
                for attempt in range(config.discovery.max_retries + 1):
                    try:
                        response = await session.request(method, url, headers=headers, **kwargs)
                        return response
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        if attempt < config.discovery.max_retries:
                            wait_time = config.discovery.retry_delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"Request failed, retrying in {wait_time}s: {url}, error: {str(e)}")
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(f"Request failed after {config.discovery.max_retries} retries: {url}, error: {str(e)}")
                            return None
        except Exception as e:
            logger.error(f"Request error: {url}, error: {str(e)}")
            return None


class RateLimiter:
    """
    Rate limiter for HTTP requests.
    
    This class implements a token bucket algorithm for rate limiting.
    """
    
    def __init__(self, rate: int, per: int):
        """
        Initialize the rate limiter.
        
        Args:
            rate: Maximum number of requests
            per: Time window in seconds
        """
        self.rate = rate
        self.per = per
        self.tokens = rate
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """
        Acquire a token for making a request.
        
        This method blocks until a token is available.
        """
        async with self.lock:
            while True:
                # Refill tokens based on elapsed time
                now = time.monotonic()
                elapsed = now - self.last_refill
                
                if elapsed > self.per:
                    self.tokens = self.rate
                    self.last_refill = now
                elif elapsed > 0:
                    new_tokens = elapsed * (self.rate / self.per)
                    self.tokens = min(self.rate, self.tokens + new_tokens)
                    self.last_refill = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                
                # Wait until next token should be available
                wait_time = (1 - self.tokens) * (self.per / self.rate)
                await asyncio.sleep(wait_time)
