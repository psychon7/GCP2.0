"""
DOT (Department of Transportation) camera discovery worker.

This module implements a discovery worker for scraping traffic cameras
from Department of Transportation websites.
"""
import asyncio
import logging
import re
from typing import List, Dict, Any, Set
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
from sqlalchemy.orm import Session

from . import DiscoveryWorker
from ..config import config

# Configure logger
logger = logging.getLogger(__name__)

# List of DOT websites to scrape
DOT_SOURCES = [
    {
        "name": "WA DOT",
        "url": "https://www.wsdot.com/traffic/cameras/",
        "country": "United States",
        "continent": "North America",
        "state": "Washington"
    },
    {
        "name": "NY DOT",
        "url": "https://511ny.org/map#Camera",
        "country": "United States",
        "continent": "North America",
        "state": "New York"
    },
    {
        "name": "CA DOT",
        "url": "https://cwwp2.dot.ca.gov/vm/iframemap.htm",
        "country": "United States",
        "continent": "North America",
        "state": "California"
    },
    {
        "name": "FL DOT",
        "url": "https://fl511.com/map",
        "country": "United States",
        "continent": "North America",
        "state": "Florida"
    },
    {
        "name": "TX DOT",
        "url": "https://drivetexas.org/#/7/32.340/-99.500?future=false",
        "country": "United States",
        "continent": "North America",
        "state": "Texas"
    },
    {
        "name": "UK Traffic Cameras",
        "url": "https://www.trafficcameras.uk/",
        "country": "United Kingdom",
        "continent": "Europe"
    },
    {
        "name": "TfL Traffic Cameras",
        "url": "https://www.tfljamcams.net/",
        "country": "United Kingdom",
        "continent": "Europe",
        "city": "London"
    },
    {
        "name": "Australia Live Traffic",
        "url": "https://www.livetraffic.com/traffic-cameras",
        "country": "Australia",
        "continent": "Oceania",
        "state": "New South Wales"
    }
]


class DOTDiscoveryWorker(DiscoveryWorker):
    """
    Discovery worker for Department of Transportation traffic cameras.
    
    This worker scrapes traffic camera feeds from various DOT websites.
    """
    
    def __init__(self, session: Session):
        """
        Initialize the DOT discovery worker.
        
        Args:
            session: SQLAlchemy session
        """
        super().__init__(session, "DOT")
        self.processed_urls = set()
    
    async def _discover(self):
        """
        Implement the DOT discovery logic.
        
        This method scrapes traffic camera feeds from DOT websites.
        """
        tasks = []
        for source in DOT_SOURCES:
            tasks.append(self._process_dot_source(source))
        
        # Run tasks concurrently with a limit
        semaphore = asyncio.Semaphore(config.discovery.max_workers)
        
        async def bounded_process(task):
            async with semaphore:
                return await task
        
        await asyncio.gather(*(bounded_process(task) for task in tasks))
    
    async def _process_dot_source(self, source: Dict[str, Any]):
        """
        Process a DOT source.
        
        Args:
            source: DOT source information
        """
        try:
            logger.info(f"Processing DOT source: {source['name']}")
            
            # Determine the appropriate scraper based on the source
            if "wsdot.com" in source["url"]:
                await self._scrape_wsdot(source)
            elif "511ny.org" in source["url"]:
                await self._scrape_511ny(source)
            elif "dot.ca.gov" in source["url"]:
                await self._scrape_caltrans(source)
            elif "fl511.com" in source["url"]:
                await self._scrape_fl511(source)
            elif "drivetexas.org" in source["url"]:
                await self._scrape_texas_dot(source)
            elif "trafficcameras.uk" in source["url"]:
                await self._scrape_uk_traffic(source)
            elif "tfljamcams.net" in source["url"]:
                await self._scrape_tfl(source)
            elif "livetraffic.com" in source["url"]:
                await self._scrape_australia_traffic(source)
            else:
                # Generic scraper for unknown sources
                await self._scrape_generic_dot(source)
            
            logger.info(f"Completed processing DOT source: {source['name']}")
        except Exception as e:
            logger.error(f"Failed to process DOT source: {source['name']}, error: {str(e)}")
    
    async def _scrape_wsdot(self, source: Dict[str, Any]):
        """
        Scrape Washington State DOT cameras.
        
        Args:
            source: DOT source information
        """
        response = await self.rate_limited_request(source["url"])
        if not response:
            return
        
        html = await response.text()
        soup = BeautifulSoup(html, "html.parser")
        
        # Find camera links
        camera_links = soup.select("a.CameraLink")
        
        for link in camera_links:
            try:
                camera_url = link.get("href")
                if not camera_url or camera_url in self.processed_urls:
                    continue
                
                self.processed_urls.add(camera_url)
                
                # Get camera details
                camera_name = link.text.strip()
                camera_page = await self.rate_limited_request(urljoin(source["url"], camera_url))
                if not camera_page:
                    continue
                
                camera_html = await camera_page.text()
                camera_soup = BeautifulSoup(camera_html, "html.parser")
                
                # Find image URL
                img_tag = camera_soup.select_one("img.CameraImage")
                if not img_tag:
                    continue
                
                img_url = img_tag.get("src")
                if not img_url:
                    continue
                
                # Create metadata
                metadata = {
                    "name": camera_name,
                    "source_name": source["name"],
                    "source_url": source["url"],
                    "geo_country": source["country"],
                    "geo_continent": source["continent"],
                    "geo_state": source.get("state"),
                    "camera_type": "traffic",
                    "refresh_rate": 60  # Typical refresh rate for traffic cameras
                }
                
                # Process the camera URL
                await self.process_camera_url(img_url, metadata)
            except Exception as e:
                logger.error(f"Failed to process WSDOT camera link: {link}, error: {str(e)}")
    
    async def _scrape_511ny(self, source: Dict[str, Any]):
        """
        Scrape New York 511 cameras.
        
        Args:
            source: DOT source information
        """
        # 511NY uses a JavaScript API, so we need to access the API directly
        api_url = "https://511ny.org/api/getjsonp/mapitemsjson"
        
        response = await self.rate_limited_request(api_url)
        if not response:
            return
        
        # Response is JSONP, extract the JSON part
        text = await response.text()
        json_match = re.search(r'mapItemsJson\((.*)\)', text)
        if not json_match:
            return
        
        try:
            data = json.loads(json_match.group(1))
            cameras = data.get("Cameras", [])
            
            for camera in cameras:
                try:
                    camera_id = camera.get("Id")
                    if not camera_id:
                        continue
                    
                    # Construct camera URL
                    camera_url = f"https://511ny.org/api/cameras/{camera_id}/image"
                    if camera_url in self.processed_urls:
                        continue
                    
                    self.processed_urls.add(camera_url)
                    
                    # Extract location information
                    location = camera.get("Name", "")
                    latitude = camera.get("Latitude")
                    longitude = camera.get("Longitude")
                    
                    # Create metadata
                    metadata = {
                        "name": location,
                        "source_name": source["name"],
                        "source_url": source["url"],
                        "geo_country": source["country"],
                        "geo_continent": source["continent"],
                        "geo_state": source.get("state"),
                        "geo_lat": latitude,
                        "geo_lon": longitude,
                        "camera_type": "traffic",
                        "refresh_rate": 60
                    }
                    
                    # Process the camera URL
                    await self.process_camera_url(camera_url, metadata)
                except Exception as e:
                    logger.error(f"Failed to process 511NY camera: {camera}, error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to parse 511NY JSON data: {str(e)}")
    
    async def _scrape_caltrans(self, source: Dict[str, Any]):
        """
        Scrape California DOT (Caltrans) cameras.
        
        Args:
            source: DOT source information
        """
        # Caltrans uses a JavaScript API
        api_url = "https://cwwp2.dot.ca.gov/data/d3/cctv/cctvStatusD03.json"
        
        response = await self.rate_limited_request(api_url)
        if not response:
            return
        
        try:
            data = await response.json()
            cameras = data.get("data", [])
            
            for camera in cameras:
                try:
                    camera_id = camera.get("id")
                    if not camera_id:
                        continue
                    
                    # Construct camera URL
                    camera_url = f"https://cwwp2.dot.ca.gov/data/d3/cctv/image/D3-CCTV-{camera_id}.jpg"
                    if camera_url in self.processed_urls:
                        continue
                    
                    self.processed_urls.add(camera_url)
                    
                    # Extract location information
                    location = camera.get("title", "")
                    latitude = camera.get("latitude")
                    longitude = camera.get("longitude")
                    
                    # Create metadata
                    metadata = {
                        "name": location,
                        "source_name": source["name"],
                        "source_url": source["url"],
                        "geo_country": source["country"],
                        "geo_continent": source["continent"],
                        "geo_state": source.get("state"),
                        "geo_lat": latitude,
                        "geo_lon": longitude,
                        "camera_type": "traffic",
                        "refresh_rate": 60
                    }
                    
                    # Process the camera URL
                    await self.process_camera_url(camera_url, metadata)
                except Exception as e:
                    logger.error(f"Failed to process Caltrans camera: {camera}, error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to parse Caltrans JSON data: {str(e)}")
    
    async def _scrape_fl511(self, source: Dict[str, Any]):
        """
        Scrape Florida 511 cameras.
        
        Args:
            source: DOT source information
        """
        # FL511 uses a JavaScript API
        api_url = "https://fl511.com/map/data/Cameras.json"
        
        response = await self.rate_limited_request(api_url)
        if not response:
            return
        
        try:
            data = await response.json()
            cameras = data.get("features", [])
            
            for camera in cameras:
                try:
                    properties = camera.get("properties", {})
                    camera_id = properties.get("id")
                    if not camera_id:
                        continue
                    
                    # Construct camera URL
                    camera_url = f"https://fl511.com/map/Cctv/{camera_id}"
                    if camera_url in self.processed_urls:
                        continue
                    
                    self.processed_urls.add(camera_url)
                    
                    # Extract location information
                    location = properties.get("name", "")
                    geometry = camera.get("geometry", {})
                    coordinates = geometry.get("coordinates", [])
                    
                    if len(coordinates) >= 2:
                        longitude, latitude = coordinates
                    else:
                        latitude, longitude = None, None
                    
                    # Create metadata
                    metadata = {
                        "name": location,
                        "source_name": source["name"],
                        "source_url": source["url"],
                        "geo_country": source["country"],
                        "geo_continent": source["continent"],
                        "geo_state": source.get("state"),
                        "geo_lat": latitude,
                        "geo_lon": longitude,
                        "camera_type": "traffic",
                        "refresh_rate": 60
                    }
                    
                    # Process the camera URL
                    await self.process_camera_url(camera_url, metadata)
                except Exception as e:
                    logger.error(f"Failed to process FL511 camera: {camera}, error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to parse FL511 JSON data: {str(e)}")
    
    async def _scrape_texas_dot(self, source: Dict[str, Any]):
        """
        Scrape Texas DOT cameras.
        
        Args:
            source: DOT source information
        """
        # Texas DOT uses a JavaScript API
        api_url = "https://drivetexas.org/map/data/cameras.json"
        
        response = await self.rate_limited_request(api_url)
        if not response:
            return
        
        try:
            data = await response.json()
            cameras = data.get("features", [])
            
            for camera in cameras:
                try:
                    properties = camera.get("properties", {})
                    camera_id = properties.get("id")
                    if not camera_id:
                        continue
                    
                    # Construct camera URL
                    camera_url = f"https://its.txdot.gov/ITS_WEB/FrontEnd/images/cameras/{camera_id}.jpg"
                    if camera_url in self.processed_urls:
                        continue
                    
                    self.processed_urls.add(camera_url)
                    
                    # Extract location information
                    location = properties.get("name", "")
                    geometry = camera.get("geometry", {})
                    coordinates = geometry.get("coordinates", [])
                    
                    if len(coordinates) >= 2:
                        longitude, latitude = coordinates
                    else:
                        latitude, longitude = None, None
                    
                    # Create metadata
                    metadata = {
                        "name": location,
                        "source_name": source["name"],
                        "source_url": source["url"],
                        "geo_country": source["country"],
                        "geo_continent": source["continent"],
                        "geo_state": source.get("state"),
                        "geo_lat": latitude,
                        "geo_lon": longitude,
                        "camera_type": "traffic",
                        "refresh_rate": 60
                    }
                    
                    # Process the camera URL
                    await self.process_camera_url(camera_url, metadata)
                except Exception as e:
                    logger.error(f"Failed to process Texas DOT camera: {camera}, error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to parse Texas DOT JSON data: {str(e)}")
    
    async def _scrape_uk_traffic(self, source: Dict[str, Any]):
        """
        Scrape UK traffic cameras.
        
        Args:
            source: DOT source information
        """
        response = await self.rate_limited_request(source["url"])
        if not response:
            return
        
        html = await response.text()
        soup = BeautifulSoup(html, "html.parser")
        
        # Find camera links
        camera_links = soup.select("a.camera-link")
        
        for link in camera_links:
            try:
                camera_url = link.get("href")
                if not camera_url or camera_url in self.processed_urls:
                    continue
                
                camera_url = urljoin(source["url"], camera_url)
                self.processed_urls.add(camera_url)
                
                # Get camera details
                camera_page = await self.rate_limited_request(camera_url)
                if not camera_page:
                    continue
                
                camera_html = await camera_page.text()
                camera_soup = BeautifulSoup(camera_html, "html.parser")
                
                # Find image URL
                img_tag = camera_soup.select_one("img.camera-image")
                if not img_tag:
                    continue
                
                img_url = img_tag.get("src")
                if not img_url:
                    continue
                
                img_url = urljoin(camera_url, img_url)
                
                # Extract location information
                location = camera_soup.select_one("h1.camera-title")
                location_text = location.text.strip() if location else ""
                
                # Create metadata
                metadata = {
                    "name": location_text,
                    "source_name": source["name"],
                    "source_url": source["url"],
                    "geo_country": source["country"],
                    "geo_continent": source["continent"],
                    "geo_city": source.get("city"),
                    "camera_type": "traffic",
                    "refresh_rate": 60
                }
                
                # Process the camera URL
                await self.process_camera_url(img_url, metadata)
            except Exception as e:
                logger.error(f"Failed to process UK traffic camera link: {link}, error: {str(e)}")
    
    async def _scrape_tfl(self, source: Dict[str, Any]):
        """
        Scrape Transport for London (TfL) traffic cameras.
        
        Args:
            source: DOT source information
        """
        response = await self.rate_limited_request(source["url"])
        if not response:
            return
        
        html = await response.text()
        soup = BeautifulSoup(html, "html.parser")
        
        # Find camera links
        camera_links = soup.select("a.jam-cam-link")
        
        for link in camera_links:
            try:
                camera_url = link.get("href")
                if not camera_url or camera_url in self.processed_urls:
                    continue
                
                camera_url = urljoin(source["url"], camera_url)
                self.processed_urls.add(camera_url)
                
                # Get camera details
                camera_page = await self.rate_limited_request(camera_url)
                if not camera_page:
                    continue
                
                camera_html = await camera_page.text()
                camera_soup = BeautifulSoup(camera_html, "html.parser")
                
                # Find image URL
                img_tag = camera_soup.select_one("img.jam-cam-image")
                if not img_tag:
                    continue
                
                img_url = img_tag.get("src")
                if not img_url:
                    continue
                
                img_url = urljoin(camera_url, img_url)
                
                # Extract location information
                location = camera_soup.select_one("h1.jam-cam-title")
                location_text = location.text.strip() if location else ""
                
                # Create metadata
                metadata = {
                    "name": location_text,
                    "source_name": source["name"],
                    "source_url": source["url"],
                    "geo_country": source["country"],
                    "geo_continent": source["continent"],
                    "geo_city": source.get("city", "London"),
                    "camera_type": "traffic",
                    "refresh_rate": 60
                }
                
                # Process the camera URL
                await self.process_camera_url(img_url, metadata)
            except Exception as e:
                logger.error(f"Failed to process TfL camera link: {link}, error: {str(e)}")
    
    async def _scrape_australia_traffic(self, source: Dict[str, Any]):
        """
        Scrape Australian traffic cameras.
        
        Args:
            source: DOT source information
        """
        # Australian traffic cameras use a JavaScript API
        api_url = "https://www.livetraffic.com/traffic-cameras/data/cameras.json"
        
        response = await self.rate_limited_request(api_url)
        if not response:
            return
        
        try:
            data = await response.json()
            cameras = data.get("features", [])
            
            for camera in cameras:
                try:
                    properties = camera.get("properties", {})
                    camera_id = properties.get("id")
                    if not camera_id:
                        continue
                    
                    # Construct camera URL
                    camera_url = f"https://www.livetraffic.com/traffic-cameras/images/{camera_id}.jpg"
                    if camera_url in self.processed_urls:
                        continue
                    
                    self.processed_urls.add(camera_url)
                    
                    # Extract location information
                    location = properties.get("name", "")
                    geometry = camera.get("geometry", {})
                    coordinates = geometry.get("coordinates", [])
                    
                    if len(coordinates) >= 2:
                        longitude, latitude = coordinates
                    else:
                        latitude, longitude = None, None
                    
                    # Create metadata
                    metadata = {
                        "name": location,
                        "source_name": source["name"],
                        "source_url": source["url"],
                        "geo_country": source["country"],
                        "geo_continent": source["continent"],
                        "geo_state": source.get("state"),
                        "geo_lat": latitude,
                        "geo_lon": longitude,
                        "camera_type": "traffic",
                        "refresh_rate": 60
                    }
                    
                    # Process the camera URL
                    await self.process_camera_url(camera_url, metadata)
                except Exception as e:
                    logger.error(f"Failed to process Australian traffic camera: {camera}, error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to parse Australian traffic JSON data: {str(e)}")
    
    async def _scrape_generic_dot(self, source: Dict[str, Any]):
        """
        Generic scraper for DOT websites.
        
        Args:
            source: DOT source information
        """
        response = await self.rate_limited_request(source["url"])
        if not response:
            return
        
        html = await response.text()
        soup = BeautifulSoup(html, "html.parser")
        
        # Look for common patterns in DOT websites
        
        # 1. Look for image tags with common camera URL patterns
        img_patterns = [
            r"camera", r"cam", r"cctv", r"traffic", r"highway", r"road",
            r"\.jpg", r"\.jpeg", r"\.png", r"\.gif"
        ]
        
        for img in soup.find_all("img"):
            try:
                img_url = img.get("src")
                if not img_url:
                    continue
                
                img_url = urljoin(source["url"], img_url)
                
                # Check if URL matches any camera pattern
                if not any(re.search(pattern, img_url, re.IGNORECASE) for pattern in img_patterns):
                    continue
                
                if img_url in self.processed_urls:
                    continue
                
                self.processed_urls.add(img_url)
                
                # Try to find a name for the camera
                name = img.get("alt", "") or img.get("title", "")
                if not name:
                    parent = img.parent
                    if parent and parent.name == "a":
                        name = parent.get("title", "") or parent.text.strip()
                
                # Create metadata
                metadata = {
                    "name": name or f"Camera from {source['name']}",
                    "source_name": source["name"],
                    "source_url": source["url"],
                    "geo_country": source["country"],
                    "geo_continent": source["continent"],
                    "geo_state": source.get("state"),
                    "geo_city": source.get("city"),
                    "camera_type": "traffic",
                    "refresh_rate": 60
                }
                
                # Process the camera URL
                await self.process_camera_url(img_url, metadata)
            except Exception as e:
                logger.error(f"Failed to process generic DOT camera image: {img}, error: {str(e)}")
        
        # 2. Look for links that might lead to camera pages
        link_patterns = [
            r"camera", r"cam", r"cctv", r"traffic", r"highway", r"road",
            r"view", r"live", r"stream"
        ]
        
        for link in soup.find_all("a"):
            try:
                href = link.get("href")
                if not href:
                    continue
                
                href = urljoin(source["url"], href)
                
                # Check if URL matches any camera pattern
                if not any(re.search(pattern, href, re.IGNORECASE) for pattern in link_patterns):
                    continue
                
                if href in self.processed_urls:
                    continue
                
                self.processed_urls.add(href)
                
                # Follow the link to find camera images
                link_response = await self.rate_limited_request(href)
                if not link_response:
                    continue
                
                link_html = await link_response.text()
                link_soup = BeautifulSoup(link_html, "html.parser")
                
                # Look for images on the linked page
                for img in link_soup.find_all("img"):
                    try:
                        img_url = img.get("src")
                        if not img_url:
                            continue
                        
                        img_url = urljoin(href, img_url)
                        
                        # Check if URL matches any camera pattern
                        if not any(re.search(pattern, img_url, re.IGNORECASE) for pattern in img_patterns):
                            continue
                        
                        if img_url in self.processed_urls:
                            continue
                        
                        self.processed_urls.add(img_url)
                        
                        # Try to find a name for the camera
                        name = img.get("alt", "") or img.get("title", "") or link.text.strip()
                        
                        # Create metadata
                        metadata = {
                            "name": name or f"Camera from {source['name']}",
                            "source_name": source["name"],
                            "source_url": source["url"],
                            "geo_country": source["country"],
                            "geo_continent": source["continent"],
                            "geo_state": source.get("state"),
                            "geo_city": source.get("city"),
                            "camera_type": "traffic",
                            "refresh_rate": 60
                        }
                        
                        # Process the camera URL
                        await self.process_camera_url(img_url, metadata)
                    except Exception as e:
                        logger.error(f"Failed to process generic DOT camera image from link: {img}, error: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to process generic DOT camera link: {link}, error: {str(e)}")
