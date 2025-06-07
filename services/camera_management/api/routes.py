"""
API routes for the Camera Management service.

This module provides FastAPI routes for camera stream discovery, filtering,
and status information.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..models import Camera, ValidationResult, HealthCheck
from ..db import get_db_session, camera_repo, validation_repo, health_repo, geo_repo, discovery_repo
from ..discovery import DOTDiscoveryWorker
from ..validation.bench_worker import ValidationWorkerPool
from ..health.health_checker import HealthMonitor
from ..config import config

# Configure logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix=config.api.api_prefix)


# Pydantic models for request/response
class CameraResponse(BaseModel):
    """Camera response model."""
    id: int
    url: str
    source: str
    status: str
    fps: Optional[float] = None
    resolution: Optional[str] = None
    latency_ms: Optional[int] = None
    entropy_quality: Optional[float] = None
    location: Optional[str] = None
    geo: Optional[Dict[str, Any]] = None
    discovered_at: str
    last_checked_at: str


class CameraDetailResponse(CameraResponse):
    """Camera detail response model."""
    validation_results: List[Dict[str, Any]] = []
    health_checks: List[Dict[str, Any]] = []
    metadata: Optional[Dict[str, Any]] = None


class CameraStatsResponse(BaseModel):
    """Camera statistics response model."""
    total_cameras: int
    active_cameras: int
    by_status: Dict[str, int]
    by_source: Dict[str, int]
    geographic_distribution: Dict[str, List[Dict[str, Any]]]


class ValidationRequest(BaseModel):
    """Validation request model."""
    camera_ids: List[int] = Field(..., description="List of camera IDs to validate")


class ValidationResponse(BaseModel):
    """Validation response model."""
    results: Dict[str, Any]


class DiscoveryRequest(BaseModel):
    """Discovery request model."""
    source: str = Field(..., description="Source to discover cameras from")


class DiscoveryResponse(BaseModel):
    """Discovery response model."""
    job_id: int
    source: str
    status: str


class HealthCheckRequest(BaseModel):
    """Health check request model."""
    camera_ids: List[int] = Field(..., description="List of camera IDs to check")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    results: Dict[str, Any]


# Dependency to get database session
def get_session():
    """Get database session."""
    with get_db_session() as session:
        yield session


@router.get("/active_cameras", response_model=List[CameraResponse])
async def get_active_cameras(
    session: Session = Depends(get_session),
    skip: int = 0,
    limit: int = 100
):
    """
    Get active cameras.
    
    Args:
        session: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List[CameraResponse]: List of active cameras
    """
    cameras = camera_repo.get_active_cameras(session, skip, limit)
    return [camera.to_dict() for camera in cameras]


@router.get("/cameras", response_model=List[CameraResponse])
async def get_cameras(
    session: Session = Depends(get_session),
    status: Optional[str] = None,
    source: Optional[str] = None,
    min_quality: Optional[float] = None,
    continent: Optional[str] = None,
    country: Optional[str] = None,
    city: Optional[str] = None,
    skip: int = 0,
    limit: int = 100
):
    """
    Get cameras with filtering.
    
    Args:
        session: Database session
        status: Filter by status
        source: Filter by source
        min_quality: Filter by minimum quality
        continent: Filter by continent
        country: Filter by country
        city: Filter by city
        skip: Number of records to skip
        limit: Maximum number of records to return
        
    Returns:
        List[CameraResponse]: List of cameras matching filters
    """
    # Base query
    query = session.query(Camera)
    
    # Apply filters
    if status:
        query = query.filter(Camera.status == status)
    
    if source:
        query = query.filter(Camera.source == source)
    
    if min_quality is not None:
        query = query.filter(Camera.entropy_quality >= min_quality)
    
    if continent:
        query = query.filter(Camera.geo_continent == continent)
    
    if country:
        query = query.filter(Camera.geo_country == country)
    
    if city:
        query = query.filter(Camera.geo_city == city)
    
    # Execute query with pagination
    cameras = query.offset(skip).limit(limit).all()
    
    return [camera.to_dict() for camera in cameras]


@router.get("/cameras/{camera_id}", response_model=CameraDetailResponse)
async def get_camera(
    camera_id: int = Path(..., description="Camera ID"),
    session: Session = Depends(get_session)
):
    """
    Get camera details.
    
    Args:
        camera_id: Camera ID
        session: Database session
        
    Returns:
        CameraDetailResponse: Camera details
    """
    camera = camera_repo.get_by_id(session, camera_id)
    
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Get validation results
    validation_results = validation_repo.get_by_camera_id(session, camera_id, limit=5)
    
    # Get health checks
    health_checks = health_repo.get_by_camera_id(session, camera_id, limit=5)
    
    # Build response
    response = camera.to_dict()
    response["validation_results"] = [result.to_dict() for result in validation_results]
    response["health_checks"] = [check.to_dict() for check in health_checks]
    
    return response


@router.get("/cameras/stats", response_model=CameraStatsResponse)
async def get_camera_stats(session: Session = Depends(get_session)):
    """
    Get camera statistics.
    
    Args:
        session: Database session
        
    Returns:
        CameraStatsResponse: Camera statistics
    """
    # Get total and active cameras
    total_cameras = camera_repo.count(session)
    active_cameras = session.query(Camera).filter(Camera.status == "active").count()
    
    # Get cameras by status
    status_counts = {}
    for status in ["active", "degraded", "offline", "discovered", "validating", "blacklisted"]:
        count = session.query(Camera).filter(Camera.status == status).count()
        status_counts[status] = count
    
    # Get cameras by source
    source_counts = {}
    sources = session.query(Camera.source).distinct().all()
    for (source,) in sources:
        count = session.query(Camera).filter(Camera.source == source).count()
        source_counts[source] = count
    
    # Get geographic distribution
    continent_distribution = geo_repo.get_by_region_type(session, "continent")
    country_distribution = geo_repo.get_by_region_type(session, "country")
    
    return {
        "total_cameras": total_cameras,
        "active_cameras": active_cameras,
        "by_status": status_counts,
        "by_source": source_counts,
        "geographic_distribution": {
            "continent": [dist.to_dict() for dist in continent_distribution],
            "country": [dist.to_dict() for dist in country_distribution]
        }
    }


@router.post("/cameras/validate", response_model=ValidationResponse)
async def validate_cameras(
    request: ValidationRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """
    Validate cameras.
    
    Args:
        request: Validation request
        background_tasks: Background tasks
        session: Database session
        
    Returns:
        ValidationResponse: Validation response
    """
    # Check if cameras exist
    for camera_id in request.camera_ids:
        camera = camera_repo.get_by_id(session, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")
    
    # Start validation in background
    async def validate_in_background():
        async with ValidationWorkerPool(session) as pool:
            await pool.validate_cameras(request.camera_ids)
    
    background_tasks.add_task(validate_in_background)
    
    return {"results": {"message": f"Validation started for {len(request.camera_ids)} cameras"}}


@router.post("/cameras/health_check", response_model=HealthCheckResponse)
async def check_camera_health(
    request: HealthCheckRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """
    Check camera health.
    
    Args:
        request: Health check request
        background_tasks: Background tasks
        session: Database session
        
    Returns:
        HealthCheckResponse: Health check response
    """
    # Check if cameras exist
    for camera_id in request.camera_ids:
        camera = camera_repo.get_by_id(session, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera not found: {camera_id}")
    
    # Start health check in background
    async def check_health_in_background():
        monitor = HealthMonitor(session)
        await monitor.check_cameras(request.camera_ids)
    
    background_tasks.add_task(check_health_in_background)
    
    return {"results": {"message": f"Health check started for {len(request.camera_ids)} cameras"}}


@router.post("/discovery", response_model=DiscoveryResponse)
async def start_discovery(
    request: DiscoveryRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """
    Start camera discovery.
    
    Args:
        request: Discovery request
        background_tasks: Background tasks
        session: Database session
        
    Returns:
        DiscoveryResponse: Discovery response
    """
    # Check if source is valid
    valid_sources = ["DOT", "EarthCam", "Insecam", "Shodan"]
    if request.source not in valid_sources:
        raise HTTPException(status_code=400, detail=f"Invalid source: {request.source}")
    
    # Create discovery job
    job = discovery_repo.create(session, {
        "source": request.source,
        "status": "scheduled"
    })
    
    session.commit()
    
    # Start discovery in background
    async def discover_in_background():
        if request.source == "DOT":
            worker = DOTDiscoveryWorker(session)
            await worker.discover()
        # Add other sources as needed
    
    background_tasks.add_task(discover_in_background)
    
    return {
        "job_id": job.id,
        "source": job.source,
        "status": job.status
    }


@router.get("/discovery/jobs", response_model=List[Dict[str, Any]])
async def get_discovery_jobs(
    session: Session = Depends(get_session),
    source: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 10
):
    """
    Get discovery jobs.
    
    Args:
        session: Database session
        source: Filter by source
        status: Filter by status
        limit: Maximum number of jobs to return
        
    Returns:
        List[Dict[str, Any]]: List of discovery jobs
    """
    # Base query
    query = session.query(DiscoveryJob)
    
    # Apply filters
    if source:
        query = query.filter(DiscoveryJob.source == source)
    
    if status:
        query = query.filter(DiscoveryJob.status == status)
    
    # Execute query with pagination
    jobs = query.order_by(DiscoveryJob.started_at.desc()).limit(limit).all()
    
    return [job.to_dict() for job in jobs]


@router.get("/metrics", response_class=JSONResponse)
async def get_metrics(session: Session = Depends(get_session)):
    """
    Get metrics for Prometheus.
    
    Args:
        session: Database session
        
    Returns:
        JSONResponse: Metrics response
    """
    # Get camera counts
    total_cameras = camera_repo.count(session)
    active_cameras = session.query(Camera).filter(Camera.status == "active").count()
    degraded_cameras = session.query(Camera).filter(Camera.status == "degraded").count()
    offline_cameras = session.query(Camera).filter(Camera.status == "offline").count()
    
    # Get geographic distribution
    continent_distribution = geo_repo.get_by_region_type(session, "continent")
    
    # Get average quality
    avg_quality = session.query(func.avg(Camera.entropy_quality)).filter(
        Camera.entropy_quality.isnot(None)
    ).scalar() or 0
    
    # Build metrics
    metrics = {
        "camera_count_total": total_cameras,
        "camera_count_active": active_cameras,
        "camera_count_degraded": degraded_cameras,
        "camera_count_offline": offline_cameras,
        "camera_quality_avg": round(avg_quality, 4),
        "geographic_distribution": {
            dist.region_name: {
                "total": dist.camera_count,
                "active": dist.active_camera_count,
                "percentage": dist.percentage
            }
            for dist in continent_distribution
        }
    }
    
    return metrics
