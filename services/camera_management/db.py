"""
Database connection and repository implementations for the Camera Management service.

This module provides database connection setup, session management, and
repository pattern implementations for database operations.
"""
import logging
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Generator, TypeVar, Generic, Type, Union
from datetime import datetime, timedelta

from sqlalchemy import create_engine, func, and_, or_, desc, text
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import sessionmaker, Session, Query
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

from .models import Base, Camera, ValidationResult, HealthCheck, GeographicDistribution, DiscoveryJob
from .config import config

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic repository
T = TypeVar('T')

# Create database engine
engine = create_engine(
    config.database.connection_string,
    pool_size=config.database.pool_size,
    max_overflow=config.database.max_overflow,
    pool_timeout=config.database.pool_timeout,
    pool_recycle=config.database.pool_recycle,
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionFactory = sessionmaker(bind=engine)


def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(engine)
    logger.info("Database tables created")


def drop_db():
    """Drop all database tables."""
    Base.metadata.drop_all(engine)
    logger.info("Database tables dropped")


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Yields:
        Session: SQLAlchemy session
        
    Example:
        with get_db_session() as session:
            cameras = session.query(Camera).all()
    """
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


class BaseRepository(Generic[T]):
    """
    Generic base repository for database operations.
    
    This class provides common database operations for any model.
    """
    
    def __init__(self, model_class: Type[T]):
        """
        Initialize the repository with a model class.
        
        Args:
            model_class: SQLAlchemy model class
        """
        self.model_class = model_class
    
    def get_by_id(self, session: Session, id: int) -> Optional[T]:
        """
        Get a record by ID.
        
        Args:
            session: SQLAlchemy session
            id: Record ID
            
        Returns:
            Optional[T]: Record if found, None otherwise
        """
        return session.query(self.model_class).filter(self.model_class.id == id).first()
    
    def get_all(self, session: Session, skip: int = 0, limit: int = 100) -> List[T]:
        """
        Get all records with pagination.
        
        Args:
            session: SQLAlchemy session
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List[T]: List of records
        """
        return session.query(self.model_class).offset(skip).limit(limit).all()
    
    def create(self, session: Session, data: Dict[str, Any]) -> T:
        """
        Create a new record.
        
        Args:
            session: SQLAlchemy session
            data: Record data
            
        Returns:
            T: Created record
        """
        db_item = self.model_class(**data)
        session.add(db_item)
        session.flush()
        return db_item
    
    def update(self, session: Session, id: int, data: Dict[str, Any]) -> Optional[T]:
        """
        Update a record by ID.
        
        Args:
            session: SQLAlchemy session
            id: Record ID
            data: Updated record data
            
        Returns:
            Optional[T]: Updated record if found, None otherwise
        """
        db_item = self.get_by_id(session, id)
        if db_item:
            for key, value in data.items():
                setattr(db_item, key, value)
            session.flush()
        return db_item
    
    def delete(self, session: Session, id: int) -> bool:
        """
        Delete a record by ID.
        
        Args:
            session: SQLAlchemy session
            id: Record ID
            
        Returns:
            bool: True if deleted, False if not found
        """
        db_item = self.get_by_id(session, id)
        if db_item:
            session.delete(db_item)
            return True
        return False
    
    def count(self, session: Session) -> int:
        """
        Count total records.
        
        Args:
            session: SQLAlchemy session
            
        Returns:
            int: Total record count
        """
        return session.query(func.count(self.model_class.id)).scalar()


class CameraRepository(BaseRepository[Camera]):
    """
    Repository for Camera model operations.
    
    This class provides specialized database operations for the Camera model.
    """
    
    def __init__(self):
        """Initialize the repository with the Camera model."""
        super().__init__(Camera)
    
    def get_by_url(self, session: Session, url: str) -> Optional[Camera]:
        """
        Get a camera by URL.
        
        Args:
            session: SQLAlchemy session
            url: Camera URL
            
        Returns:
            Optional[Camera]: Camera if found, None otherwise
        """
        return session.query(Camera).filter(Camera.url == url).first()
    
    def get_by_status(self, session: Session, status: str, skip: int = 0, limit: int = 100) -> List[Camera]:
        """
        Get cameras by status.
        
        Args:
            session: SQLAlchemy session
            status: Camera status
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List[Camera]: List of cameras with the specified status
        """
        return session.query(Camera).filter(Camera.status == status).offset(skip).limit(limit).all()
    
    def get_active_cameras(self, session: Session, skip: int = 0, limit: int = 100) -> List[Camera]:
        """
        Get active cameras.
        
        Args:
            session: SQLAlchemy session
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List[Camera]: List of active cameras
        """
        return session.query(Camera).filter(Camera.status == "active").offset(skip).limit(limit).all()
    
    def get_cameras_by_quality(self, session: Session, min_quality: float, skip: int = 0, limit: int = 100) -> List[Camera]:
        """
        Get cameras by minimum entropy quality.
        
        Args:
            session: SQLAlchemy session
            min_quality: Minimum entropy quality score
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List[Camera]: List of cameras with quality >= min_quality
        """
        return session.query(Camera).filter(Camera.entropy_quality >= min_quality).offset(skip).limit(limit).all()
    
    def get_cameras_by_region(self, session: Session, region_type: str, region_name: str, skip: int = 0, limit: int = 100) -> List[Camera]:
        """
        Get cameras by geographic region.
        
        Args:
            session: SQLAlchemy session
            region_type: Region type (continent, country, city)
            region_name: Region name
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List[Camera]: List of cameras in the specified region
        """
        query = session.query(Camera)
        
        if region_type == "continent":
            query = query.filter(Camera.geo_continent == region_name)
        elif region_type == "country":
            query = query.filter(Camera.geo_country == region_name)
        elif region_type == "city":
            query = query.filter(Camera.geo_city == region_name)
        
        return query.offset(skip).limit(limit).all()
    
    def get_cameras_by_source(self, session: Session, source: str, skip: int = 0, limit: int = 100) -> List[Camera]:
        """
        Get cameras by source.
        
        Args:
            session: SQLAlchemy session
            source: Camera source
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List[Camera]: List of cameras from the specified source
        """
        return session.query(Camera).filter(Camera.source == source).offset(skip).limit(limit).all()
    
    def get_cameras_needing_validation(self, session: Session, limit: int = 10) -> List[Camera]:
        """
        Get cameras that need validation.
        
        Args:
            session: SQLAlchemy session
            limit: Maximum number of cameras to return
            
        Returns:
            List[Camera]: List of cameras needing validation
        """
        # Get cameras with status 'discovered' or cameras that haven't been validated recently
        return session.query(Camera).filter(
            or_(
                Camera.status == "discovered",
                and_(
                    Camera.status.in_(["active", "degraded"]),
                    Camera.last_checked_at < datetime.utcnow() - timedelta(hours=24)
                )
            )
        ).order_by(Camera.last_checked_at).limit(limit).all()
    
    def get_cameras_needing_health_check(self, session: Session, check_interval: int = 30, limit: int = 20) -> List[Camera]:
        """
        Get cameras that need health check.
        
        Args:
            session: SQLAlchemy session
            check_interval: Health check interval in seconds
            limit: Maximum number of cameras to return
            
        Returns:
            List[Camera]: List of cameras needing health check
        """
        # Get active cameras that haven't been checked recently
        return session.query(Camera).filter(
            and_(
                Camera.status.in_(["active", "degraded"]),
                Camera.last_checked_at < datetime.utcnow() - timedelta(seconds=check_interval)
            )
        ).order_by(Camera.last_checked_at).limit(limit).all()
    
    def get_camera_distribution_stats(self, session: Session) -> Dict[str, Any]:
        """
        Get camera distribution statistics.
        
        Args:
            session: SQLAlchemy session
            
        Returns:
            Dict[str, Any]: Camera distribution statistics
        """
        total_cameras = self.count(session)
        active_cameras = session.query(func.count(Camera.id)).filter(Camera.status == "active").scalar()
        
        continent_stats = session.query(
            Camera.geo_continent,
            func.count(Camera.id).label("total"),
            func.count(Camera.id).filter(Camera.status == "active").label("active")
        ).group_by(Camera.geo_continent).all()
        
        country_stats = session.query(
            Camera.geo_country,
            func.count(Camera.id).label("total"),
            func.count(Camera.id).filter(Camera.status == "active").label("active")
        ).group_by(Camera.geo_country).all()
        
        return {
            "total_cameras": total_cameras,
            "active_cameras": active_cameras,
            "continent_distribution": [
                {
                    "continent": stat[0] or "Unknown",
                    "total": stat[1],
                    "active": stat[2],
                    "percentage": round((stat[1] / total_cameras) * 100, 2) if total_cameras > 0 else 0
                }
                for stat in continent_stats
            ],
            "country_distribution": [
                {
                    "country": stat[0] or "Unknown",
                    "total": stat[1],
                    "active": stat[2],
                    "percentage": round((stat[1] / total_cameras) * 100, 2) if total_cameras > 0 else 0
                }
                for stat in country_stats
            ]
        }
    
    def update_camera_status(self, session: Session, camera_id: int, status: str) -> Optional[Camera]:
        """
        Update camera status.
        
        Args:
            session: SQLAlchemy session
            camera_id: Camera ID
            status: New status
            
        Returns:
            Optional[Camera]: Updated camera if found, None otherwise
        """
        camera = self.get_by_id(session, camera_id)
        if camera:
            camera.status = status
            camera.last_checked_at = datetime.utcnow()
            session.flush()
        return camera


class ValidationResultRepository(BaseRepository[ValidationResult]):
    """
    Repository for ValidationResult model operations.
    
    This class provides specialized database operations for the ValidationResult model.
    """
    
    def __init__(self):
        """Initialize the repository with the ValidationResult model."""
        super().__init__(ValidationResult)
    
    def get_by_camera_id(self, session: Session, camera_id: int, limit: int = 10) -> List[ValidationResult]:
        """
        Get validation results for a camera.
        
        Args:
            session: SQLAlchemy session
            camera_id: Camera ID
            limit: Maximum number of results to return
            
        Returns:
            List[ValidationResult]: List of validation results for the camera
        """
        return session.query(ValidationResult).filter(
            ValidationResult.camera_id == camera_id
        ).order_by(desc(ValidationResult.validated_at)).limit(limit).all()
    
    def get_latest_validation(self, session: Session, camera_id: int) -> Optional[ValidationResult]:
        """
        Get the latest validation result for a camera.
        
        Args:
            session: SQLAlchemy session
            camera_id: Camera ID
            
        Returns:
            Optional[ValidationResult]: Latest validation result if exists, None otherwise
        """
        return session.query(ValidationResult).filter(
            ValidationResult.camera_id == camera_id
        ).order_by(desc(ValidationResult.validated_at)).first()
    
    def get_validation_stats(self, session: Session, camera_id: int) -> Dict[str, Any]:
        """
        Get validation statistics for a camera.
        
        Args:
            session: SQLAlchemy session
            camera_id: Camera ID
            
        Returns:
            Dict[str, Any]: Validation statistics
        """
        total_validations = session.query(func.count(ValidationResult.id)).filter(
            ValidationResult.camera_id == camera_id
        ).scalar()
        
        successful_validations = session.query(func.count(ValidationResult.id)).filter(
            and_(
                ValidationResult.camera_id == camera_id,
                ValidationResult.connection_successful == True
            )
        ).scalar()
        
        avg_fps = session.query(func.avg(ValidationResult.measured_fps)).filter(
            ValidationResult.camera_id == camera_id,
            ValidationResult.measured_fps.isnot(None)
        ).scalar()
        
        avg_latency = session.query(func.avg(ValidationResult.stream_latency_ms)).filter(
            ValidationResult.camera_id == camera_id,
            ValidationResult.stream_latency_ms.isnot(None)
        ).scalar()
        
        avg_quality = session.query(func.avg(ValidationResult.overall_quality)).filter(
            ValidationResult.camera_id == camera_id,
            ValidationResult.overall_quality.isnot(None)
        ).scalar()
        
        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "success_rate": round((successful_validations / total_validations) * 100, 2) if total_validations > 0 else 0,
            "avg_fps": round(avg_fps, 2) if avg_fps else None,
            "avg_latency_ms": round(avg_latency, 2) if avg_latency else None,
            "avg_quality": round(avg_quality, 2) if avg_quality else None
        }


class HealthCheckRepository(BaseRepository[HealthCheck]):
    """
    Repository for HealthCheck model operations.
    
    This class provides specialized database operations for the HealthCheck model.
    """
    
    def __init__(self):
        """Initialize the repository with the HealthCheck model."""
        super().__init__(HealthCheck)
    
    def get_by_camera_id(self, session: Session, camera_id: int, limit: int = 10) -> List[HealthCheck]:
        """
        Get health checks for a camera.
        
        Args:
            session: SQLAlchemy session
            camera_id: Camera ID
            limit: Maximum number of health checks to return
            
        Returns:
            List[HealthCheck]: List of health checks for the camera
        """
        return session.query(HealthCheck).filter(
            HealthCheck.camera_id == camera_id
        ).order_by(desc(HealthCheck.checked_at)).limit(limit).all()
    
    def get_latest_health_check(self, session: Session, camera_id: int) -> Optional[HealthCheck]:
        """
        Get the latest health check for a camera.
        
        Args:
            session: SQLAlchemy session
            camera_id: Camera ID
            
        Returns:
            Optional[HealthCheck]: Latest health check if exists, None otherwise
        """
        return session.query(HealthCheck).filter(
            HealthCheck.camera_id == camera_id
        ).order_by(desc(HealthCheck.checked_at)).first()
    
    def get_health_stats(self, session: Session, camera_id: int) -> Dict[str, Any]:
        """
        Get health statistics for a camera.
        
        Args:
            session: SQLAlchemy session
            camera_id: Camera ID
            
        Returns:
            Dict[str, Any]: Health statistics
        """
        total_checks = session.query(func.count(HealthCheck.id)).filter(
            HealthCheck.camera_id == camera_id
        ).scalar()
        
        available_checks = session.query(func.count(HealthCheck.id)).filter(
            and_(
                HealthCheck.camera_id == camera_id,
                HealthCheck.is_available == True
            )
        ).scalar()
        
        avg_response_time = session.query(func.avg(HealthCheck.response_time_ms)).filter(
            HealthCheck.camera_id == camera_id,
            HealthCheck.response_time_ms.isnot(None)
        ).scalar()
        
        # Get recent availability (last 24 hours)
        recent_total = session.query(func.count(HealthCheck.id)).filter(
            and_(
                HealthCheck.camera_id == camera_id,
                HealthCheck.checked_at >= datetime.utcnow() - timedelta(hours=24)
            )
        ).scalar()
        
        recent_available = session.query(func.count(HealthCheck.id)).filter(
            and_(
                HealthCheck.camera_id == camera_id,
                HealthCheck.checked_at >= datetime.utcnow() - timedelta(hours=24),
                HealthCheck.is_available == True
            )
        ).scalar()
        
        return {
            "total_checks": total_checks,
            "available_checks": available_checks,
            "availability_rate": round((available_checks / total_checks) * 100, 2) if total_checks > 0 else 0,
            "avg_response_time_ms": round(avg_response_time, 2) if avg_response_time else None,
            "recent_availability_rate": round((recent_available / recent_total) * 100, 2) if recent_total > 0 else 0
        }


class GeographicDistributionRepository(BaseRepository[GeographicDistribution]):
    """
    Repository for GeographicDistribution model operations.
    
    This class provides specialized database operations for the GeographicDistribution model.
    """
    
    def __init__(self):
        """Initialize the repository with the GeographicDistribution model."""
        super().__init__(GeographicDistribution)
    
    def get_by_region(self, session: Session, region_type: str, region_name: str) -> Optional[GeographicDistribution]:
        """
        Get geographic distribution by region.
        
        Args:
            session: SQLAlchemy session
            region_type: Region type (continent, country, city)
            region_name: Region name
            
        Returns:
            Optional[GeographicDistribution]: Geographic distribution if found, None otherwise
        """
        return session.query(GeographicDistribution).filter(
            and_(
                GeographicDistribution.region_type == region_type,
                GeographicDistribution.region_name == region_name
            )
        ).first()
    
    def get_by_region_type(self, session: Session, region_type: str) -> List[GeographicDistribution]:
        """
        Get geographic distributions by region type.
        
        Args:
            session: SQLAlchemy session
            region_type: Region type (continent, country, city)
            
        Returns:
            List[GeographicDistribution]: List of geographic distributions for the region type
        """
        return session.query(GeographicDistribution).filter(
            GeographicDistribution.region_type == region_type
        ).order_by(desc(GeographicDistribution.camera_count)).all()
    
    def update_distribution_stats(self, session: Session):
        """
        Update geographic distribution statistics.
        
        Args:
            session: SQLAlchemy session
        """
        # Update continent stats
        continent_stats = session.query(
            Camera.geo_continent,
            func.count(Camera.id).label("total"),
            func.count(Camera.id).filter(Camera.status == "active").label("active")
        ).filter(Camera.geo_continent.isnot(None)).group_by(Camera.geo_continent).all()
        
        total_cameras = session.query(func.count(Camera.id)).scalar()
        
        for continent, count, active_count in continent_stats:
            dist = self.get_by_region(session, "continent", continent)
            percentage = round((count / total_cameras) * 100, 2) if total_cameras > 0 else 0
            
            if dist:
                self.update(session, dist.id, {
                    "camera_count": count,
                    "active_camera_count": active_count,
                    "percentage": percentage,
                    "updated_at": datetime.utcnow()
                })
            else:
                self.create(session, {
                    "region_type": "continent",
                    "region_name": continent,
                    "camera_count": count,
                    "active_camera_count": active_count,
                    "percentage": percentage
                })
        
        # Update country stats
        country_stats = session.query(
            Camera.geo_country,
            func.count(Camera.id).label("total"),
            func.count(Camera.id).filter(Camera.status == "active").label("active")
        ).filter(Camera.geo_country.isnot(None)).group_by(Camera.geo_country).all()
        
        for country, count, active_count in country_stats:
            dist = self.get_by_region(session, "country", country)
            percentage = round((count / total_cameras) * 100, 2) if total_cameras > 0 else 0
            
            if dist:
                self.update(session, dist.id, {
                    "camera_count": count,
                    "active_camera_count": active_count,
                    "percentage": percentage,
                    "updated_at": datetime.utcnow()
                })
            else:
                self.create(session, {
                    "region_type": "country",
                    "region_name": country,
                    "camera_count": count,
                    "active_camera_count": active_count,
                    "percentage": percentage
                })


class DiscoveryJobRepository(BaseRepository[DiscoveryJob]):
    """
    Repository for DiscoveryJob model operations.
    
    This class provides specialized database operations for the DiscoveryJob model.
    """
    
    def __init__(self):
        """Initialize the repository with the DiscoveryJob model."""
        super().__init__(DiscoveryJob)
    
    def get_by_source(self, session: Session, source: str, limit: int = 10) -> List[DiscoveryJob]:
        """
        Get discovery jobs by source.
        
        Args:
            session: SQLAlchemy session
            source: Job source
            limit: Maximum number of jobs to return
            
        Returns:
            List[DiscoveryJob]: List of discovery jobs for the source
        """
        return session.query(DiscoveryJob).filter(
            DiscoveryJob.source == source
        ).order_by(desc(DiscoveryJob.started_at)).limit(limit).all()
    
    def get_latest_job(self, session: Session, source: str) -> Optional[DiscoveryJob]:
        """
        Get the latest discovery job for a source.
        
        Args:
            session: SQLAlchemy session
            source: Job source
            
        Returns:
            Optional[DiscoveryJob]: Latest discovery job if exists, None otherwise
        """
        return session.query(DiscoveryJob).filter(
            DiscoveryJob.source == source
        ).order_by(desc(DiscoveryJob.started_at)).first()
    
    def get_running_jobs(self, session: Session) -> List[DiscoveryJob]:
        """
        Get running discovery jobs.
        
        Args:
            session: SQLAlchemy session
            
        Returns:
            List[DiscoveryJob]: List of running discovery jobs
        """
        return session.query(DiscoveryJob).filter(
            DiscoveryJob.status == "running"
        ).all()
    
    def complete_job(self, session: Session, job_id: int, stats: Dict[str, Any]) -> Optional[DiscoveryJob]:
        """
        Complete a discovery job.
        
        Args:
            session: SQLAlchemy session
            job_id: Job ID
            stats: Job statistics
            
        Returns:
            Optional[DiscoveryJob]: Completed job if found, None otherwise
        """
        job = self.get_by_id(session, job_id)
        if job:
            job.completed_at = datetime.utcnow()
            job.status = "completed"
            job.total_discovered = stats.get("total_discovered", 0)
            job.new_cameras = stats.get("new_cameras", 0)
            job.updated_cameras = stats.get("updated_cameras", 0)
            job.failed_urls = stats.get("failed_urls", 0)
            session.flush()
        return job
    
    def fail_job(self, session: Session, job_id: int, error_message: str) -> Optional[DiscoveryJob]:
        """
        Mark a discovery job as failed.
        
        Args:
            session: SQLAlchemy session
            job_id: Job ID
            error_message: Error message
            
        Returns:
            Optional[DiscoveryJob]: Failed job if found, None otherwise
        """
        job = self.get_by_id(session, job_id)
        if job:
            job.completed_at = datetime.utcnow()
            job.status = "failed"
            job.error_message = error_message
            session.flush()
        return job


# Create repository instances
camera_repo = CameraRepository()
validation_repo = ValidationResultRepository()
health_repo = HealthCheckRepository()
geo_repo = GeographicDistributionRepository()
discovery_repo = DiscoveryJobRepository()
