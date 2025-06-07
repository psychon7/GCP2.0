"""
Database models for the Camera Management service.

This module defines SQLAlchemy ORM models for camera streams, validation results,
health status, and geographic distribution.
"""
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    ForeignKey, Text, Index, JSON, Enum, func, text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()

class CameraStatus(enum.Enum):
    """Enum for camera status values."""
    DISCOVERED = "discovered"
    VALIDATING = "validating"
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    BLACKLISTED = "blacklisted"


class Camera(Base):
    """
    Camera stream model representing a public camera feed.
    
    This model stores information about camera streams including their URL,
    source, status, and geographic location.
    """
    __tablename__ = "cameras"
    
    id = Column(Integer, primary_key=True)
    url = Column(String(512), nullable=False, unique=True, index=True)
    source = Column(String(50), nullable=False, index=True)
    discovered_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_checked_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    status = Column(String(20), nullable=False, index=True)
    
    # Technical metrics
    fps = Column(Float)
    resolution_width = Column(Integer)
    resolution_height = Column(Integer)
    latency_ms = Column(Integer)
    entropy_quality = Column(Float)
    
    # Geographic information
    geo_lat = Column(Float, index=True)
    geo_lon = Column(Float, index=True)
    geo_city = Column(String(100))
    geo_country = Column(String(100), index=True)
    geo_continent = Column(String(50), index=True)
    
    # Additional metadata
    metadata = Column(JSON)
    
    # Relationships
    validation_results = relationship("ValidationResult", back_populates="camera", cascade="all, delete-orphan")
    health_checks = relationship("HealthCheck", back_populates="camera", cascade="all, delete-orphan")
    
    # Indexes for efficient querying
    __table_args__ = (
        Index('idx_cameras_status_quality', status, entropy_quality),
        Index('idx_cameras_geo', geo_country, geo_city),
        Index('idx_cameras_discovery', source, discovered_at),
    )
    
    def __repr__(self):
        return f"<Camera(id={self.id}, url='{self.url}', status='{self.status}')>"
    
    @property
    def resolution(self):
        """Get the camera resolution as a string."""
        if self.resolution_width and self.resolution_height:
            return f"{self.resolution_width}x{self.resolution_height}"
        return None
    
    @property
    def geo_location(self):
        """Get the camera location as a formatted string."""
        parts = []
        if self.geo_city:
            parts.append(self.geo_city)
        if self.geo_country:
            parts.append(self.geo_country)
        
        if parts:
            return ", ".join(parts)
        return None
    
    def to_dict(self):
        """Convert the camera model to a dictionary."""
        return {
            "id": self.id,
            "url": self.url,
            "source": self.source,
            "discovered_at": self.discovered_at.isoformat() if self.discovered_at else None,
            "last_checked_at": self.last_checked_at.isoformat() if self.last_checked_at else None,
            "status": self.status,
            "fps": self.fps,
            "resolution": self.resolution,
            "latency_ms": self.latency_ms,
            "entropy_quality": self.entropy_quality,
            "location": self.geo_location,
            "geo": {
                "lat": self.geo_lat,
                "lon": self.geo_lon,
                "city": self.geo_city,
                "country": self.geo_country,
                "continent": self.geo_continent
            },
            "metadata": self.metadata
        }


class ValidationResult(Base):
    """
    Validation result model for camera stream validation.
    
    This model stores the results of camera stream validation including
    connection success, metrics, and quality scores.
    """
    __tablename__ = "validation_results"
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    validated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Connection metrics
    connection_successful = Column(Boolean, nullable=False)
    connection_time_ms = Column(Integer)
    
    # Stream metrics
    measured_fps = Column(Float)
    frame_width = Column(Integer)
    frame_height = Column(Integer)
    stream_latency_ms = Column(Integer)
    
    # Quality metrics
    entropy_score = Column(Float)
    stability_score = Column(Float)
    overall_quality = Column(Float)
    
    # Error information
    error_type = Column(String(100))
    error_message = Column(Text)
    
    # Relationship
    camera = relationship("Camera", back_populates="validation_results")
    
    __table_args__ = (
        Index('idx_validation_camera_date', camera_id, validated_at),
    )
    
    def __repr__(self):
        return f"<ValidationResult(id={self.id}, camera_id={self.camera_id}, success={self.connection_successful})>"
    
    def to_dict(self):
        """Convert the validation result to a dictionary."""
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "connection_successful": self.connection_successful,
            "connection_time_ms": self.connection_time_ms,
            "measured_fps": self.measured_fps,
            "resolution": f"{self.frame_width}x{self.frame_height}" if self.frame_width and self.frame_height else None,
            "stream_latency_ms": self.stream_latency_ms,
            "entropy_score": self.entropy_score,
            "stability_score": self.stability_score,
            "overall_quality": self.overall_quality,
            "error": {
                "type": self.error_type,
                "message": self.error_message
            } if self.error_type else None
        }


class HealthCheck(Base):
    """
    Health check model for camera stream health monitoring.
    
    This model stores the results of periodic health checks for camera streams.
    """
    __tablename__ = "health_checks"
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False, index=True)
    checked_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Health status
    is_available = Column(Boolean, nullable=False)
    response_time_ms = Column(Integer)
    
    # Error information
    error_type = Column(String(100))
    error_message = Column(Text)
    
    # Relationship
    camera = relationship("Camera", back_populates="health_checks")
    
    __table_args__ = (
        Index('idx_health_camera_date', camera_id, checked_at),
    )
    
    def __repr__(self):
        return f"<HealthCheck(id={self.id}, camera_id={self.camera_id}, available={self.is_available})>"
    
    def to_dict(self):
        """Convert the health check to a dictionary."""
        return {
            "id": self.id,
            "camera_id": self.camera_id,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "is_available": self.is_available,
            "response_time_ms": self.response_time_ms,
            "error": {
                "type": self.error_type,
                "message": self.error_message
            } if self.error_type else None
        }


class GeographicDistribution(Base):
    """
    Geographic distribution model for tracking camera distribution.
    
    This model stores statistics about camera distribution across geographic regions.
    """
    __tablename__ = "geographic_distribution"
    
    id = Column(Integer, primary_key=True)
    region_type = Column(String(20), nullable=False)  # 'continent', 'country', 'city'
    region_name = Column(String(100), nullable=False)
    camera_count = Column(Integer, nullable=False, default=0)
    active_camera_count = Column(Integer, nullable=False, default=0)
    percentage = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_geo_dist_unique', region_type, region_name, unique=True),
    )
    
    def __repr__(self):
        return f"<GeographicDistribution(region={self.region_name}, count={self.camera_count})>"
    
    def to_dict(self):
        """Convert the geographic distribution to a dictionary."""
        return {
            "id": self.id,
            "region_type": self.region_type,
            "region_name": self.region_name,
            "camera_count": self.camera_count,
            "active_camera_count": self.active_camera_count,
            "percentage": self.percentage,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DiscoveryJob(Base):
    """
    Discovery job model for tracking camera discovery jobs.
    
    This model stores information about scheduled and completed discovery jobs.
    """
    __tablename__ = "discovery_jobs"
    
    id = Column(Integer, primary_key=True)
    source = Column(String(50), nullable=False, index=True)
    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Job metrics
    total_discovered = Column(Integer, default=0)
    new_cameras = Column(Integer, default=0)
    updated_cameras = Column(Integer, default=0)
    failed_urls = Column(Integer, default=0)
    
    # Job status
    status = Column(String(20), nullable=False, default="running", index=True)
    error_message = Column(Text)
    
    def __repr__(self):
        return f"<DiscoveryJob(id={self.id}, source='{self.source}', status='{self.status}')>"
    
    def to_dict(self):
        """Convert the discovery job to a dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_discovered": self.total_discovered,
            "new_cameras": self.new_cameras,
            "updated_cameras": self.updated_cameras,
            "failed_urls": self.failed_urls,
            "status": self.status,
            "error_message": self.error_message
        }
