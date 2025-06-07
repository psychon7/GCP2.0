"""
Configuration module for the Camera Management service.

This module handles loading and validating configuration from a TOML file
and environment variables.
"""
import os
import tomli
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import structlog

logger = structlog.get_logger(__name__)

class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    connection_string: str = Field(
        default="postgresql://user:pass@localhost:5432/cameras",
        description="PostgreSQL connection string"
    )
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum connection overflow")
    pool_timeout: int = Field(default=30, description="Connection pool timeout in seconds")
    pool_recycle: int = Field(default=1800, description="Connection recycle time in seconds")

class DiscoveryConfig(BaseModel):
    """Configuration for camera discovery workers."""
    max_workers: int = Field(default=10, description="Maximum number of concurrent discovery workers")
    request_timeout: int = Field(default=30, description="HTTP request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries for failed requests")
    retry_delay: int = Field(default=5, description="Delay between retries in seconds")
    
    # Source-specific settings
    dot_enabled: bool = Field(default=True, description="Enable DOT camera discovery")
    earthcam_enabled: bool = Field(default=True, description="Enable EarthCam discovery")
    insecam_enabled: bool = Field(default=True, description="Enable Insecam discovery")
    shodan_enabled: bool = Field(default=True, description="Enable Shodan discovery")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=10, description="Maximum requests per time window")
    rate_limit_window: int = Field(default=60, description="Rate limit time window in seconds")

class ValidationConfig(BaseModel):
    """Configuration for camera validation and benchmarking."""
    max_workers: int = Field(default=5, description="Maximum number of concurrent validation workers")
    connection_timeout: int = Field(default=10, description="Stream connection timeout in seconds")
    fps_threshold: float = Field(default=1.0, description="Minimum acceptable FPS")
    max_latency_ms: int = Field(default=2000, description="Maximum acceptable latency in milliseconds")
    entropy_quality_threshold: float = Field(default=0.7, description="Minimum entropy quality score")
    validation_duration: int = Field(default=10, description="Duration to validate a stream in seconds")
    frame_sample_count: int = Field(default=30, description="Number of frames to sample for validation")

class HealthConfig(BaseModel):
    """Configuration for health monitoring."""
    check_interval: int = Field(default=30, description="Health check interval in seconds")
    min_pool_size: int = Field(default=100, description="Minimum camera pool size")
    availability_threshold: float = Field(default=0.8, description="Minimum availability threshold")
    max_consecutive_failures: int = Field(default=5, description="Maximum consecutive failures before pruning")
    backoff_base: float = Field(default=2.0, description="Exponential backoff base")
    backoff_max: int = Field(default=3600, description="Maximum backoff time in seconds")

class ApiConfig(BaseModel):
    """Configuration for the API server."""
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    workers: int = Field(default=4, description="Number of worker processes")
    cors_origins: list = Field(default=["*"], description="Allowed CORS origins")
    api_prefix: str = Field(default="/api/v1", description="API endpoint prefix")

class GeographicConfig(BaseModel):
    """Configuration for geographic distribution."""
    max_continent_percentage: float = Field(default=0.3, description="Maximum percentage of cameras per continent")
    max_country_percentage: float = Field(default=0.2, description="Maximum percentage of cameras per country")
    min_continents: int = Field(default=3, description="Minimum number of continents with cameras")
    min_countries: int = Field(default=10, description="Minimum number of countries with cameras")

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="json", description="Log format (json or console)")
    output: str = Field(default="stdout", description="Log output destination")
    
    @validator("level")
    def validate_level(cls, v):
        """Validate logging level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()
    
    @validator("format")
    def validate_format(cls, v):
        """Validate log format."""
        allowed = ["json", "console"]
        if v.lower() not in allowed:
            raise ValueError(f"Log format must be one of {allowed}")
        return v.lower()

class CameraManagementConfig(BaseSettings):
    """Main configuration class for the Camera Management service."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    geographic: GeographicConfig = Field(default_factory=GeographicConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    class Config:
        env_prefix = "CAMERA_MGMT_"
        case_sensitive = False

def load_config(config_path: Optional[str] = None) -> CameraManagementConfig:
    """
    Load configuration from a TOML file and environment variables.
    
    Args:
        config_path: Path to the TOML configuration file. If None, looks for
                    camera_mgmt.toml in the current directory and common locations.
    
    Returns:
        CameraManagementConfig: The loaded and validated configuration.
    """
    config_dict = {}
    
    # Default config paths to check
    default_paths = [
        Path("camera_mgmt.toml"),
        Path("config/camera_mgmt.toml"),
        Path("/etc/gcp-trng/camera_mgmt.toml"),
        Path(os.path.expanduser("~/.config/gcp-trng/camera_mgmt.toml")),
    ]
    
    # If config_path is provided, use it instead of defaults
    if config_path:
        paths_to_check = [Path(config_path)]
    else:
        paths_to_check = default_paths
    
    # Try to load config from file
    for path in paths_to_check:
        try:
            if path.exists():
                with open(path, "rb") as f:
                    config_dict = tomli.load(f)
                logger.info(f"Loaded configuration from {path}")
                break
        except Exception as e:
            logger.warning(f"Failed to load configuration from {path}: {e}")
    
    # Create config object, which will also load from environment variables
    config = CameraManagementConfig(**config_dict)
    
    return config

# Global config instance
config = load_config()
