"""
Camera stream validation and benchmarking module.

This module provides functionality for validating and benchmarking camera streams,
measuring metrics like connection success, latency, frame rate, resolution, and
entropy quality.
"""
import asyncio
import logging
import time
from datetime import datetime
import statistics
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np
from sqlalchemy.orm import Session

from ..models import Camera, ValidationResult
from ..db import camera_repo, validation_repo
from ..config import config

# Configure logger
logger = logging.getLogger(__name__)


class StreamValidator:
    """
    Camera stream validator and benchmarker.
    
    This class validates camera streams and measures various metrics.
    """
    
    def __init__(self, session: Session):
        """
        Initialize the stream validator.
        
        Args:
            session: SQLAlchemy session
        """
        self.session = session
    
    async def validate_camera(self, camera_id: int) -> Dict[str, Any]:
        """
        Validate a camera stream.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            # Get camera from database
            camera = camera_repo.get_by_id(self.session, camera_id)
            if not camera:
                logger.error(f"Camera not found: {camera_id}")
                return {"success": False, "error": "Camera not found"}
            
            logger.info(f"Validating camera: {camera.id} ({camera.url})")
            
            # Update camera status to validating
            camera_repo.update_camera_status(self.session, camera.id, "validating")
            self.session.commit()
            
            # Run validation in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            validation_result = await loop.run_in_executor(
                None, self._validate_stream, camera.url
            )
            
            # Create validation result record
            validation_data = {
                "camera_id": camera.id,
                "connection_successful": validation_result["success"],
                "connection_time_ms": validation_result.get("connection_time_ms"),
                "measured_fps": validation_result.get("fps"),
                "frame_width": validation_result.get("width"),
                "frame_height": validation_result.get("height"),
                "stream_latency_ms": validation_result.get("latency_ms"),
                "entropy_score": validation_result.get("entropy_score"),
                "stability_score": validation_result.get("stability_score"),
                "overall_quality": validation_result.get("overall_quality")
            }
            
            if not validation_result["success"]:
                validation_data["error_type"] = validation_result.get("error_type")
                validation_data["error_message"] = validation_result.get("error_message")
            
            validation = validation_repo.create(self.session, validation_data)
            
            # Update camera with validation results
            camera_data = {
                "last_checked_at": datetime.utcnow()
            }
            
            if validation_result["success"]:
                camera_data.update({
                    "status": "active" if validation_result.get("overall_quality", 0) >= config.validation.entropy_quality_threshold else "degraded",
                    "fps": validation_result.get("fps"),
                    "resolution_width": validation_result.get("width"),
                    "resolution_height": validation_result.get("height"),
                    "latency_ms": validation_result.get("latency_ms"),
                    "entropy_quality": validation_result.get("overall_quality")
                })
            else:
                camera_data["status"] = "offline"
            
            camera_repo.update(self.session, camera.id, camera_data)
            self.session.commit()
            
            logger.info(f"Validation completed for camera: {camera.id}, success: {validation_result['success']}")
            
            return validation_result
        except Exception as e:
            logger.error(f"Validation failed for camera: {camera_id}, error: {str(e)}")
            return {"success": False, "error_type": "Exception", "error_message": str(e)}
    
    def _validate_stream(self, url: str) -> Dict[str, Any]:
        """
        Validate a camera stream.
        
        Args:
            url: Camera URL
            
        Returns:
            Dict[str, Any]: Validation results
        """
        result = {
            "success": False,
            "url": url
        }
        
        cap = None
        frames = []
        frame_times = []
        
        try:
            # Measure connection time
            start_time = time.time()
            cap = cv2.VideoCapture(url)
            connection_time = time.time() - start_time
            result["connection_time_ms"] = int(connection_time * 1000)
            
            # Check if connection was successful
            if not cap.isOpened():
                result["error_type"] = "ConnectionError"
                result["error_message"] = "Failed to open camera stream"
                return result
            
            # Get stream properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if width <= 0 or height <= 0:
                result["error_type"] = "InvalidResolution"
                result["error_message"] = f"Invalid resolution: {width}x{height}"
                return result
            
            result["width"] = width
            result["height"] = height
            
            # Capture frames for validation duration
            validation_duration = config.validation.validation_duration
            frame_count = config.validation.frame_sample_count
            target_interval = validation_duration / frame_count
            
            for i in range(frame_count):
                frame_start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    # If we can't read a frame, the stream might be unreliable
                    if i == 0:
                        result["error_type"] = "NoFrames"
                        result["error_message"] = "Could not read any frames from stream"
                        return result
                    break
                
                frame_time = time.time()
                frames.append(frame)
                frame_times.append(frame_time)
                
                # Wait for the next frame capture time
                elapsed = time.time() - frame_start_time
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
            
            # Calculate metrics
            if len(frames) < 2:
                result["error_type"] = "InsufficientFrames"
                result["error_message"] = f"Insufficient frames captured: {len(frames)}"
                return result
            
            # Calculate FPS
            if len(frame_times) >= 2:
                time_diffs = [frame_times[i+1] - frame_times[i] for i in range(len(frame_times)-1)]
                avg_time_diff = statistics.mean(time_diffs)
                fps = 1.0 / avg_time_diff if avg_time_diff > 0 else 0
                result["fps"] = round(fps, 2)
            else:
                result["fps"] = 0
            
            # Calculate latency (approximate)
            result["latency_ms"] = int(connection_time * 1000)
            
            # Calculate entropy score
            entropy_scores = []
            for frame in frames:
                entropy = self._calculate_entropy(frame)
                entropy_scores.append(entropy)
            
            avg_entropy = statistics.mean(entropy_scores)
            result["entropy_score"] = round(avg_entropy, 4)
            
            # Calculate stability score
            if len(frames) >= 2:
                stability_scores = []
                for i in range(len(frames) - 1):
                    stability = self._calculate_frame_stability(frames[i], frames[i+1])
                    stability_scores.append(stability)
                
                avg_stability = statistics.mean(stability_scores)
                result["stability_score"] = round(avg_stability, 4)
            else:
                result["stability_score"] = 0
            
            # Calculate overall quality score (0-1)
            entropy_weight = 0.4
            stability_weight = 0.3
            fps_weight = 0.2
            resolution_weight = 0.1
            
            # Normalize FPS (assuming 30 FPS is max)
            normalized_fps = min(result["fps"] / 30.0, 1.0)
            
            # Normalize resolution (assuming 1920x1080 is max)
            max_resolution = 1920 * 1080
            actual_resolution = width * height
            normalized_resolution = min(actual_resolution / max_resolution, 1.0)
            
            # Calculate overall quality
            overall_quality = (
                entropy_weight * result["entropy_score"] +
                stability_weight * result["stability_score"] +
                fps_weight * normalized_fps +
                resolution_weight * normalized_resolution
            )
            
            result["overall_quality"] = round(overall_quality, 4)
            
            # Set success flag
            result["success"] = True
            
            return result
        except Exception as e:
            result["error_type"] = type(e).__name__
            result["error_message"] = str(e)
            return result
        finally:
            if cap:
                cap.release()
    
    def _calculate_entropy(self, frame: np.ndarray) -> float:
        """
        Calculate the entropy of a frame.
        
        Args:
            frame: Frame image
            
        Returns:
            float: Entropy score (0-1)
        """
        # Convert to grayscale
        if len(frame.shape) > 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        
        # Calculate entropy
        entropy = 0
        for i in hist:
            if i > 0:
                entropy -= i * np.log2(i)
        
        # Normalize to 0-1 (max entropy for 8-bit image is 8)
        normalized_entropy = entropy / 8.0
        
        return normalized_entropy
    
    def _calculate_frame_stability(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate the stability between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            float: Stability score (0-1)
        """
        # Convert to grayscale
        if len(frame1.shape) > 2:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
        
        if len(frame2.shape) > 2:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Calculate mean difference
        mean_diff = np.mean(diff) / 255.0
        
        # Stability is inverse of difference
        stability = 1.0 - mean_diff
        
        return stability


class EntropyQualityTest:
    """
    Entropy quality test for camera streams.
    
    This class implements NIST frequency test for entropy quality assessment.
    """
    
    def __init__(self):
        """Initialize the entropy quality test."""
        pass
    
    def test_entropy_quality(self, frames: List[np.ndarray]) -> float:
        """
        Test the entropy quality of a sequence of frames.
        
        Args:
            frames: List of frames
            
        Returns:
            float: Entropy quality score (0-1)
        """
        if not frames:
            return 0.0
        
        # Convert frames to bit sequence
        bit_sequence = self._frames_to_bits(frames)
        
        # Run NIST frequency test
        p_value = self._frequency_test(bit_sequence)
        
        # Convert p-value to quality score (0-1)
        # Higher p-value means better randomness
        quality_score = min(p_value * 2, 1.0)
        
        return quality_score
    
    def _frames_to_bits(self, frames: List[np.ndarray]) -> List[int]:
        """
        Convert frames to a bit sequence.
        
        Args:
            frames: List of frames
            
        Returns:
            List[int]: Bit sequence
        """
        bit_sequence = []
        
        for frame in frames:
            # Convert to grayscale
            if len(frame.shape) > 2:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Flatten and threshold
            flat = gray.flatten()
            threshold = np.mean(flat)
            
            # Convert to bits
            bits = [1 if pixel > threshold else 0 for pixel in flat]
            
            # Add to sequence (limit to 1000 bits per frame)
            bit_sequence.extend(bits[:1000])
        
        return bit_sequence
    
    def _frequency_test(self, bit_sequence: List[int]) -> float:
        """
        Run NIST frequency test on a bit sequence.
        
        Args:
            bit_sequence: Bit sequence
            
        Returns:
            float: p-value
        """
        if not bit_sequence:
            return 0.0
        
        # Count ones
        ones_count = sum(bit_sequence)
        
        # Calculate statistic
        n = len(bit_sequence)
        s_obs = abs(2 * ones_count - n) / np.sqrt(n)
        
        # Calculate p-value
        p_value = np.exp(-s_obs ** 2 / 2)
        
        return p_value


class ValidationWorkerPool:
    """
    Pool of validation workers.
    
    This class manages a pool of validation workers for concurrent validation.
    """
    
    def __init__(self, session: Session, max_workers: int = None):
        """
        Initialize the validation worker pool.
        
        Args:
            session: SQLAlchemy session
            max_workers: Maximum number of concurrent workers
        """
        self.session = session
        self.max_workers = max_workers or config.validation.max_workers
        self.validator = StreamValidator(session)
        self.semaphore = asyncio.Semaphore(self.max_workers)
    
    async def validate_cameras(self, camera_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Validate multiple cameras concurrently.
        
        Args:
            camera_ids: List of camera IDs
            
        Returns:
            Dict[int, Dict[str, Any]]: Validation results by camera ID
        """
        tasks = []
        for camera_id in camera_ids:
            tasks.append(self._validate_camera_with_semaphore(camera_id))
        
        results = await asyncio.gather(*tasks)
        
        return {camera_id: result for camera_id, result in zip(camera_ids, results)}
    
    async def _validate_camera_with_semaphore(self, camera_id: int) -> Dict[str, Any]:
        """
        Validate a camera with semaphore for concurrency control.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Dict[str, Any]: Validation result
        """
        async with self.semaphore:
            return await self.validator.validate_camera(camera_id)
    
    async def validate_cameras_needing_validation(self, limit: int = None) -> Dict[int, Dict[str, Any]]:
        """
        Validate cameras that need validation.
        
        Args:
            limit: Maximum number of cameras to validate
            
        Returns:
            Dict[int, Dict[str, Any]]: Validation results by camera ID
        """
        # Get cameras needing validation
        with self.session.begin():
            cameras = camera_repo.get_cameras_needing_validation(
                self.session, limit or self.max_workers * 2
            )
        
        camera_ids = [camera.id for camera in cameras]
        
        if not camera_ids:
            logger.info("No cameras need validation")
            return {}
        
        logger.info(f"Validating {len(camera_ids)} cameras")
        
        # Validate cameras
        results = await self.validate_cameras(camera_ids)
        
        return results
