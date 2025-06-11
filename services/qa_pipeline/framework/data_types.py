"""
Data type definitions for the QA Pipeline statistical testing framework.

This module contains Pydantic models for:
- EntropySample: Represents an entropy data sample to be tested
- TestConfiguration: Parameters and settings for statistical tests
- TestResult: Results from individual statistical tests
- OverallQualityScore: Aggregated quality score for a sample
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    """Possible status values for a statistical test result."""
    
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"  # Test encountered an error during execution
    SKIPPED = "skipped"  # Test was not run


class EntropySample(BaseModel):
    """Representation of an entropy sample to be tested."""
    
    # Unique identifier for the sample
    sample_id: str = Field(..., description="Unique identifier for this sample")
    
    # Binary/hexadecimal/raw data content
    data: Union[bytes, str, List[int]] = Field(..., description="The entropy data")
    
    # Data format specification
    data_format: str = Field("binary", description="Format of the data (binary, hex, int_array)")
    
    # Metadata
    source_id: Optional[str] = Field(None, description="ID of the entropy source that produced this sample")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this sample was collected")
    
    # Optional additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the sample")


class TestParameter(BaseModel):
    """Parameter for a statistical test."""
    
    name: str
    value: Any
    description: Optional[str] = None


class TestConfiguration(BaseModel):
    """Configuration for a statistical test."""
    
    test_id: str = Field(..., description="Unique identifier for the test")
    enabled: bool = Field(True, description="Whether this test is enabled")
    
    # Test-specific parameters
    parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Test-specific parameters"
    )
    
    # Pass/fail threshold - interpretation depends on the specific test
    threshold: Optional[float] = Field(
        None, 
        description="Threshold for pass/fail determination"
    )
    
    # Weight of this test in the overall quality score calculation (0.0 to 1.0)
    weight: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Weight of this test in overall score calculation"
    )


class TestResult(BaseModel):
    """Result of a statistical test."""
    
    # Test identification
    test_id: str = Field(..., description="ID of the test that produced this result")
    test_name: str = Field(..., description="Human-readable name of the test")
    
    # Test execution details
    status: TestStatus = Field(..., description="Status of the test execution")
    
    # Test metrics and results
    p_value: Optional[float] = Field(None, description="p-value of the test, if applicable")
    score: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=100.0,
        description="Score from 0-100 representing quality based on this test"
    )
    
    # Pass/fail result
    passed: Optional[bool] = Field(None, description="Whether the test was passed")
    
    # Free-form test statistics
    statistics: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Test-specific statistics and metrics"
    )
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message if test failed with an error")
    
    # Time taken to run the test
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")


class OverallQualityScore(BaseModel):
    """Overall quality score for an entropy sample."""
    
    # Sample identification
    sample_id: str = Field(..., description="ID of the sample that was tested")
    
    # Overall score
    score: float = Field(
        ..., 
        ge=0.0, 
        le=100.0,
        description="Overall quality score from 0-100"
    )
    
    # Individual test results
    test_results: List[TestResult] = Field(
        ..., 
        description="Results of individual tests"
    )
    
    # Summary stats
    total_tests: int = Field(..., description="Total number of tests run")
    passed_tests: int = Field(..., description="Number of tests passed")
    failed_tests: int = Field(..., description="Number of tests failed")
    error_tests: int = Field(..., description="Number of tests that encountered errors")
    skipped_tests: int = Field(..., description="Number of tests that were skipped")
    
    # Timestamp
    timestamp: datetime = Field(
        default_factory=datetime.now, 
        description="When the tests were completed"
    )
