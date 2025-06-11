"""
Abstract base class for all statistical tests in the QA Pipeline framework.

This module defines the StatisticalTest ABC that all concrete test implementations
must inherit from to be pluggable into the test framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .data_types import EntropySample, TestConfiguration, TestResult


class StatisticalTest(ABC):
    """
    Abstract base class for all statistical tests.
    
    Every statistical test must inherit from this class and implement its abstract methods
    to be usable with the test runner.
    """
    
    def __init__(self, config: TestConfiguration):
        """
        Initialize the test with a configuration.
        
        Args:
            config: Test-specific configuration parameters
        """
        self.config = config
        self.id = config.test_id
        self.threshold = config.threshold
        self._result: Optional[TestResult] = None
        self._raw_data: Optional[EntropySample] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return a human-readable name for this test.
        
        Returns:
            Human-readable name of the test
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Return a detailed description of what this test evaluates.
        
        Returns:
            Description of the test's purpose and metrics
        """
        pass
    
    @abstractmethod
    def prepare_data(self, sample: EntropySample) -> Any:
        """
        Prepare/preprocess the input data for testing.
        
        This method should convert the raw entropy sample into the format
        required by the specific test implementation.
        
        Args:
            sample: The entropy sample to prepare
            
        Returns:
            Processed data in a format suitable for the test algorithm
        
        Raises:
            ValueError: If the data format is invalid or incompatible with this test
        """
        pass
    
    @abstractmethod
    def _execute_test(self, processed_data: Any) -> Dict[str, Any]:
        """
        Execute the statistical test algorithm.
        
        This should be implemented by each specific test to perform its
        actual statistical analysis.
        
        Args:
            processed_data: Data that has been prepared by prepare_data()
            
        Returns:
            Dictionary of test statistics
            
        Raises:
            Exception: If the test fails to execute
        """
        pass
    
    @abstractmethod
    def calculate_score(self, test_statistics: Dict[str, Any]) -> float:
        """
        Calculate a quality score from the test statistics.
        
        Convert test-specific statistics into a quality score from 0-100.
        
        Args:
            test_statistics: The statistics computed by _execute_test()
            
        Returns:
            Quality score from 0-100
        """
        pass
    
    @abstractmethod
    def interpret_results(self, test_statistics: Dict[str, Any]) -> bool:
        """
        Determine if the test was passed based on statistics and thresholds.
        
        Args:
            test_statistics: The statistics computed by _execute_test()
            
        Returns:
            True if test passed, False if failed
        """
        pass
    
    def run_test(self, sample: EntropySample) -> TestResult:
        """
        Run the full test on an entropy sample.
        
        This orchestrates the test workflow:
        1. Prepare the data
        2. Execute the test algorithm
        3. Calculate the quality score
        4. Interpret the results
        5. Package the result into a TestResult object
        
        Args:
            sample: The entropy sample to test
            
        Returns:
            Test result object containing scores and statistics
            
        Raises:
            Exception: If any part of the test fails
        """
        import time
        start_time = time.time()
        
        self._raw_data = sample
        
        try:
            # Step 1: Prepare the data
            processed_data = self.prepare_data(sample)
            
            # Step 2: Execute the test algorithm
            test_statistics = self._execute_test(processed_data)
            
            # Step 3: Calculate quality score
            score = self.calculate_score(test_statistics)
            
            # Step 4: Determine pass/fail
            passed = self.interpret_results(test_statistics)
            
            # Calculate p-value if available
            p_value = test_statistics.get("p_value")
            
            # Step 5: Package the result
            self._result = TestResult(
                test_id=self.id,
                test_name=self.name,
                status="passed" if passed else "failed",
                p_value=p_value,
                score=score,
                passed=passed,
                statistics=test_statistics,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
            return self._result
        
        except Exception as e:
            # If test execution fails, return an error result
            self._result = TestResult(
                test_id=self.id,
                test_name=self.name,
                status="error",
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
            return self._result
    
    def get_result(self) -> Optional[TestResult]:
        """
        Get the last test result.
        
        Returns:
            The most recent test result, or None if run_test hasn't been called yet
        """
        return self._result
