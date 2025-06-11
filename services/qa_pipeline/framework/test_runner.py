"""
Test runner for executing multiple statistical tests and aggregating results.

This module provides the TestRunner class which orchestrates the execution
of multiple statistical tests on entropy samples and produces an overall quality score.
"""

import importlib
import inspect
import logging
import os
import pkgutil
import time
from typing import Dict, List, Optional, Type, Union

from .data_types import EntropySample, OverallQualityScore, TestConfiguration, TestResult
from .statistical_test import StatisticalTest


logger = logging.getLogger(__name__)


class TestRunner:
    """
    Runner for executing multiple statistical tests on entropy samples.
    
    The TestRunner loads test implementations, configures them, runs them on
    entropy samples, and aggregates the results into an overall quality score.
    """
    
    def __init__(self, test_configs: Optional[List[TestConfiguration]] = None):
        """
        Initialize the test runner.
        
        Args:
            test_configs: Optional list of test configurations to use.
                          If None, tests must be registered manually.
        """
        self.tests: Dict[str, StatisticalTest] = {}
        
        # Register tests from configurations if provided
        if test_configs:
            for config in test_configs:
                if config.enabled:
                    self.register_test_from_config(config)
    
    def register_test(self, test: StatisticalTest) -> None:
        """
        Register a test instance with the runner.
        
        Args:
            test: An instantiated statistical test
        """
        self.tests[test.id] = test
        logger.info(f"Registered test: {test.id} ({test.name})")
    
    def register_test_from_config(self, config: TestConfiguration) -> None:
        """
        Create and register a test instance from a configuration.
        
        Args:
            config: Test configuration
            
        Raises:
            ValueError: If test implementation for the given ID cannot be found
        """
        # Test ID format should be module_name.class_name
        if "." not in config.test_id:
            raise ValueError(f"Invalid test_id format: {config.test_id}. Expected format: 'module.class_name'")
        
        module_path, class_name = config.test_id.rsplit(".", 1)
        
        try:
            # Import the module
            module = importlib.import_module(f"qa_pipeline.tests.{module_path}")
            
            # Find the test class
            for name, obj in inspect.getmembers(module):
                if (name == class_name and inspect.isclass(obj) and 
                    issubclass(obj, StatisticalTest) and obj != StatisticalTest):
                    
                    # Create an instance of the test
                    test_instance = obj(config)
                    self.register_test(test_instance)
                    return
                    
            raise ValueError(f"Test class {class_name} not found in module {module_path}")
            
        except ImportError as e:
            raise ValueError(f"Could not import test module: {module_path}. Error: {e}")
    
    def discover_tests(self, package_path: str = "qa_pipeline.tests") -> List[str]:
        """
        Discover available test classes in the specified package.
        
        This method scans the provided package for modules containing subclasses of
        StatisticalTest, but does not register them.
        
        Args:
            package_path: Dotted import path to the package containing test implementations
            
        Returns:
            List of fully qualified test IDs (module_name.class_name)
        """
        test_ids = []
        
        try:
            package = importlib.import_module(package_path)
            package_dir = os.path.dirname(package.__file__)
            
            for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
                if is_pkg:
                    # Recursively scan subpackages
                    sub_ids = self.discover_tests(f"{package_path}.{module_name}")
                    test_ids.extend(sub_ids)
                else:
                    module = importlib.import_module(f"{package_path}.{module_name}")
                    
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and issubclass(obj, StatisticalTest) and 
                            obj != StatisticalTest):
                            test_ids.append(f"{module_name}.{name}")
        
        except ImportError as e:
            logger.error(f"Error discovering tests in {package_path}: {e}")
            
        return test_ids
    
    def run_all_tests(self, sample: EntropySample) -> OverallQualityScore:
        """
        Run all registered tests on the provided sample.
        
        Args:
            sample: The entropy sample to test
            
        Returns:
            Overall quality score with individual test results
        """
        if not self.tests:
            raise ValueError("No tests registered with the runner")
        
        start_time = time.time()
        logger.info(f"Running {len(self.tests)} tests on sample {sample.sample_id}")
        
        # Execute all tests and collect results
        results: List[TestResult] = []
        for test_id, test in self.tests.items():
            try:
                logger.debug(f"Running test {test_id}")
                result = test.run_test(sample)
                results.append(result)
            except Exception as e:
                logger.error(f"Error running test {test_id}: {e}")
                # Create an error result
                results.append(TestResult(
                    test_id=test_id,
                    test_name=test.name,
                    status="error",
                    error_message=str(e)
                ))
        
        # Calculate overall quality score
        total_score = 0.0
        total_weight = 0.0
        
        # Count tests by status
        passed_tests = 0
        failed_tests = 0
        error_tests = 0
        skipped_tests = 0
        
        for result in results:
            weight = self.tests[result.test_id].config.weight
            
            if result.status == "passed":
                passed_tests += 1
                if result.score is not None:
                    total_score += result.score * weight
                    total_weight += weight
            elif result.status == "failed":
                failed_tests += 1
                if result.score is not None:
                    total_score += result.score * weight
                    total_weight += weight
            elif result.status == "error":
                error_tests += 1
            elif result.status == "skipped":
                skipped_tests += 1
        
        # Calculate weighted average score
        final_score = 0.0
        if total_weight > 0:
            final_score = total_score / total_weight
        
        execution_time = time.time() - start_time
        logger.info(f"Tests completed in {execution_time:.2f}s with score: {final_score:.2f}")
        
        # Create and return the overall quality score
        return OverallQualityScore(
            sample_id=sample.sample_id,
            score=final_score,
            test_results=results,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            skipped_tests=skipped_tests
        )
    
    def run_test_by_id(self, sample: EntropySample, test_id: str) -> TestResult:
        """
        Run a specific test by its ID.
        
        Args:
            sample: The entropy sample to test
            test_id: The ID of the test to run
            
        Returns:
            Test result
            
        Raises:
            ValueError: If the test ID is not registered
        """
        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not registered")
        
        logger.info(f"Running test {test_id} on sample {sample.sample_id}")
        return self.tests[test_id].run_test(sample)
