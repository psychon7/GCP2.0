"""
Implementation of the NIST Runs Test.

The Runs test examines the total number of runs in a sequence, where a run is defined as
an uninterrupted sequence of identical bits (either all 1's or all 0's). 
The test checks if the oscillation between 0's and 1's is too fast or too slow.
"""

import math
import numpy as np
from scipy import special
from typing import Any, Dict, List, Union

from ..framework.statistical_test import StatisticalTest
from ..framework.data_types import EntropySample


class NistRunsTest(StatisticalTest):
    """
    NIST Runs Test.
    
    Checks if the oscillation between 0's and 1's in a sequence appears random.
    Too many runs indicates the oscillation is too fast; too few runs indicates too slow.
    """
    
    @property
    def name(self) -> str:
        return "NIST Runs Test"
    
    @property
    def description(self) -> str:
        return (
            "This test examines the total number of runs in the sequence, where a run "
            "is an uninterrupted sequence of identical bits. The test determines if "
            "the oscillation between 0's and 1's is too fast or too slow compared to "
            "what's expected for a random sequence."
        )
    
    def prepare_data(self, sample: EntropySample) -> np.ndarray:
        """
        Convert the entropy sample to a numpy array of 0s and 1s.
        
        Args:
            sample: The entropy sample
            
        Returns:
            Numpy array containing 0s and 1s
            
        Raises:
            ValueError: If the data cannot be converted to a binary sequence
        """
        if isinstance(sample.data, bytes):
            # Convert bytes to binary and then to array of ints
            bit_array = []
            for byte in sample.data:
                # For each byte, extract bits and append to array
                for i in range(8):
                    bit = (byte >> i) & 1
                    bit_array.append(bit)
        elif isinstance(sample.data, str):
            if sample.data_format == "binary":
                # Assume string of '0's and '1's
                bit_array = [int(bit) for bit in sample.data if bit in "01"]
            elif sample.data_format == "hex":
                # Convert hex string to binary
                bit_array = []
                for hex_char in sample.data:
                    if hex_char in "0123456789abcdefABCDEF":
                        value = int(hex_char, 16)
                        for i in range(4):  # 4 bits per hex character
                            bit = (value >> i) & 1
                            bit_array.append(bit)
            else:
                raise ValueError(f"Unsupported string data format: {sample.data_format}")
        elif isinstance(sample.data, List):
            # Already a list of integers, assume binary (0s and 1s)
            bit_array = [int(bit) for bit in sample.data]
        else:
            raise ValueError(f"Unsupported data type: {type(sample.data)}")
        
        # Validate that we have a proper binary sequence
        if not all(bit in [0, 1] for bit in bit_array):
            raise ValueError("Data contains non-binary values")
        
        return np.array(bit_array, dtype=int)
    
    def _execute_test(self, processed_data: np.ndarray) -> Dict[str, Any]:
        """
        Execute the NIST Runs Test on the processed data.
        
        Args:
            processed_data: Numpy array of 0s and 1s
            
        Returns:
            Dictionary with test statistics
        """
        n = len(processed_data)
        
        # Pre-test: Check if the proportion of ones is close enough to 1/2
        # This is a prerequisite for the runs test
        ones_count = np.count_nonzero(processed_data)
        proportion = ones_count / n
        tau = 2.0 / np.sqrt(n)  # Threshold for the pre-test
        
        # Compute the number of runs
        # A run is a consecutive sequence of identical bits
        # Count the number of transitions from 0 to 1 or 1 to 0
        runs = 1  # Start with 1 for the first run
        for i in range(1, n):
            if processed_data[i] != processed_data[i-1]:
                runs += 1
        
        # Calculate the test statistic
        # Under the null hypothesis of randomness, the number of runs is normally distributed
        p = proportion
        expected_runs = 2 * n * p * (1 - p)
        std_dev = np.sqrt(2 * n * p * (1 - p) * (2 * n * p * (1 - p) - n))
        
        if std_dev == 0:  # Edge case: avoid division by zero
            z_score = 0.0
        else:
            z_score = (runs - expected_runs) / std_dev
        
        # Calculate the p-value
        # Two-tailed test: we reject if there are too many or too few runs
        p_value = 2.0 * (1.0 - special.ndtr(abs(z_score)))
        
        return {
            "n": n,
            "ones_count": int(ones_count),
            "zeros_count": int(n - ones_count),
            "proportion": float(proportion),
            "tau": float(tau),
            "pre_test_passed": abs(proportion - 0.5) < tau,
            "runs": int(runs),
            "expected_runs": float(expected_runs),
            "z_score": float(z_score),
            "p_value": float(p_value)
        }
    
    def calculate_score(self, test_statistics: Dict[str, Any]) -> float:
        """
        Calculate a quality score from the test statistics.
        
        The score is based on the p-value and whether the pre-test passed.
        
        Args:
            test_statistics: Statistics from the runs test
            
        Returns:
            Quality score from 0-100
        """
        # First check if the pre-test passed
        # If not, the sequence doesn't have a proportion near 1/2, 
        # which is a prerequisite for the runs test
        if not test_statistics["pre_test_passed"]:
            return 30.0  # Arbitrary low score for failing the pre-test
        
        p_value = test_statistics["p_value"]
        
        if p_value < 0.001:  # Extremely non-random
            return 0.0
        elif p_value < 0.01:  # Very non-random
            return 30.0
        elif p_value < 0.05:  # Somewhat non-random
            return 60.0
        elif p_value < 0.1:   # Borderline
            return 80.0
        else:  # Acceptably random
            # Scale the score from 90-100 based on p-value
            # The closer to 1.0, the better
            return 90.0 + 10.0 * (p_value - 0.1) / 0.9
    
    def interpret_results(self, test_statistics: Dict[str, Any]) -> bool:
        """
        Determine if the test was passed based on the p-value.
        
        Following standard statistical practice, we use a significance level
        of 0.01 (1%) to determine if the sequence is non-random.
        
        Additionally, check if the pre-test passed.
        
        Args:
            test_statistics: The statistics from _execute_test
            
        Returns:
            True if the test is passed (sequence appears random), False otherwise
        """
        # First check if the pre-test passed
        if not test_statistics["pre_test_passed"]:
            return False
        
        p_value = test_statistics["p_value"]
        
        # Use either a configured threshold or the default 0.01
        threshold = self.threshold if self.threshold is not None else 0.01
        
        # The null hypothesis is that the sequence is random
        # If p-value < threshold, reject the null hypothesis (fail the test)
        # If p-value >= threshold, do not reject the null hypothesis (pass the test)
        return p_value >= threshold
