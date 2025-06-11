"""
Implementation of the NIST Frequency (Monobit) Test.

This test examines the proportion of ones and zeros in a sequence.
The purpose of this test is to determine whether the number of ones and zeros
in a sequence is approximately the same as would be expected from a truly random sequence.
"""

import math
import numpy as np
from scipy import special
from typing import Any, Dict, List, Union

from ..framework.statistical_test import StatisticalTest
from ..framework.data_types import EntropySample


class NistFrequencyTest(StatisticalTest):
    """
    NIST Frequency (Monobit) Test.
    
    Checks if the proportion of 1s in the binary sequence is close to 1/2,
    as would be expected under the assumption of randomness.
    """
    
    @property
    def name(self) -> str:
        return "NIST Frequency (Monobit) Test"
    
    @property
    def description(self) -> str:
        return (
            "This test examines whether the proportion of ones and zeros in the "
            "entropy sample is approximately equal (as would be expected for a "
            "random sequence). It computes the normalized difference between the "
            "number of ones and the number of zeros, then calculates the p-value "
            "using the complementary error function."
        )
    
    def prepare_data(self, sample: EntropySample) -> np.ndarray:
        """
        Convert the entropy sample to a numpy array of -1 and 1.
        
        The test works by summing -1s (for 0 bits) and 1s (for 1 bits).
        
        Args:
            sample: The entropy sample
            
        Returns:
            Numpy array containing -1s and 1s
            
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
        
        # Convert to numpy array and transform to -1/1
        binary_data = np.array(bit_array, dtype=int)
        return 2 * binary_data - 1  # Convert 0->-1 and 1->1
    
    def _execute_test(self, processed_data: np.ndarray) -> Dict[str, Any]:
        """
        Execute the NIST Frequency Test on the processed data.
        
        Args:
            processed_data: Numpy array of -1 and 1 values
            
        Returns:
            Dictionary with test statistics
        """
        n = len(processed_data)
        
        # Sum all the -1/1 values
        s_obs = abs(np.sum(processed_data))
        
        # Calculate the test statistic
        s_obs_normalized = s_obs / np.sqrt(n)
        
        # Calculate the p-value
        p_value = special.erfc(s_obs_normalized / np.sqrt(2))
        
        # Calculate the proportion of ones
        ones_count = np.count_nonzero(processed_data == 1)
        proportion = ones_count / n
        
        return {
            "n": n,
            "s_obs": float(s_obs),
            "s_obs_normalized": float(s_obs_normalized),
            "p_value": float(p_value),
            "proportion_ones": float(proportion),
            "ones_count": int(ones_count),
            "zeros_count": int(n - ones_count)
        }
    
    def calculate_score(self, test_statistics: Dict[str, Any]) -> float:
        """
        Calculate a quality score from the test statistics.
        
        The score is based on how close the p-value is to 0.5,
        which represents the ideal balance between 0s and 1s.
        
        Args:
            test_statistics: Statistics from the frequency test
            
        Returns:
            Quality score from 0-100
        """
        p_value = test_statistics["p_value"]
        
        # For perfect randomness, p-value would be close to 1.0
        # We want to give higher scores for p-values closer to 1.0
        
        # However, we need to account for the case where p-value is very small,
        # which indicates non-randomness
        
        if p_value < 0.001:  # Extremely non-random
            return 0.0
        elif p_value < 0.01:  # Very non-random
            return 20.0
        elif p_value < 0.05:  # Somewhat non-random
            return 50.0
        elif p_value < 0.1:   # Borderline
            return 70.0
        else:  # Acceptably random
            # Scale the score from 80-100 based on p-value
            # The closer to 1.0, the better
            return 80.0 + 20.0 * (p_value - 0.1) / 0.9
    
    def interpret_results(self, test_statistics: Dict[str, Any]) -> bool:
        """
        Determine if the test was passed based on the p-value.
        
        Following standard statistical practice, we use a significance level
        of 0.01 (1%) to determine if the sequence is non-random.
        
        Args:
            test_statistics: The statistics from _execute_test
            
        Returns:
            True if the test is passed (sequence appears random), False otherwise
        """
        p_value = test_statistics["p_value"]
        
        # Use either a configured threshold or the default 0.01
        threshold = self.threshold if self.threshold is not None else 0.01
        
        # The null hypothesis is that the sequence is random
        # If p-value < threshold, reject the null hypothesis (fail the test)
        # If p-value >= threshold, do not reject the null hypothesis (pass the test)
        return p_value >= threshold
