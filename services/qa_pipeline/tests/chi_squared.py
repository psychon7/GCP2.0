"""
Implementation of the Chi-Squared Test for randomness.

The chi-squared test examines the frequency distribution of n-bit patterns in the
sequence to determine if they occur with the frequencies expected of a random sequence.
"""

import math
import numpy as np
from scipy import stats
from typing import Any, Dict, List, Union

from ..framework.statistical_test import StatisticalTest
from ..framework.data_types import EntropySample


class ChiSquaredTest(StatisticalTest):
    """
    Chi-Squared Test for randomness.
    
    Tests whether n-bit patterns in the entropy data follow a uniform distribution
    as would be expected in a random sequence.
    """
    
    @property
    def name(self) -> str:
        return "Chi-Squared Uniformity Test"
    
    @property
    def description(self) -> str:
        return (
            "This test examines the frequency distribution of n-bit patterns in the "
            "entropy sequence to determine if they occur with the frequencies expected "
            "of a random sequence. The test divides the data into blocks and counts "
            "the occurrences of each possible n-bit pattern, then compares this "
            "distribution to the uniform distribution expected in random data."
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
        Execute the Chi-Squared Test on the processed data.
        
        This implementation allows for different block sizes, configurable through
        the parameters. Default is to test the distribution of bytes (8-bit patterns).
        
        Args:
            processed_data: Numpy array of 0s and 1s
            
        Returns:
            Dictionary with test statistics
        """
        # Get block size from configuration, default to 8 (test bytes)
        block_size = self.config.parameters.get("block_size", 8)
        
        n = len(processed_data)
        
        # Ensure we have enough bits for a meaningful test
        if n < block_size * 5:  # Need at least a few blocks
            raise ValueError(
                f"Sample too small for chi-squared test with block_size={block_size}. "
                f"Need at least {block_size * 5} bits, but got {n}."
            )
        
        # Truncate to a multiple of block_size
        truncated_length = (n // block_size) * block_size
        truncated_data = processed_data[:truncated_length]
        
        # Reshape to get blocks
        blocks = truncated_data.reshape(-1, block_size)
        
        # Convert each block to an integer (its pattern value)
        # Use binary (base-2) representation
        pattern_values = np.zeros(len(blocks), dtype=int)
        for i in range(block_size):
            pattern_values += blocks[:, i] * (2 ** i)
        
        # Count occurrences of each pattern
        pattern_counts = np.zeros(2 ** block_size, dtype=int)
        for value in pattern_values:
            pattern_counts[value] += 1
        
        # Calculate expected counts (uniform distribution)
        expected_count = len(blocks) / (2 ** block_size)
        expected_counts = np.full(2 ** block_size, expected_count)
        
        # Calculate chi-squared statistic
        # The formula is: sum((observed - expected)^2 / expected)
        chi_squared = np.sum((pattern_counts - expected_counts) ** 2 / expected_counts)
        
        # Degrees of freedom is (number of categories - 1)
        degrees_of_freedom = (2 ** block_size) - 1
        
        # Calculate p-value using the chi-squared distribution
        p_value = 1.0 - stats.chi2.cdf(chi_squared, degrees_of_freedom)
        
        # Create a simplified histogram for the most extreme deviations
        # Find patterns with the largest deviations from expected
        deviations = abs(pattern_counts - expected_counts) / expected_counts
        sorted_indices = np.argsort(deviations)[::-1]  # Sort in descending order
        top_deviations = {}
        for i in range(min(10, len(sorted_indices))):  # Show top 10 deviations
            pattern = sorted_indices[i]
            top_deviations[str(pattern)] = {
                "observed": int(pattern_counts[pattern]),
                "expected": float(expected_counts[pattern]),
                "deviation": float(deviations[pattern])
            }
        
        return {
            "n": n,
            "block_size": block_size,
            "num_blocks": len(blocks),
            "chi_squared": float(chi_squared),
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": float(p_value),
            "expected_count_per_pattern": float(expected_count),
            "top_deviations": top_deviations
        }
    
    def calculate_score(self, test_statistics: Dict[str, Any]) -> float:
        """
        Calculate a quality score from the test statistics.
        
        For the chi-squared test, extreme p-values in either direction are concerning.
        A p-value very close to 0 means the distribution is too uneven.
        A p-value very close to 1 means the distribution is too even (which is also suspicious).
        
        Args:
            test_statistics: Statistics from the chi-squared test
            
        Returns:
            Quality score from 0-100
        """
        p_value = test_statistics["p_value"]
        
        # For chi-squared, extreme values in either direction are concerning
        # The ideal p-value distribution for truly random data should be uniform on [0, 1]
        
        if p_value < 0.001 or p_value > 0.999:
            # Extremely non-random (either too uneven or too even)
            return 0.0
        elif p_value < 0.01 or p_value > 0.99:
            # Very non-random
            return 30.0
        elif p_value < 0.05 or p_value > 0.95:
            # Somewhat non-random
            return 60.0
        else:
            # Acceptably random - score highest at p-value = 0.5
            # Calculate distance from 0.5 and convert to a score
            distance_from_ideal = abs(p_value - 0.5) / 0.5  # Normalized to [0, 1]
            return 100.0 - (distance_from_ideal * 40.0)  # Map to [60, 100]
    
    def interpret_results(self, test_statistics: Dict[str, Any]) -> bool:
        """
        Determine if the test was passed based on the p-value.
        
        For chi-squared, we need to check both tails, as both extremely low and
        extremely high p-values indicate non-randomness.
        
        Args:
            test_statistics: The statistics from _execute_test
            
        Returns:
            True if the test is passed (sequence appears random), False otherwise
        """
        p_value = test_statistics["p_value"]
        
        # Use either a configured threshold or the default 0.01
        threshold = self.threshold if self.threshold is not None else 0.01
        
        # For chi-squared test, check both tails
        # Reject if p_value < threshold (too uneven)
        # Reject if p_value > (1 - threshold) (too even)
        return threshold <= p_value <= (1.0 - threshold)
