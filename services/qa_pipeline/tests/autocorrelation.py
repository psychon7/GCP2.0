"""
Implementation of the Autocorrelation Test.

The autocorrelation test examines the correlation between a sequence and a shifted version of itself.
This helps detect periodic patterns or dependencies between bits at different positions.
"""

import numpy as np
from typing import Any, Dict, List, Union

from ..framework.statistical_test import StatisticalTest
from ..framework.data_types import EntropySample


class AutocorrelationTest(StatisticalTest):
    """
    Autocorrelation Test for randomness.
    
    Tests for correlations between a sequence and shifted versions of itself,
    which could indicate periodic patterns or dependencies between bits.
    """
    
    @property
    def name(self) -> str:
        return "Binary Autocorrelation Test"
    
    @property
    def description(self) -> str:
        return (
            "This test examines the correlation between a binary sequence and "
            "a shifted version of itself. For random sequences, the autocorrelation "
            "should be close to zero for all non-zero shift values. Significant "
            "autocorrelation indicates repeating patterns or dependencies between bits."
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
    
    def _calculate_autocorrelation(self, data: np.ndarray, lag: int) -> float:
        """
        Calculate the autocorrelation for a specific lag.
        
        Args:
            data: Binary data array
            lag: Lag value (number of positions to shift)
            
        Returns:
            Autocorrelation value (-1 to 1)
        """
        # Convert 0/1 to -1/1 for proper correlation calculation
        normalized_data = 2 * data - 1
        
        n = len(normalized_data)
        
        # Calculate the dot product between original and shifted sequence
        # Only consider the overlapping part
        shifted_data = np.roll(normalized_data, lag)
        
        # For non-circular autocorrelation, we should exclude the wrapped-around values
        # This would require truncating both arrays by 'lag' positions
        # However, for large sequences compared to the lag, this makes minimal difference
        # For simplicity and efficiency, we'll use the numpy's roll function without truncation
        
        # Calculate autocorrelation
        autocorr = np.sum(normalized_data * shifted_data) / n
        
        return autocorr
    
    def _execute_test(self, processed_data: np.ndarray) -> Dict[str, Any]:
        """
        Execute the Autocorrelation Test on the processed data.
        
        Calculates autocorrelation for multiple lags and performs a statistical test
        to determine if there are significant correlations.
        
        Args:
            processed_data: Numpy array of 0s and 1s
            
        Returns:
            Dictionary with test statistics
        """
        n = len(processed_data)
        
        # Get the maximum lag to test from configuration, default to min(50, n//4)
        max_lag = self.config.parameters.get("max_lag", min(50, n // 4))
        
        # Ensure the lag isn't too large relative to the sample size
        max_lag = min(max_lag, n // 4)
        
        # Initialize arrays to store results
        lags = list(range(1, max_lag + 1))
        autocorrelations = []
        p_values = []
        significant_lags = []
        
        # Calculate autocorrelation for each lag and test significance
        for lag in lags:
            # Calculate autocorrelation
            autocorr = self._calculate_autocorrelation(processed_data, lag)
            autocorrelations.append(autocorr)
            
            # For random data, autocorr should be approximately normally distributed
            # with mean 0 and standard deviation 1/sqrt(n)
            std_dev = 1.0 / np.sqrt(n)
            
            # Calculate z-score
            z_score = autocorr / std_dev
            
            # Calculate two-sided p-value from standard normal distribution
            # We're interested in whether the autocorrelation is significantly different from 0
            # in either direction (positive or negative correlation)
            from scipy import stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            p_values.append(p_value)
            
            # Bonferroni correction for multiple testing
            # We're testing 'max_lag' different lags, so adjust the significance level
            bonferroni_threshold = 0.01 / max_lag
            
            # If p-value is less than the threshold, the autocorrelation is significant
            if p_value < bonferroni_threshold:
                significant_lags.append({
                    "lag": lag,
                    "autocorr": float(autocorr),
                    "z_score": float(z_score),
                    "p_value": float(p_value)
                })
        
        # Find the lag with the maximum absolute autocorrelation
        abs_autocorr = np.abs(autocorrelations)
        max_autocorr_index = np.argmax(abs_autocorr)
        max_autocorr_lag = lags[max_autocorr_index]
        max_autocorr_value = autocorrelations[max_autocorr_index]
        max_autocorr_p_value = p_values[max_autocorr_index]
        
        # Overall p-value for the test is the minimum p-value after Bonferroni correction
        if p_values:
            overall_p_value = min(p_values) * max_lag  # Bonferroni correction
            overall_p_value = min(1.0, overall_p_value)  # Cap at 1.0
        else:
            overall_p_value = 1.0
            
        return {
            "n": n,
            "max_lag": max_lag,
            "autocorrelations": [float(ac) for ac in autocorrelations],
            "max_autocorr_lag": max_autocorr_lag,
            "max_autocorr_value": float(max_autocorr_value),
            "max_autocorr_p_value": float(max_autocorr_p_value),
            "significant_lags": significant_lags,
            "num_significant_lags": len(significant_lags),
            "p_value": float(overall_p_value)
        }
    
    def calculate_score(self, test_statistics: Dict[str, Any]) -> float:
        """
        Calculate a quality score from the test statistics.
        
        The score is based on the number of significant autocorrelations found
        and the strength of the maximum autocorrelation.
        
        Args:
            test_statistics: Statistics from the autocorrelation test
            
        Returns:
            Quality score from 0-100
        """
        # Extract relevant statistics
        num_significant_lags = test_statistics["num_significant_lags"]
        max_autocorr_value = abs(test_statistics["max_autocorr_value"])
        p_value = test_statistics["p_value"]
        
        # Base score on p-value
        if p_value < 0.001:  # Extremely non-random
            base_score = 0.0
        elif p_value < 0.01:  # Very non-random
            base_score = 30.0
        elif p_value < 0.05:  # Somewhat non-random
            base_score = 60.0
        elif p_value < 0.1:   # Borderline
            base_score = 80.0
        else:  # Acceptably random
            base_score = 100.0
        
        # Adjust score based on number of significant lags
        # Each significant lag reduces the score
        lag_penalty = min(30.0, 10.0 * num_significant_lags)
        
        # Adjust score based on maximum autocorrelation strength
        # Larger absolute autocorrelation values indicate stronger patterns
        # For truly random data, autocorrelation should be near zero
        autocorr_penalty = min(30.0, 100.0 * max_autocorr_value)
        
        # Calculate final score
        final_score = max(0.0, base_score - lag_penalty - autocorr_penalty)
        
        return final_score
    
    def interpret_results(self, test_statistics: Dict[str, Any]) -> bool:
        """
        Determine if the test was passed based on the test statistics.
        
        Args:
            test_statistics: The statistics from _execute_test
            
        Returns:
            True if the test is passed (sequence appears random), False otherwise
        """
        # For autocorrelation test, we care about:
        # 1. The overall p-value (after Bonferroni correction)
        # 2. The number of significant autocorrelations
        
        p_value = test_statistics["p_value"]
        num_significant_lags = test_statistics["num_significant_lags"]
        
        # Use either a configured threshold or the default 0.01
        threshold = self.threshold if self.threshold is not None else 0.01
        
        # Pass the test if:
        # 1. p-value is above the threshold (no significant autocorrelation after correction)
        # 2. There are no individually significant lags
        return p_value >= threshold and num_significant_lags == 0
