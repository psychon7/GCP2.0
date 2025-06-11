"""
Implementation of the Approximate Entropy (ApEn) Test.

Approximate Entropy measures the frequency of patterns overlapping with other patterns
of similar length. The test compares the frequency of overlapping blocks of length m
with the frequency of overlapping blocks of length m+1.
"""

import math
import numpy as np
from typing import Any, Dict, List, Union

from ..framework.statistical_test import StatisticalTest
from ..framework.data_types import EntropySample


class ApproximateEntropyTest(StatisticalTest):
    """
    Approximate Entropy (ApEn) Test.
    
    Tests the frequency of overlapping patterns, which provides a measure of the
    predictability or the randomness of a sequence.
    """
    
    @property
    def name(self) -> str:
        return "Approximate Entropy (ApEn) Test"
    
    @property
    def description(self) -> str:
        return (
            "The Approximate Entropy test measures the frequency of overlapping patterns "
            "of different lengths. It compares the frequency of m-bit patterns with the "
            "frequency of (m+1)-bit patterns to detect predictable patterns in the sequence. "
            "Random sequences have high approximate entropy, while predictable sequences "
            "have lower approximate entropy."
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
    
    def _compute_freq_of_patterns(self, data: np.ndarray, m: int) -> Dict[int, float]:
        """
        Compute the frequency of each m-bit pattern in the data.
        
        Args:
            data: The binary sequence
            m: Length of patterns to find
            
        Returns:
            Dictionary mapping pattern values to their frequencies
        """
        n = len(data)
        
        # Create overlapping m-bit windows of the data with wrap-around
        # Concatenate data with itself to handle wrap-around
        padded_data = np.concatenate([data, data[:m-1]])
        
        pattern_counts = {}
        max_pattern = 2 ** m
        
        # For each position, compute the pattern value and count it
        for i in range(n):
            # Extract the m-bit pattern starting at position i
            pattern = padded_data[i:i+m]
            
            # Convert the pattern to an integer (bitwise)
            value = 0
            for j in range(m):
                value += pattern[j] * (2 ** (m - 1 - j))
                
            # Count this pattern
            if value in pattern_counts:
                pattern_counts[value] += 1
            else:
                pattern_counts[value] = 1
        
        # Convert counts to frequencies
        pattern_freqs = {k: v / n for k, v in pattern_counts.items()}
        
        # Ensure all possible patterns are represented (even those with zero counts)
        for i in range(max_pattern):
            if i not in pattern_freqs:
                pattern_freqs[i] = 0.0
                
        return pattern_freqs
    
    def _compute_phi(self, pattern_freqs: Dict[int, float]) -> float:
        """
        Compute the Phi value from pattern frequencies.
        
        Phi is the sum of the logarithms of pattern frequencies.
        
        Args:
            pattern_freqs: Dictionary mapping pattern values to frequencies
            
        Returns:
            Phi value
        """
        phi = 0.0
        
        for freq in pattern_freqs.values():
            if freq > 0:  # Avoid log(0)
                phi += freq * np.log(freq)
                
        return phi
    
    def _execute_test(self, processed_data: np.ndarray) -> Dict[str, Any]:
        """
        Execute the Approximate Entropy Test on the processed data.
        
        Args:
            processed_data: Numpy array of 0s and 1s
            
        Returns:
            Dictionary with test statistics
        """
        n = len(processed_data)
        
        # Get block size m from configuration, default to 2
        m = self.config.parameters.get("block_size", 2)
        
        # Ensure we have enough bits for a meaningful test
        if n < 100:  # Need a reasonably large sample
            raise ValueError(
                f"Sample too small for approximate entropy test. "
                f"Need at least 100 bits, but got {n}."
            )
        
        # Calculate pattern frequencies for lengths m and m+1
        freq_m = self._compute_freq_of_patterns(processed_data, m)
        freq_m_plus_1 = self._compute_freq_of_patterns(processed_data, m + 1)
        
        # Calculate phi values
        phi_m = self._compute_phi(freq_m)
        phi_m_plus_1 = self._compute_phi(freq_m_plus_1)
        
        # Calculate ApEn
        ap_en = phi_m - phi_m_plus_1
        
        # Calculate chi-squared statistic
        # chi_squared = 2 * n * (ln(2) - ap_en)
        chi_squared = 2 * n * (np.log(2) - ap_en)
        
        # For ApEn, the test statistic follows a chi-squared distribution
        # with 2^m degrees of freedom
        degrees_of_freedom = 2 ** m
        
        # Calculate p-value
        # For a one-sided test, p-value = 1 - CDF(chi_squared, df)
        from scipy import stats
        p_value = 1.0 - stats.chi2.cdf(chi_squared, degrees_of_freedom)
        
        return {
            "n": n,
            "m": m,
            "ap_en": float(ap_en),
            "phi_m": float(phi_m),
            "phi_m_plus_1": float(phi_m_plus_1),
            "chi_squared": float(chi_squared),
            "degrees_of_freedom": degrees_of_freedom,
            "p_value": float(p_value)
        }
    
    def calculate_score(self, test_statistics: Dict[str, Any]) -> float:
        """
        Calculate a quality score from the test statistics.
        
        The score is based on the p-value and the approximate entropy value.
        Higher ApEn indicates better randomness, but we also want the p-value
        to be high enough to not reject the null hypothesis.
        
        Args:
            test_statistics: Statistics from the approximate entropy test
            
        Returns:
            Quality score from 0-100
        """
        p_value = test_statistics["p_value"]
        ap_en = test_statistics["ap_en"]
        
        # A higher ApEn value is better (more entropy/randomness)
        # The theoretical maximum ApEn for a binary sequence approaches ln(2) ≈ 0.693
        # Use this to scale part of the score
        max_ap_en = np.log(2)
        ap_en_score = min(50.0 * (ap_en / max_ap_en), 50.0)
        
        # The p-value component (similar to other tests)
        if p_value < 0.001:  # Extremely non-random
            p_value_score = 0.0
        elif p_value < 0.01:  # Very non-random
            p_value_score = 10.0
        elif p_value < 0.05:  # Somewhat non-random
            p_value_score = 25.0
        elif p_value < 0.1:   # Borderline
            p_value_score = 40.0
        else:  # Acceptably random
            p_value_score = 50.0
            
        # Combine the two components
        return ap_en_score + p_value_score
    
    def interpret_results(self, test_statistics: Dict[str, Any]) -> bool:
        """
        Determine if the test was passed based on the p-value.
        
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
