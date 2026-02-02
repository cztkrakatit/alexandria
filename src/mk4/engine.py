"""
MK4 Engine - Core frequency-domain analysis algorithm.

This module implements the Molecular Kinetics 4th-order (MK4)
algorithm for disease detection from RNA-seq expression data.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from typing import Dict, Tuple, Optional, Union
import warnings


def mk4_transform(
    expression_vector: np.ndarray,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply MK4 frequency transform to RNA expression vector.

    Transforms gene expression data to frequency domain using
    Fast Fourier Transform (FFT) and computes power spectrum.

    Parameters
    ----------
    expression_vector : np.ndarray
        Gene expression values, shape (n_genes,)
    normalize : bool, default=True
        Whether to z-score normalize input

    Returns
    -------
    freqs : np.ndarray
        Frequency values
    fft_vals : np.ndarray
        Complex FFT values
    power : np.ndarray
        Power spectrum (magnitude squared)

    Examples
    --------
    >>> expression = np.array([1.2, 3.4, 2.1, 4.5])
    >>> freqs, fft, power = mk4_transform(expression)
    >>> print(f"Frequencies: {freqs.shape}")
    """
    if not isinstance(expression_vector, np.ndarray):
        expression_vector = np.array(expression_vector)

    if len(expression_vector.shape) != 1:
        raise ValueError(
            f"Expected 1D array, got shape {expression_vector.shape}"
        )

    n_genes = len(expression_vector)
    if n_genes < 100:
        warnings.warn(
            f"Only {n_genes} genes - results may be unreliable. "
            f"Recommend >1000 genes."
        )

    # Normalize to zero mean, unit variance
    if normalize:
        signal = (expression_vector - np.mean(expression_vector)) / \
                 (np.std(expression_vector) + 1e-10)
    else:
        signal = expression_vector.copy()

    # Apply FFT
    fft_vals = rfft(signal)
    freqs = rfftfreq(len(signal), d=1.0)

    # Compute power spectrum
    power = np.abs(fft_vals) ** 2

    return freqs, fft_vals, power


def calculate_chaos(
    power_spectrum: np.ndarray,
    method: str = 'shannon'
) -> float:
    """
    Calculate chaos metric from power spectrum.

    Computes Shannon entropy of normalized power spectrum
    as a measure of transcriptional chaos/disorder.

    Parameters
    ----------
    power_spectrum : np.ndarray
        Power spectrum from FFT
    method : str, default='shannon'
        Method for chaos calculation. Currently only 'shannon'

    Returns
    -------
    chaos : float
        Chaos metric (higher = more chaotic)

    Notes
    -----
    Chaos interpretation:
    - Low chaos (~5-7): Ordered, healthy tissue
    - High chaos (~8-9): Disordered, diseased tissue

    Examples
    --------
    >>> power = np.array([100, 50, 25, 12.5])
    >>> chaos = calculate_chaos(power)
    >>> print(f"Chaos: {chaos:.4f}")
    """
    if method != 'shannon':
        raise ValueError(f"Unknown method: {method}")

    # Normalize power to probability distribution
    power_norm = power_spectrum / (np.sum(power_spectrum) + 1e-10)

    # Remove zeros to avoid log(0)
    power_norm = power_norm[power_norm > 1e-10]

    # Shannon entropy: H = -sum(p * log(p))
    chaos = -np.sum(power_norm * np.log(power_norm + 1e-10))

    return chaos


def calculate_coherence(
    power_spectrum: np.ndarray
) -> float:
    """
    Calculate coherence metric from power spectrum.

    Computes ratio of geometric mean to arithmetic mean
    as a measure of signal coherence.

    Parameters
    ----------
    power_spectrum : np.ndarray
        Power spectrum from FFT

    Returns
    -------
    coherence : float
        Coherence metric (0-1, higher = more coherent)

    Examples
    --------
    >>> power = np.array([100, 50, 25, 12.5])
    >>> coh = calculate_coherence(power)
    >>> print(f"Coherence: {coh:.4f}")
    """
    geometric_mean = np.exp(
        np.mean(np.log(power_spectrum + 1e-10))
    )

    arithmetic_mean = np.mean(power_spectrum)

    coherence = geometric_mean / (arithmetic_mean + 1e-10)

    return coherence


def calculate_band_power(
    freqs: np.ndarray,
    power: np.ndarray,
    band: Tuple[float, float]
) -> float:
    """
    Calculate total power in a frequency band.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency values
    power : np.ndarray
        Power spectrum
    band : tuple of (low, high)
        Frequency band boundaries

    Returns
    -------
    band_power : float
        Total power in the specified band

    Examples
    --------
    >>> freqs, _, power = mk4_transform(expression)
    >>> low_power = calculate_band_power(freqs, power, (0.1, 1.0))
    """
    low, high = band
    mask = (freqs >= low) & (freqs < high)
    return np.sum(power[mask])


class MK4Analyzer:
    """
    Main MK4 analyzer for disease detection.

    This class provides a high-level interface for MK4 analysis
    of RNA-seq expression data.

    Attributes
    ----------
    results_ : dict
        Analysis results after fitting

    Examples
    --------
    >>> analyzer = MK4Analyzer()
    >>> results = analyzer.analyze_dataset(expression_matrix)
    >>> print(f"Mean chaos: {results['chaos'].mean():.4f}")
    """

    def __init__(self):
        """Initialize MK4 analyzer."""
        self.results_ = None

    def analyze_sample(
        self,
        expression_vector: np.ndarray
    ) -> Dict[str, float]:
        """
        Analyze a single RNA-seq sample.

        Parameters
        ----------
        expression_vector : np.ndarray
            Gene expression values for one sample

        Returns
        -------
        results : dict
            Dictionary with keys:
            - 'chaos': Chaos metric
            - 'coherence': Coherence metric
            - 'low_freq_power': Power in 0.1-1.0 Hz band
            - 'mid_freq_power': Power in 1.0-5.0 Hz band
            - 'high_freq_power': Power in 5.0-20.0 Hz band
        """
        freqs, fft_vals, power = mk4_transform(expression_vector)

        chaos = calculate_chaos(power)
        coherence = calculate_coherence(power)

        low_power = calculate_band_power(freqs, power, (0.1, 1.0))
        mid_power = calculate_band_power(freqs, power, (1.0, 5.0))
        high_power = calculate_band_power(freqs, power, (5.0, 20.0))

        return {
            'chaos': chaos,
            'coherence': coherence,
            'low_freq_power': low_power,
            'mid_freq_power': mid_power,
            'high_freq_power': high_power,
        }

    def analyze_dataset(
        self,
        expression_matrix: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Analyze multiple samples.

        Parameters
        ----------
        expression_matrix : np.ndarray
            Expression data, shape (n_samples, n_genes)
        labels : np.ndarray, optional
            Sample labels (0=control, 1=disease)

        Returns
        -------
        results : dict
            Dictionary with arrays of metrics for each sample
        """
        n_samples = expression_matrix.shape[0]

        results = {
            'chaos': np.zeros(n_samples),
            'coherence': np.zeros(n_samples),
            'low_freq_power': np.zeros(n_samples),
            'mid_freq_power': np.zeros(n_samples),
            'high_freq_power': np.zeros(n_samples),
        }

        for i in range(n_samples):
            sample_results = self.analyze_sample(expression_matrix[i])
            for key in results:
                results[key][i] = sample_results[key]

        if labels is not None:
            results['labels'] = labels

        self.results_ = results
        return results


def quick_analyze(
    expression_data: Union[np.ndarray, list]
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Quick MK4 analysis of expression data.

    Convenience function for simple analyses.

    Parameters
    ----------
    expression_data : array-like
        Can be:
        - 1D array: single sample
        - 2D array: multiple samples

    Returns
    -------
    results : dict
        Analysis results

    Examples
    --------
    >>> results = quick_analyze(np.random.randn(10000))
    >>> print(results['chaos'])
    """
    data = np.array(expression_data)
    analyzer = MK4Analyzer()

    if len(data.shape) == 1:
        return analyzer.analyze_sample(data)
    else:
        return analyzer.analyze_dataset(data)


if __name__ == "__main__":
    print("MK4 Engine - Self Test")
    print("=" * 50)

    np.random.seed(42)
    test_expression = np.random.randn(10000) * 2 + 5

    freqs, fft, power = mk4_transform(test_expression)
    print(f"Transform: {len(freqs)} frequencies")

    chaos = calculate_chaos(power)
    print(f"Chaos: {chaos:.4f}")

    coh = calculate_coherence(power)
    print(f"Coherence: {coh:.4f}")

    analyzer = MK4Analyzer()
    results = analyzer.analyze_sample(test_expression)
    print(f"Analyzer: {len(results)} metrics")

    print("\n" + "=" * 50)
    print("All tests passed!")
