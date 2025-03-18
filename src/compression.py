"""Amplitude compression for CI simulation."""

import numpy as np


def compress_logarithmic(signal, base_level=0.0001, compression_ratio=0.3):
    """Apply logarithmic compression to map acoustic to electrical range."""
    signal = np.maximum(signal, 0)
    max_val = np.max(signal)
    if max_val > 0:
        signal_norm = signal / max_val
    else:
        return signal

    compressed = np.log(1 + signal_norm / base_level) / np.log(1 + 1 / base_level)
    return compressed * compression_ratio


def compress_power_law(signal, exponent=0.3):
    """Apply power-law compression (loudness model)."""
    signal = np.maximum(signal, 0)
    max_val = np.max(signal)
    if max_val > 0:
        signal_norm = signal / max_val
    else:
        return signal
    return np.power(signal_norm, exponent)


def compress_amplitude(signal, method="logarithmic", base_level=0.0001, compression_ratio=0.3, exponent=0.3):
    """Apply amplitude compression with configurable method."""
    if method == "logarithmic":
        return compress_logarithmic(signal, base_level, compression_ratio)
    elif method == "power":
        return compress_power_law(signal, exponent)
    else:
        raise ValueError(f"Unknown compression method: {method}")


def map_to_stimulation_levels(compressed_signal, threshold_level=0.1, comfort_level=1.0, n_levels=256):
    """Map compressed amplitude to discrete stimulation levels."""
    dynamic_range = comfort_level - threshold_level
    mapped = threshold_level + compressed_signal * dynamic_range
    quantized = np.round(mapped * (n_levels - 1)) / (n_levels - 1)
    quantized = np.where(compressed_signal > 0.01, quantized, 0)
    return quantized


class AmplitudeCompressor:
    """Configurable amplitude compressor for CI simulation."""

    def __init__(
        self,
        method="logarithmic",
        base_level=0.0001,
        compression_ratio=0.3,
        exponent=0.3,
        quantize=False,
        n_levels=256,
        threshold_level=0.1,
        comfort_level=1.0,
    ):
        self.method = method
        self.base_level = base_level
        self.compression_ratio = compression_ratio
        self.exponent = exponent
        self.quantize = quantize
        self.n_levels = n_levels
        self.threshold_level = threshold_level
        self.comfort_level = comfort_level

    def compress(self, signal):
        """Apply compression to input signal."""
        compressed = compress_amplitude(
            signal,
            method=self.method,
            base_level=self.base_level,
            compression_ratio=self.compression_ratio,
            exponent=self.exponent,
        )

        if self.quantize:
            compressed = map_to_stimulation_levels(
                compressed,
                self.threshold_level,
                self.comfort_level,
                self.n_levels,
            )

        return compressed
