"""Envelope extraction for CI processing."""

import numpy as np
from scipy.signal import butter, sosfilt


def extract_envelope(signal, sample_rate, cutoff_freq=400, filter_order=2):
    """Extract envelope via half-wave rectification + lowpass filtering."""
    input_1d = signal.ndim == 1
    if input_1d:
        signal = signal.reshape(1, -1)

    n_channels, n_samples = signal.shape
    envelopes = np.zeros_like(signal)

    nyquist = sample_rate / 2
    cutoff_norm = min(cutoff_freq / nyquist, 0.99)
    sos = butter(filter_order, cutoff_norm, btype="low", output="sos")

    for i in range(n_channels):
        rectified = np.maximum(signal[i, :], 0)
        envelopes[i, :] = sosfilt(sos, rectified)

    if input_1d:
        return envelopes.flatten()
    return envelopes


def extract_envelope_hilbert(signal):
    """Extract envelope using Hilbert transform."""
    from scipy.signal import hilbert

    input_1d = signal.ndim == 1
    if input_1d:
        signal = signal.reshape(1, -1)

    n_channels, n_samples = signal.shape
    envelopes = np.zeros_like(signal)

    for i in range(n_channels):
        analytic = hilbert(signal[i, :])
        envelopes[i, :] = np.abs(analytic)

    if input_1d:
        return envelopes.flatten()
    return envelopes


def extract_envelope_rms(signal, sample_rate, window_ms=10):
    """Extract envelope using RMS in sliding windows."""
    input_1d = signal.ndim == 1
    if input_1d:
        signal = signal.reshape(1, -1)

    n_channels, n_samples = signal.shape
    window_samples = int(sample_rate * window_ms / 1000)
    if window_samples % 2 == 0:
        window_samples += 1

    envelopes = np.zeros_like(signal)

    for i in range(n_channels):
        squared = signal[i, :] ** 2
        window = np.ones(window_samples) / window_samples
        rms_squared = np.convolve(squared, window, mode="same")
        envelopes[i, :] = np.sqrt(np.maximum(rms_squared, 0))

    if input_1d:
        return envelopes.flatten()
    return envelopes


class EnvelopeExtractor:
    """Configurable envelope extractor."""

    def __init__(self, sample_rate, method="rectify", cutoff_freq=400, window_ms=10, filter_order=2):
        self.sample_rate = sample_rate
        self.method = method
        self.cutoff_freq = cutoff_freq
        self.window_ms = window_ms
        self.filter_order = filter_order

    def extract(self, signal):
        """Extract envelope using configured method."""
        if self.method == "rectify":
            return extract_envelope(signal, self.sample_rate, self.cutoff_freq, self.filter_order)
        elif self.method == "hilbert":
            return extract_envelope_hilbert(signal)
        elif self.method == "rms":
            return extract_envelope_rms(signal, self.sample_rate, self.window_ms)
        else:
            raise ValueError(f"Unknown method: {self.method}")
