"""Mel-scale filterbank for CI simulation."""

import librosa
import numpy as np
from scipy.signal import butter, sosfilt


def compute_center_frequencies(n_channels, freq_low=200, freq_high=7000):
    """Get mel-spaced center frequencies."""
    mel_freqs = librosa.mel_frequencies(n_mels=n_channels + 2, fmin=freq_low, fmax=freq_high)
    return mel_freqs[1:-1]


def compute_band_edges(n_channels, freq_low=200, freq_high=7000):
    """Get band edges for each channel."""
    mel_freqs = librosa.mel_frequencies(n_mels=n_channels + 2, fmin=freq_low, fmax=freq_high)
    edges = np.zeros((n_channels, 2))
    for i in range(n_channels):
        edges[i, 0] = mel_freqs[i]
        edges[i, 1] = mel_freqs[i + 2]
    return edges


class CochlearFilterbank:
    """Bank of bandpass filters using Mel-scale frequency spacing."""

    def __init__(
        self,
        n_channels=16,
        sample_rate=16000,
        freq_low=200,
        freq_high=7000,
        filter_order=4,
    ):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.filter_order = filter_order

        self.center_freqs = compute_center_frequencies(n_channels, freq_low, freq_high)
        self.band_edges = compute_band_edges(n_channels, freq_low, freq_high)

        # Clamp band edges to valid range
        nyquist = sample_rate / 2
        self.band_edges = np.clip(self.band_edges, 20, nyquist - 10)

        # Design Butterworth bandpass filters
        self.filters = []
        for i in range(n_channels):
            low = self.band_edges[i, 0]
            high = self.band_edges[i, 1]

            low_norm = max(0.001, min(low / nyquist, 0.999))
            high_norm = max(low_norm + 0.001, min(high / nyquist, 0.999))

            sos = butter(filter_order, [low_norm, high_norm], btype="band", output="sos")
            self.filters.append(sos)

    def filter(self, signal):
        """Apply filterbank to input signal."""
        n_samples = len(signal)
        output = np.zeros((self.n_channels, n_samples))

        for i, sos in enumerate(self.filters):
            output[i, :] = sosfilt(sos, signal)

        return output

    def get_channel_info(self):
        """Get center freq and band edges for each channel."""
        info = []
        for i in range(self.n_channels):
            info.append({
                "channel": i + 1,
                "center_freq": self.center_freqs[i],
                "low_edge": self.band_edges[i, 0],
                "high_edge": self.band_edges[i, 1],
            })
        return info
