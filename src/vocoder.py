"""Noise-band vocoder for CI simulation."""

import numpy as np
from scipy.signal import butter, sosfilt


def generate_noise_carriers(n_channels, n_samples, band_edges, sample_rate):
    """Generate bandpass-filtered noise carriers for each channel."""
    carriers = np.zeros((n_channels, n_samples))
    nyquist = sample_rate / 2

    for i in range(n_channels):
        noise = np.random.randn(n_samples)
        low = band_edges[i, 0]
        high = band_edges[i, 1]

        low_norm = max(0.001, min(low / nyquist, 0.999))
        high_norm = max(low_norm + 0.001, min(high / nyquist, 0.999))

        sos = butter(4, [low_norm, high_norm], btype="band", output="sos")
        carriers[i, :] = sosfilt(sos, noise)
        carriers[i, :] /= np.max(np.abs(carriers[i, :])) + 1e-10

    return carriers


def vocode_noise(envelopes, band_edges, sample_rate):
    """Create noise-vocoded audio from envelopes."""
    n_channels, n_samples = envelopes.shape

    carriers = generate_noise_carriers(n_channels, n_samples, band_edges, sample_rate)
    modulated = carriers * envelopes
    output = np.sum(modulated, axis=0)

    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.9

    return output


class CochlearVocoder:
    """Complete CI vocoder: filterbank -> envelope -> n-of-m -> compression -> synthesis."""

    def __init__(
        self,
        n_channels=16,
        sample_rate=16000,
        freq_low=200,
        freq_high=7000,
        n_select=8,
        envelope_cutoff=400,
        compression_ratio=0.3,
    ):
        from src.channel_selection import ChannelSelector
        from src.compression import AmplitudeCompressor
        from src.envelope import EnvelopeExtractor
        from src.filterbank import CochlearFilterbank

        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.n_select = n_select
        self.envelope_cutoff = envelope_cutoff

        self.filterbank = CochlearFilterbank(
            n_channels=n_channels,
            sample_rate=sample_rate,
            freq_low=freq_low,
            freq_high=freq_high,
        )

        self.envelope_extractor = EnvelopeExtractor(
            sample_rate=sample_rate,
            method="rectify",
            cutoff_freq=envelope_cutoff,
        )

        self.channel_selector = ChannelSelector(
            n_select=n_select,
            n_total=n_channels,
        )

        self.compressor = AmplitudeCompressor(
            method="logarithmic",
            compression_ratio=compression_ratio,
        )

    def process(self, signal, return_intermediate=False):
        """Process audio through the CI pipeline."""
        filtered = self.filterbank.filter(signal)
        envelopes = self.envelope_extractor.extract(filtered)

        selection_result = self.channel_selector.select(envelopes, self.sample_rate)
        selected_envelopes = selection_result["selected_envelopes"]

        compressed = self.compressor.compress(selected_envelopes)

        vocoded = vocode_noise(
            compressed,
            self.filterbank.band_edges,
            self.sample_rate,
        )

        if return_intermediate:
            return {
                "input": signal,
                "filtered": filtered,
                "envelopes": envelopes,
                "selection_mask": selection_result["mask"],
                "channel_energy": selection_result["energy"],
                "frame_times": selection_result["frame_times"],
                "selected_envelopes": selected_envelopes,
                "compressed": compressed,
                "vocoded": vocoded,
                "center_freqs": self.filterbank.center_freqs,
                "band_edges": self.filterbank.band_edges,
            }

        return vocoded
