"""N-of-M channel selection strategy."""

import numpy as np


def compute_frame_energy(envelopes, sample_rate, frame_ms=8, hop_ms=2):
    """Compute energy for each channel in sliding time frames."""
    n_channels, n_samples = envelopes.shape
    frame_samples = int(sample_rate * frame_ms / 1000)
    hop_samples = int(sample_rate * hop_ms / 1000)

    n_frames = (n_samples - frame_samples) // hop_samples + 1
    if n_frames <= 0:
        n_frames = 1
        frame_samples = n_samples

    energy = np.zeros((n_channels, n_frames))
    frame_times = np.zeros(n_frames)

    for f in range(n_frames):
        start = f * hop_samples
        end = start + frame_samples
        if end > n_samples:
            end = n_samples

        frame_data = envelopes[:, start:end]
        energy[:, f] = np.sqrt(np.mean(frame_data**2, axis=1))
        frame_times[f] = (start + end) / 2 / sample_rate

    return energy, frame_times


def select_channels_n_of_m(envelopes, n_select, sample_rate, frame_ms=8, hop_ms=2):
    """Select n channels with highest energy per frame."""
    n_channels, n_samples = envelopes.shape
    n_select = max(1, min(n_select, n_channels))

    energy, frame_times = compute_frame_energy(envelopes, sample_rate, frame_ms, hop_ms)
    n_frames = energy.shape[1]
    selected_mask = np.zeros((n_channels, n_frames), dtype=bool)

    for f in range(n_frames):
        frame_energy = energy[:, f]
        top_indices = np.argsort(frame_energy)[-n_select:]
        selected_mask[top_indices, f] = True

    return selected_mask, energy, frame_times


def apply_channel_selection(envelopes, selected_mask, sample_rate, frame_ms=8, hop_ms=2):
    """Apply channel selection mask to envelopes."""
    n_channels, n_samples = envelopes.shape
    n_frames = selected_mask.shape[1]
    hop_samples = int(sample_rate * hop_ms / 1000)
    frame_samples = int(sample_rate * frame_ms / 1000)

    sample_mask = np.zeros_like(envelopes, dtype=bool)

    for f in range(n_frames):
        start = f * hop_samples
        end = start + frame_samples
        if end > n_samples:
            end = n_samples

        for ch in range(n_channels):
            if selected_mask[ch, f]:
                sample_mask[ch, start:end] = True

    return envelopes * sample_mask


class ChannelSelector:
    """N-of-M channel selector (ACE strategy)."""

    def __init__(self, n_select=8, n_total=22, frame_ms=8, hop_ms=2):
        self.n_select = n_select
        self.n_total = n_total
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms

    def select(self, envelopes, sample_rate):
        """Apply n-of-m selection to envelopes."""
        selected_mask, energy, frame_times = select_channels_n_of_m(
            envelopes, self.n_select, sample_rate, self.frame_ms, self.hop_ms
        )

        selected_envelopes = apply_channel_selection(
            envelopes, selected_mask, sample_rate, self.frame_ms, self.hop_ms
        )

        return {
            "selected_envelopes": selected_envelopes,
            "mask": selected_mask,
            "energy": energy,
            "frame_times": frame_times,
        }
