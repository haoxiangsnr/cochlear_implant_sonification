"""Cochlear Implant Sonification"""

from src.channel_selection import select_channels_n_of_m
from src.compression import compress_amplitude
from src.envelope import extract_envelope
from src.filterbank import CochlearFilterbank
from src.vocoder import CochlearVocoder, vocode_noise

__all__ = [
    "CochlearFilterbank",
    "extract_envelope",
    "select_channels_n_of_m",
    "compress_amplitude",
    "vocode_noise",
    "CochlearVocoder",
]
