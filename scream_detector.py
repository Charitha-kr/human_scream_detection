import numpy as np
from scipy import signal

class ScreamDetector:
    def __init__(self):
        self.threshold = 0.7
        self.min_frequency = 800
        self.max_frequency = 8000

    def detect_scream(self, audio_data, sample_rate):
        frequencies, times, spectrogram = signal.spectrogram(
            audio_data,
            fs=sample_rate,
            nperseg=1024,
            noverlap=512
        )

        mask = (frequencies >= self.min_frequency) & (frequencies <= self.max_frequency)
        relevant_spectrogram = spectrogram[mask]

        energy = np.mean(relevant_spectrogram)

        return energy > self.threshold, energy