"""
Frequency compounding -related algorithms.
"""
import numpy as np
from arrus.utils.imaging import Operation
import cupy as cp



class FrequencyCompound(Operation):

    def __init__(self, bands):
        self.bands = bands

    def prepare(self, const_metadata):
        fs = const_metadata.data_description.sampling_frequency
        n_samples = const_metadata.input_shape[-1]
        f = cp.fft.fftfreq(n_samples, d = 1/fs)
        self.masks = []
        for f_low, f_high in self.bands:
            mask = cp.logical_and(f >= f_low, f <= f_high)
            mask = mask.astype(cp.float32)
            self.masks.append(mask)

        return const_metadata

    def process(self, data):
        parts = []
        data_f = cp.fft.fft(data, axis=-1)
        for mask in self.masks:
            p = data_f * mask
            p = cp.fft.ifft(p, axis=-1)
            parts.append(p)
        parts = cp.stack(parts)
        return cp.mean(cp.abs(parts), axis=0)




