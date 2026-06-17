import numpy as np


class SignalQualityAssessor:
    def __init__(self):
        self.weights = {'snr': 0.4, 'thd': 0.3, 'clipping': 0.3}

    def assess(self, signal):
        if len(signal) < 8:
            return {
                'qualityScore': 0.0,
                'snr': 0.0,
                'thd': 0.0,
                'clippingRatio': 0.0,
                'dynamicRange': 0.0 if len(signal) == 0 else np.max(signal) - np.min(signal),
                'meanAbsSlope': 0.0 if len(signal) <= 1 else np.mean(np.abs(np.diff(signal))),
            }

        snr = self._estimate_snr(signal)
        thd = self._estimate_thd(signal)
        clipping_ratio = self._estimate_clipping(signal)
        dynamic_range = self._dynamic_range(signal)
        slope = self._mean_abs_slope(signal)

        snr_score = min(1.0, max(0.0, snr / 40))
        thd_score = 1.0 - min(1.0, thd)
        clip_score = 1.0 - clipping_ratio

        quality_score = (
            self.weights['snr'] * snr_score +
            self.weights['thd'] * thd_score +
            self.weights['clipping'] * clip_score
        )

        return {
            'qualityScore': quality_score,
            'snr': snr,
            'thd': thd,
            'clippingRatio': clipping_ratio,
            'dynamicRange': dynamic_range,
            'meanAbsSlope': slope,
        }

    def is_usable(self, signal, threshold=0.5):
        result = self.assess(signal)
        return result['qualityScore'] >= threshold

    def _estimate_snr(self, signal):
        spectrum = np.abs(np.fft.rfft(signal))
        n = len(spectrum)
        signal_band = spectrum[:n // 4]
        noise_band = spectrum[3 * n // 4:]
        signal_power = np.mean(signal_band ** 2)
        noise_power = np.mean(noise_band ** 2)
        if noise_power == 0:
            return 40.0
        return 10 * np.log10(signal_power / noise_power)

    def _estimate_thd(self, signal):
        spectrum = np.abs(np.fft.rfft(signal))
        fundamental_idx = np.argmax(spectrum[1:]) + 1
        fundamental_power = spectrum[fundamental_idx] ** 2
        if fundamental_power == 0:
            return 0
        harmonic_power = 0
        for h in range(2, 6):
            idx = fundamental_idx * h
            if idx < len(spectrum):
                harmonic_power += spectrum[idx] ** 2
        return np.sqrt(harmonic_power / fundamental_power)

    def _estimate_clipping(self, signal):
        max_val = np.max(np.abs(signal))
        if max_val == 0:
            return 0
        threshold = 0.99 * max_val
        clipped = np.sum(np.abs(signal) >= threshold)
        return clipped / len(signal)

    def _dynamic_range(self, signal):
        return np.max(signal) - np.min(signal)

    def _mean_abs_slope(self, signal):
        return np.mean(np.abs(np.diff(signal)))
