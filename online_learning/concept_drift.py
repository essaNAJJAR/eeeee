import numpy as np


class ConceptDriftDetector:
    def __init__(self, window_size=100, threshold=2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = []
        self.current_window = []
        self.reference_stats = None

    def update(self, new_data):
        values = np.mean(new_data, axis=0) if new_data.ndim > 1 else new_data
        self.current_window.extend(values.flatten()[:10])

        if len(self.current_window) > self.window_size:
            self.current_window = self.current_window[-self.window_size:]

        if len(self.current_window) >= self.window_size and self.reference_stats is None:
            self.reference_stats = {
                'mean': np.mean(self.current_window),
                'std': np.std(self.current_window),
                'median': np.median(self.current_window),
            }
            self.reference_window = list(self.current_window)

    def detect(self):
        if self.reference_stats is None or len(self.current_window) < self.window_size // 2:
            return False

        current_mean = np.mean(self.current_window)
        current_std = np.std(self.current_window)

        ref_mean = self.reference_stats['mean']
        ref_std = self.reference_stats['std']

        if ref_std == 0:
            return False

        z_score = abs(current_mean - ref_mean) / (ref_std + 1e-12)

        if z_score > self.threshold:
            self.reference_stats = {
                'mean': current_mean,
                'std': current_std,
                'median': np.median(self.current_window),
            }
            self.reference_window = list(self.current_window)
            return True

        return False
