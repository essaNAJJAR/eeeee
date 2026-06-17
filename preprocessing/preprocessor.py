import numpy as np
from scipy.signal import savgol_filter


def zscore_outlier_removal(signal, threshold=3):
    if len(signal) == 0:
        return signal.copy(), np.zeros(0, dtype=bool)

    mu = np.mean(signal)
    sigma = np.std(signal)
    if sigma == 0:
        return signal.copy(), np.zeros(len(signal), dtype=bool)

    z_scores = (signal - mu) / sigma
    outlier_mask = np.abs(z_scores) > threshold

    clean_signal = signal.copy()
    window_size = 10
    half_win = window_size // 2

    outlier_indices = np.where(outlier_mask)[0]
    for i in outlier_indices:
        start_idx = max(0, i - half_win)
        end_idx = min(len(signal), i + half_win + 1)
        clean_signal[i] = np.median(signal[start_idx:end_idx])

    return clean_signal, outlier_mask


def savitzky_golay_smooth(signal, frame_length=11, poly_order=3):
    if len(signal) == 0:
        return signal.copy()

    if len(signal) < frame_length:
        frame_length = len(signal)
        if frame_length % 2 == 0:
            frame_length -= 1
        if frame_length < poly_order + 1:
            return signal.copy()

    if frame_length % 2 == 0:
        frame_length += 1

    smoothed = savgol_filter(signal, frame_length, poly_order)
    return smoothed


def min_max_normalize(signal, range_min=0, range_max=1):
    if len(signal) == 0:
        return signal.copy()

    sig_min = np.min(signal)
    sig_max = np.max(signal)
    if sig_max - sig_min < 1e-10:
        return np.full_like(signal, range_min)
    normalized = (signal - sig_min) / (sig_max - sig_min)
    normalized = normalized * (range_max - range_min) + range_min
    return np.clip(normalized, range_min, range_max)


def preprocess_signal(signal, config):
    outlier_threshold = config.get('outlierRemoval', {}).get('threshold', 3)
    frame_length = config.get('smoothing', {}).get('frameLength', 11)
    poly_order = config.get('smoothing', {}).get('polyOrder', 3)

    clean, outliers = zscore_outlier_removal(signal, outlier_threshold)
    smoothed = savitzky_golay_smooth(clean, frame_length, poly_order)
    normalized = min_max_normalize(smoothed)

    return {
        'original': signal,
        'cleaned': clean,
        'smoothed': smoothed,
        'normalized': normalized,
        'outlierMask': outliers,
        'nOutliers': int(np.sum(outliers)),
    }
