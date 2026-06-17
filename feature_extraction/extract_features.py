import numpy as np


def extract_features(modes, teo_energy):
    feature_names = []
    features = []

    for k in range(modes.shape[0]):
        mode = modes[k]
        feature_names.extend([
            f'mode{k}_mean', f'mode{k}_std', f'mode{k}_max', f'mode{k}_min',
        ])
        features.extend([
            np.mean(mode), np.std(mode), np.max(mode), np.min(mode),
        ])

    for k in range(teo_energy.shape[0]):
        teo = teo_energy[k]
        feature_names.extend([
            f'teo{k}_mean', f'teo{k}_std', f'teo{k}_max',
        ])
        features.extend([
            np.mean(teo), np.std(teo), np.max(teo),
        ])

    for k in range(modes.shape[0]):
        spectrum = np.abs(np.fft.rfft(modes[k]))
        feature_names.extend([
            f'fft{k}_mean', f'fft{k}_max',
        ])
        features.extend([
            np.mean(spectrum), np.max(spectrum),
        ])

    for i in range(modes.shape[0]):
        for j in range(i + 1, modes.shape[0]):
            if len(modes[i]) < 2 or np.std(modes[i]) == 0 or np.std(modes[j]) == 0:
                corr = 0.0
            else:
                corr = np.corrcoef(modes[i], modes[j])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            feature_names.append(f'cross{i}{j}_corr')
            features.append(corr)

    return np.array(features), feature_names


def extract_batch_features(windows, teo_energy_batch=None):
    all_features = []
    all_names = None

    for i in range(windows.shape[0]):
        window = windows[i]
        from .vmd import feedback_vmd
        modes, omega, _, _ = feedback_vmd(window)

        if teo_energy_batch is not None:
            teo_energy = teo_energy_batch[i] if i < len(teo_energy_batch) else np.zeros_like(modes)
        else:
            from .teo import teager_energy_operator
            teo_energy = np.zeros_like(modes)
            for k in range(modes.shape[0]):
                teo_energy[k, :] = teager_energy_operator(modes[k, :])

        features, names = extract_features(modes, teo_energy)
        all_features.append(features)
        if all_names is None:
            all_names = names

    return np.array(all_features), all_names
