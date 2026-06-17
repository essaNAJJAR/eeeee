import numpy as np
import os

APPLIANCES_UKDALE = ['Kettle', 'Fridge', 'Microwave', 'Washer', 'Dishwasher']
APPLIANCE_PROFILES_UKDALE = {
    'Kettle': {'power': 2500, 'std': 200, 'min_duration': 60, 'max_duration': 300, 'cycle_prob': 0.3},
    'Fridge': {'power': 150, 'std': 20, 'min_duration': 300, 'max_duration': 1800, 'cycle_prob': 0.7},
    'Microwave': {'power': 1000, 'std': 100, 'min_duration': 10, 'max_duration': 120, 'cycle_prob': 0.2},
    'Washer': {'power': 2000, 'std': 300, 'min_duration': 1800, 'max_duration': 5400, 'cycle_prob': 0.15},
    'Dishwasher': {'power': 1800, 'std': 250, 'min_duration': 2400, 'max_duration': 5400, 'cycle_prob': 0.1},
}

APPLIANCES_REFIT = ['Oven', 'TV', 'PC', 'Heater', 'Fan']
APPLIANCE_PROFILES_REFIT = {
    'Oven': {'power': 2000, 'std': 150, 'min_duration': 300, 'max_duration': 3600, 'cycle_prob': 0.1},
    'TV': {'power': 100, 'std': 15, 'min_duration': 600, 'max_duration': 7200, 'cycle_prob': 0.8},
    'PC': {'power': 200, 'std': 30, 'min_duration': 1200, 'max_duration': 14400, 'cycle_prob': 0.6},
    'Heater': {'power': 1500, 'std': 200, 'min_duration': 600, 'max_duration': 7200, 'cycle_prob': 0.5},
    'Fan': {'power': 50, 'std': 10, 'min_duration': 300, 'max_duration': 3600, 'cycle_prob': 0.7},
}

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'UK-DALE')


def _ensure_data_dir():
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    return LOCAL_DATA_DIR


def download_ukdale(home_id=1, url=None):
    import urllib.request
    _ensure_data_dir()
    filename = f"uk-dale-house-{home_id}.h5"
    filepath = os.path.join(LOCAL_DATA_DIR, filename)
    if os.path.exists(filepath):
        return filepath
    if url is None:
        url = f"http://data.ukedc.rl.ac.uk/simpleweb/index.php/download/uk-dale-house-{home_id}-hdf5"
    print(f"Downloading UK-DALE house {home_id}...")
    print(f"Place manually at: {filepath}")
    try:
        urllib.request.urlretrieve(url, filepath)
        return filepath
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def _load_h5_data(home_id=1):
    filepath = os.path.join(LOCAL_DATA_DIR, f"uk-dale-house-{home_id}.h5")
    if not os.path.exists(filepath):
        for alt in [f"uk_dale_house_{home_id}.h5", f"house_{home_id}.h5", f"{home_id}.h5"]:
            p = os.path.join(LOCAL_DATA_DIR, alt)
            if os.path.exists(p):
                filepath = p
                break
        else:
            return None
    try:
        import h5py
    except ImportError:
        return None
    data = {}
    try:
        with h5py.File(filepath, 'r') as f:
            building = f[f'building/{home_id}'] if 'building' in f else f[str(home_id)]
            if 'elec' not in building:
                return None
            elec = building['elec']
            aggregate = None
            appliance_data = {}
            for mk in elec.keys():
                m = elec[mk]
                if 'power' in m and 'active' in m['power']:
                    pd_ = np.array(m['power']['active']).flatten().astype(np.float64)
                elif 'power' in m:
                    pd_ = np.array(m['power']).flatten().astype(np.float64)
                else:
                    continue
                pd_ = np.nan_to_num(pd_, nan=0.0, posinf=0.0, neginf=0.0)
                pd_ = np.maximum(pd_, 0)
                mt = m.attrs.get('type', b'unknown')
                if isinstance(mt, bytes):
                    mt = mt.decode('utf-8')
                if mt == 'aggregate' or aggregate is None:
                    aggregate = pd_
                appliance_data[mk] = {'power': pd_, 'type': mt}
            if aggregate is not None:
                data = {'aggregate': aggregate, 'appliance_power': appliance_data}
    except Exception as e:
        print(f"HDF5 error: {e}")
        return None
    return data if 'aggregate' in data else None


def _create_windows_fast(signal, window_size, overlap):
    step = window_size - overlap
    n = len(signal)
    if n < window_size:
        return signal.reshape(1, -1)[:, :window_size]
    n_windows = (n - window_size) // step + 1
    shape = (n_windows, window_size)
    strides = (signal.strides[0] * step, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()


def _compute_labels_fast(aggregate, appliance_map, appliance_labels, window_size, overlap):
    step = window_size - overlap
    n_windows = (len(aggregate) - window_size) // step + 1
    n_apps = len(appliance_labels)
    app_keys = [k for k in appliance_labels if k in appliance_map]
    if not app_keys:
        return np.zeros(n_windows, dtype=int)

    starts = np.arange(n_windows) * step
    ends = starts + window_size
    app_powers = np.column_stack([
        np.array([np.mean(appliance_map[k][s:e]) for k in app_keys for s, e in zip(starts, ends)]).reshape(len(app_keys), n_windows)
    ])
    labels = np.argmax(app_powers, axis=0).astype(int)
    return labels


def load_ukdale(home_id=1, config=None):
    if config is None:
        config = {}
    sampling_rate = config.get('samplingRate', 0.1667)
    window_size = config.get('windowSize', 256)
    overlap = config.get('overlap', 128)
    train_ratio = config.get('trainRatio', 0.8)
    duration = config.get('duration', None)

    real_data = _load_h5_data(home_id)

    if real_data is not None:
        aggregate = real_data['aggregate']
        if duration:
            max_samples = int(duration / sampling_rate)
            aggregate = aggregate[:max_samples]
        appliance_map = {name: np.zeros_like(aggregate) for name in APPLIANCES_UKDALE}
        for key, info in real_data.get('appliance_power', {}).items():
            mt = info.get('type', '').lower()
            pw = info['power']
            for name in APPLIANCES_UKDALE:
                if name.lower() in mt or mt in name.lower():
                    if len(pw) >= len(aggregate):
                        appliance_map[name] = pw[:len(aggregate)]
                    else:
                        appliance_map[name][:len(pw)] = pw
                    break
        print(f"Loaded real UK-DALE house {home_id}: {len(aggregate)} samples")
    else:
        if duration is None:
            duration = 86400
        real_data = _generate_realistic_data_fast(APPLIANCES_UKDALE, APPLIANCE_PROFILES_UKDALE,
                                                  sampling_rate, duration)
        aggregate = real_data['aggregate']
        appliance_map = real_data['appliancePower']

    windows = _create_windows_fast(aggregate, window_size, overlap)
    appliance_labels = list(appliance_map.keys())
    labels = _compute_labels_fast(aggregate, appliance_map, appliance_labels, window_size, overlap)

    n_train = int(len(windows) * train_ratio)
    indices = np.random.permutation(len(windows))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return {
        'trainData': windows[train_idx],
        'testData': windows[test_idx],
        'trainLabels': labels[train_idx],
        'testLabels': labels[test_idx],
        'applianceLabels': appliance_labels,
        'aggregate': aggregate,
        'appliancePower': appliance_map,
        'applianceNames': appliance_labels,
    }


def load_refit(home_id=1, config=None):
    if config is None:
        config = {}
    sampling_rate = config.get('samplingRate', 0.1667)
    window_size = config.get('windowSize', 256)
    overlap = config.get('overlap', 128)
    train_ratio = config.get('trainRatio', 0.8)
    duration = config.get('duration', None)

    refit_dir = os.path.join(os.path.dirname(LOCAL_DATA_DIR), 'REFIT')
    os.makedirs(refit_dir, exist_ok=True)
    csv_file = os.path.join(refit_dir, f"CLEAN_House{home_id}.csv")
    real_data = _load_refit_csv(csv_file)

    if real_data is not None:
        aggregate = real_data['aggregate']
        if duration:
            max_samples = int(duration / sampling_rate)
            aggregate = aggregate[:max_samples]
        appliance_map = real_data.get('appliancePower', {})
        print(f"Loaded real REFIT house {home_id}: {len(aggregate)} samples")
    else:
        if duration is None:
            duration = 86400
        real_data = _generate_realistic_data_fast(APPLIANCES_REFIT, APPLIANCE_PROFILES_REFIT,
                                                  sampling_rate, duration)
        aggregate = real_data['aggregate']
        appliance_map = real_data['appliancePower']

    windows = _create_windows_fast(aggregate, window_size, overlap)
    appliance_labels = list(appliance_map.keys()) if appliance_map else APPLIANCES_REFIT
    labels = _compute_labels_fast(aggregate, appliance_map, appliance_labels, window_size, overlap)

    n_train = int(len(windows) * train_ratio)
    indices = np.random.permutation(len(windows))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return {
        'trainData': windows[train_idx],
        'testData': windows[test_idx],
        'trainLabels': labels[train_idx],
        'testLabels': labels[test_idx],
        'applianceLabels': appliance_labels,
        'aggregate': aggregate,
        'appliancePower': appliance_map,
        'applianceNames': appliance_labels,
    }


def _load_refit_csv(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
    except Exception:
        return None
    columns = df.columns.tolist()
    time_col = next((c for c in columns if 'time' in c.lower() or 'date' in c.lower()), columns[0])
    agg_col = next((c for c in columns if c != time_col and ('aggregate' in c.lower() or 'whole' in c.lower())), columns[1] if len(columns) > 1 else None)
    if agg_col is None:
        return None
    aggregate = np.maximum(np.nan_to_num(df[agg_col].values.astype(np.float64), nan=0.0), 0)
    name_map = {'kettle': 'Kettle', 'fridge': 'Fridge', 'microwave': 'Microwave',
                'washer': 'Washer', 'dishwasher': 'Dishwasher', 'oven': 'Oven',
                'tv': 'TV', 'television': 'TV', 'pc': 'PC', 'computer': 'PC',
                'heater': 'Heater', 'fan': 'Fan', 'light': 'Lighting', 'lighting': 'Lighting'}
    appliance_power = {}
    for col in columns:
        if col in (time_col, agg_col):
            continue
        cl = col.lower()
        for pat, name in name_map.items():
            if pat in cl:
                vals = np.maximum(np.nan_to_num(df[col].values.astype(np.float64), nan=0.0), 0)
                appliance_power[name] = vals
                break
    return {'aggregate': aggregate, 'appliancePower': appliance_power}


def generate_smart_meter_data(duration=86400, sampling_rate=1/6, appliances=None, profiles=None, noise_level=10):
    if appliances is None:
        appliances = APPLIANCES_UKDALE
    if profiles is None:
        profiles = APPLIANCE_PROFILES_UKDALE
    return _generate_realistic_data_fast(appliances, profiles, sampling_rate, duration)


def _generate_realistic_data_fast(appliances, profiles, sampling_rate, duration):
    n_samples = int(duration / sampling_rate)
    t = np.arange(n_samples) * sampling_rate
    hours_of_day = (t / 3600) % 24

    aggregate = 50 + np.sin(2 * np.pi * hours_of_day / 24) * 20 + 50 + np.random.randn(n_samples) * 5
    appliance_power = {}

    for name in appliances:
        p = profiles[name]
        signal = np.zeros(n_samples)
        i = 0
        min_dur = int(p['min_duration'] / sampling_rate)
        max_dur = int(p['max_duration'] / sampling_rate) + 1
        while i < n_samples:
            hour = hours_of_day[i]
            prob = p['cycle_prob'] * (1.5 if 7 <= hour <= 10 or 17 <= hour <= 22 else 0.3 if hour >= 23 or hour <= 5 else 1.0)
            if np.random.rand() < prob:
                remaining = n_samples - i
                if remaining < min_dur:
                    break
                high_val = min(max_dur, remaining)
                if high_val <= min_dur:
                    dur = min_dur
                else:
                    dur = np.random.randint(min_dur, high_val)
                end = min(i + dur, n_samples)
                pw = max(0, p['power'] + np.random.randn() * p['std'])
                on = max(1, int(0.05 * (end - i)))
                signal[i:i + on] = pw * np.linspace(0, 1, on)
                signal[i + on:end] = pw + np.random.randn(end - i - on) * p['std'] * 0.05
                i = end + np.random.randint(200, 1000)
            else:
                i += np.random.randint(100, 500)
        appliance_power[name] = signal
        aggregate += signal

    return {'aggregate': aggregate, 'appliancePower': appliance_power, 'time': t, 'samplingRate': sampling_rate}
