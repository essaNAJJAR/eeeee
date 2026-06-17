import numpy as np


def detect_events(signal, sampling_rate=1/6, threshold=0.1, min_gap=3):
    diff = np.diff(signal)
    abs_diff = np.abs(diff)
    adaptive_threshold = threshold * np.std(abs_diff) + np.mean(abs_diff)

    rising = np.where(diff > adaptive_threshold)[0]
    falling = np.where(diff < -adaptive_threshold)[0]

    events = {
        'indices': [],
        'types': [],
        'timestamps': [],
        'powerLevels': [],
        'durations': [],
    }

    all_edges = []
    for idx in rising:
        all_edges.append((idx, 'on'))
    for idx in falling:
        all_edges.append((idx, 'off'))
    all_edges.sort(key=lambda x: x[0])

    if not all_edges:
        return events

    merged = [all_edges[0]]
    for edge in all_edges[1:]:
        if edge[0] - merged[-1][0] < min_gap:
            if edge[1] != merged[-1][1]:
                merged[-1] = edge
        else:
            merged.append(edge)

    for edge in merged:
        idx, etype = edge
        events['indices'].append(idx)
        events['types'].append(etype)
        events['timestamps'].append(idx * sampling_rate)

        if etype == 'on':
            window_start = max(0, idx - 5)
            window_end = min(len(signal), idx + 20)
            events['powerLevels'].append(np.mean(signal[window_start:idx]) if idx > 0 else 0)
        else:
            window_start = max(0, idx - 5)
            events['powerLevels'].append(np.mean(signal[window_start:idx]) if idx > 0 else 0)

        events['durations'].append(0)

    for i in range(len(events['indices']) - 1):
        if events['types'][i] == 'on':
            events['durations'][i] = (events['indices'][i + 1] - events['indices'][i]) * sampling_rate

    return events


APPLIANCE_CATEGORIES = {
    'Kettle': {'power_range': (1500, 3500), 'duration_range': (30, 600)},
    'Fridge': {'power_range': (50, 300), 'duration_range': (100, 3600)},
    'Microwave': {'power_range': (500, 2000), 'duration_range': (5, 300)},
    'Washer': {'power_range': (500, 3000), 'duration_range': (600, 7200)},
    'Dishwasher': {'power_range': (1000, 3000), 'duration_range': (600, 7200)},
    'Oven': {'power_range': (1000, 3500), 'duration_range': (120, 5400)},
    'Heater': {'power_range': (500, 3000), 'duration_range': (300, 14400)},
    'TV': {'power_range': (20, 200), 'duration_range': (300, 14400)},
    'PC': {'power_range': (50, 500), 'duration_range': (300, 28800)},
    'Lighting': {'power_range': (10, 200), 'duration_range': (60, 43200)},
}


def classify_events(events, signal, sampling_rate=1/6):
    classifications = {
        'appliance': [],
        'confidence': [],
        'powerDelta': [],
    }

    for i, idx in enumerate(events['indices']):
        if events['types'][i] == 'on' and i + 1 < len(events['indices']):
            next_idx = events['indices'][i + 1]
            segment = signal[idx:next_idx] if next_idx > idx else signal[idx:idx + 10]
            power_level = np.max(segment) - np.min(segment)
            duration = (next_idx - idx) * sampling_rate
        else:
            power_level = events['powerLevels'][i] if i < len(events['powerLevels']) else 0
            duration = events['durations'][i] if i < len(events['durations']) else 0

        best_match = 'Unknown'
        best_score = 0

        for name, specs in APPLIANCE_CATEGORIES.items():
            p_low, p_high = specs['power_range']
            d_low, d_high = specs['duration_range']

            p_score = 1.0 - min(abs(power_level - (p_low + p_high) / 2) / ((p_high - p_low) / 2 + 1), 1.0)
            d_score = 1.0 - min(abs(duration - (d_low + d_high) / 2) / ((d_high - d_low) / 2 + 1), 1.0)
            score = 0.6 * p_score + 0.4 * d_score

            if score > best_score:
                best_score = score
                best_match = name

        classifications['appliance'].append(best_match)
        classifications['confidence'].append(best_score)
        classifications['powerDelta'].append(power_level)

    return classifications
