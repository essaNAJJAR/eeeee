import numpy as np


def getfield_default(struct, field, default=None):
    return struct.get(field, default) if isinstance(struct, dict) else getattr(struct, field, default)


def calculate_nilm_metrics(predictions, ground_truth, appliance_names=None):
    predictions = np.asarray(predictions).flatten()
    ground_truth = np.asarray(ground_truth).flatten()
    n = min(len(predictions), len(ground_truth))
    predictions = predictions[:n]
    ground_truth = ground_truth[:n]

    metrics = {}
    metrics['accuracy'] = float(np.mean(predictions == ground_truth))

    classes = np.unique(np.concatenate([predictions, ground_truth]))
    metrics['perClass'] = {}

    for c in classes:
        tp = int(np.sum((predictions == c) & (ground_truth == c)))
        fp = int(np.sum((predictions == c) & (ground_truth != c)))
        fn = int(np.sum((predictions != c) & (ground_truth == c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics['perClass'][str(c)] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(np.sum(ground_truth == c)),
        }

    metrics['macroF1'] = float(np.mean([v['f1'] for v in metrics['perClass'].values()]))
    metrics['weightedF1'] = float(np.average(
        [v['f1'] for v in metrics['perClass'].values()],
        weights=[v['support'] for v in metrics['perClass'].values()]
    ))

    metrics['mae'] = float(np.mean(np.abs(predictions - ground_truth)))
    metrics['rmse'] = float(np.sqrt(np.mean((predictions - ground_truth) ** 2)))
    nonzero = ground_truth != 0
    if np.any(nonzero):
        metrics['mape'] = float(np.mean(np.abs((ground_truth[nonzero] - predictions[nonzero]) / ground_truth[nonzero])) * 100)
    else:
        metrics['mape'] = 0.0

    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i, ct in enumerate(classes):
        for j, cp in enumerate(classes):
            cm[i, j] = int(np.sum((ground_truth == ct) & (predictions == cp)))
    metrics['confusionMatrix'] = cm

    return metrics


def reconstruction_metrics(original, reconstructed):
    original = np.asarray(original).flatten()
    reconstructed = np.asarray(reconstructed).flatten()
    n = min(len(original), len(reconstructed))
    original = original[:n]
    reconstructed = reconstructed[:n]

    mse = float(np.mean((original - reconstructed) ** 2))
    mae = float(np.mean(np.abs(original - reconstructed)))
    signal_power = np.mean(original ** 2)
    snr = 10 * np.log10(signal_power / (mse + 1e-12))
    ss_res = np.sum((original - reconstructed) ** 2)
    ss_tot = np.sum((original - np.mean(original)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    return {'mse': mse, 'mae': mae, 'snr': float(snr), 'r2': float(r2)}
