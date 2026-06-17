import numpy as np


class EvaluationFramework:
    def __init__(self):
        self.metrics = {}
        self.confusion_matrix = None

    def evaluate(self, predictions, ground_truth):
        predictions = np.asarray(predictions).flatten()
        ground_truth = np.asarray(ground_truth).flatten()
        n = min(len(predictions), len(ground_truth))
        predictions = predictions[:n]
        ground_truth = ground_truth[:n]

        metrics = {}
        metrics['accuracy'] = np.mean(predictions == ground_truth)

        classes = np.unique(np.concatenate([predictions, ground_truth]))
        metrics['perClass'] = {}

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_samples = 0

        for c in classes:
            tp = np.sum((predictions == c) & (ground_truth == c))
            fp = np.sum((predictions == c) & (ground_truth != c))
            fn = np.sum((predictions != c) & (ground_truth == c))
            tn = np.sum((predictions != c) & (ground_truth != c))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics['perClass'][str(c)] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': int(np.sum(ground_truth == c)),
            }

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_samples += np.sum(ground_truth == c)

        metrics['macroPrecision'] = np.mean([v['precision'] for v in metrics['perClass'].values()])
        metrics['macroRecall'] = np.mean([v['recall'] for v in metrics['perClass'].values()])
        metrics['macroF1'] = np.mean([v['f1'] for v in metrics['perClass'].values()])

        weighted_f1 = sum(
            v['f1'] * v['support'] for v in metrics['perClass'].values()
        ) / max(total_samples, 1)
        metrics['weightedF1'] = weighted_f1

        if np.issubdtype(predictions.dtype, np.floating) or np.issubdtype(ground_truth.dtype, np.floating):
            metrics['mae'] = float(np.mean(np.abs(predictions - ground_truth)))
            metrics['rmse'] = float(np.sqrt(np.mean((predictions - ground_truth) ** 2)))
            nonzero = ground_truth != 0
            if np.any(nonzero):
                metrics['mape'] = float(np.mean(np.abs((ground_truth[nonzero] - predictions[nonzero]) / ground_truth[nonzero])) * 100)
            else:
                metrics['mape'] = 0.0

        n_classes = len(classes)
        self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
        for i, c_true in enumerate(classes):
            for j, c_pred in enumerate(classes):
                self.confusion_matrix[i, j] = np.sum((ground_truth == c_true) & (predictions == c_pred))

        self.metrics = metrics
        return metrics

    def evaluate_events(self, predicted_events, true_events, tolerance=5):
        if not true_events or not predicted_events:
            return {'precision': 0, 'recall': 0, 'f1': 0}

        true_indices = set(true_events.get('indices', []))
        pred_indices = set(predicted_events.get('indices', []))

        matched_pred = set()
        matched_true = set()

        for ti in true_indices:
            for pi in pred_indices:
                if abs(ti - pi) <= tolerance and pi not in matched_pred:
                    matched_pred.add(pi)
                    matched_true.add(ti)
                    break

        tp = len(matched_true)
        fp = len(pred_indices) - len(matched_pred)
        fn = len(true_indices) - len(matched_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

    def evaluate_reconstruction(self, original, reconstructed):
        original = np.asarray(original).flatten()
        reconstructed = np.asarray(reconstructed).flatten()
        n = min(len(original), len(reconstructed))
        if n == 0:
            return {'mse': 0.0, 'mae': 0.0, 'snr': 0.0, 'r2': 0.0}
        original = original[:n]
        reconstructed = reconstructed[:n]

        mse = float(np.mean((original - reconstructed) ** 2))
        mae = float(np.mean(np.abs(original - reconstructed)))

        signal_power = np.mean(original ** 2)
        noise_power = mse
        snr = 10 * np.log10(signal_power / (noise_power + 1e-12))

        ss_res = np.sum((original - reconstructed) ** 2)
        ss_tot = np.sum((original - np.mean(original)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)

        return {'mse': mse, 'mae': mae, 'snr': float(snr), 'r2': float(r2)}

    def report(self):
        lines = ["=" * 50]
        lines.append("NILM Evaluation Report")
        lines.append("=" * 50)

        if self.metrics:
            lines.append(f"\nAccuracy: {self.metrics.get('accuracy', 0):.4f}")
            lines.append(f"Macro F1: {self.metrics.get('macroF1', 0):.4f}")
            lines.append(f"Weighted F1: {self.metrics.get('weightedF1', 0):.4f}")

            if 'perClass' in self.metrics:
                lines.append("\nPer-Class Metrics:")
                for cls, vals in self.metrics['perClass'].items():
                    lines.append(f"  Class {cls}: P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f}")

            if 'mae' in self.metrics:
                lines.append(f"\nMAE: {self.metrics['mae']:.4f}")
                lines.append(f"RMSE: {self.metrics.get('rmse', 0):.4f}")
                lines.append(f"MAPE: {self.metrics.get('mape', 0):.2f}%")

        if self.confusion_matrix is not None:
            lines.append("\nConfusion Matrix:")
            lines.append(str(self.confusion_matrix))

        lines.append("=" * 50)
        return "\n".join(lines)
