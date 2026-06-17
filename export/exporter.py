import numpy as np
import json
import os
import csv


class NILMExporter:
    def __init__(self, output_path=None):
        if output_path is None:
            output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Export')
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def export_all(self, results, config=None):
        self.export_json(results)
        self.export_csv(results)
        self.export_latex(results)
        self.generate_report(results, config)

    def export_json(self, results, filename='results.json'):
        path = os.path.join(self.output_path, filename)
        serializable = self._make_serializable(results)
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        return path

    def export_csv(self, results, filename='results.csv'):
        path = os.path.join(self.output_path, filename)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            if 'metrics' in results:
                for key, val in results['metrics'].items():
                    if isinstance(val, dict):
                        for k, v in val.items():
                            writer.writerow([f'{key}.{k}', v])
                    else:
                        writer.writerow([key, val])
            if 'executionTime' in results:
                writer.writerow(['executionTime', results['executionTime']])
        return path

    def export_latex(self, results, filename='results.tex'):
        path = os.path.join(self.output_path, filename)
        with open(path, 'w') as f:
            f.write('\\documentclass{article}\n')
            f.write('\\usepackage{booktabs}\n')
            f.write('\\usepackage{amsmath}\n')
            f.write('\\begin{document}\n')
            f.write('\\section{NILM Results}\n\n')

            f.write('\\begin{table}[h]\n')
            f.write('\\centering\n')
            f.write('\\caption{Classification Metrics}\n')
            f.write('\\begin{tabular}{lccccc}\n')
            f.write('\\toprule\n')
            f.write('Class & Precision & Recall & F1 & Support \\\\\n')
            f.write('\\midrule\n')

            if 'metrics' in results and 'perClass' in results['metrics']:
                for cls, vals in results['metrics']['perClass'].items():
                    f.write(f'{cls} & {vals["precision"]:.3f} & {vals["recall"]:.3f} & '
                            f'{vals["f1"]:.3f} & {vals["support"]} \\\\\n')
            f.write('\\bottomrule\n')
            f.write('\\end{tabular}\n')
            f.write('\\end{table}\n\n')

            if 'metrics' in results:
                f.write('\\subsection{Overall Metrics}\n')
                f.write('\\begin{itemize}\n')
                for key in ['accuracy', 'macroF1', 'weightedF1']:
                    if key in results['metrics']:
                        f.write(f'\\item {key}: {results["metrics"][key]:.4f}\n')
                f.write('\\end{itemize}\n')

            f.write('\\end{document}\n')
        return path

    def generate_report(self, results, config=None, filename='report.txt'):
        path = os.path.join(self.output_path, filename)
        with open(path, 'w') as f:
            f.write('=' * 60 + '\n')
            f.write('NILM System Report\n')
            f.write('=' * 60 + '\n\n')

            if config:
                f.write(f"Project: {config.get('project', {}).get('name', 'NILM')}\n")
                f.write(f"Version: {config.get('project', {}).get('version', '1.0')}\n")
                f.write(f"Dataset: {config.get('data', {}).get('dataset', 'Unknown')}\n\n")

            if 'executionTime' in results:
                f.write(f"Execution Time: {results['executionTime']:.3f} seconds\n\n")

            if 'metrics' in results:
                f.write('Classification Metrics:\n')
                f.write('-' * 40 + '\n')
                f.write(f"  Accuracy: {results['metrics'].get('accuracy', 0):.4f}\n")
                f.write(f"  Macro F1: {results['metrics'].get('macroF1', 0):.4f}\n")
                f.write(f"  Weighted F1: {results['metrics'].get('weightedF1', 0):.4f}\n")

                if 'mae' in results['metrics']:
                    f.write(f"\n  MAE: {results['metrics']['mae']:.4f}\n")
                    f.write(f"  RMSE: {results['metrics'].get('rmse', 0):.4f}\n")
                    f.write(f"  MAPE: {results['metrics'].get('mape', 0):.2f}%\n")

                if 'perClass' in results['metrics']:
                    f.write('\nPer-Class Breakdown:\n')
                    for cls, vals in results['metrics']['perClass'].items():
                        f.write(f"  {cls}: P={vals['precision']:.3f} R={vals['recall']:.3f} F1={vals['f1']:.3f}\n")

            f.write('\n' + '=' * 60 + '\n')
        return path

    def _make_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
