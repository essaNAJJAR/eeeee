import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from preprocessing.preprocessor import zscore_outlier_removal, savitzky_golay_smooth, min_max_normalize, preprocess_signal
from feature_extraction.vmd import vmd, feedback_vmd
from feature_extraction.teo import teager_energy_operator
from feature_extraction.extract_features import extract_features
from fuzzy_logic.fuzzy_system import FuzzyLogicSystem, evaluate_fuzzy_logic
from deep_learning.model import VAEDCCNNAtt, predict_with_model
from federated_learning.multi_krum import multi_krum, detect_byzantine_gat
from evaluation.evaluation_framework import EvaluationFramework
from advanced.signal_quality import SignalQualityAssessor
from advanced.attention import AttentionMechanism


class NILMTestRunner:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0

    def run_test(self, name, test_func):
        try:
            test_func()
            self.results.append({'name': name, 'status': 'PASS'})
            self.passed += 1
        except Exception as e:
            self.results.append({'name': name, 'status': 'FAIL', 'error': str(e)})
            self.failed += 1

    def report(self):
        lines = ["=" * 50]
        lines.append("NILM Test Report")
        lines.append("=" * 50)
        for r in self.results:
            status = r['status']
            name = r['name']
            if status == 'PASS':
                lines.append(f"  [PASS] {name}")
            else:
                lines.append(f"  [FAIL] {name}: {r.get('error', '')}")
        lines.append(f"\nTotal: {self.passed + self.failed} | Passed: {self.passed} | Failed: {self.failed}")
        lines.append("=" * 50)
        return "\n".join(lines)


def test_preprocessing():
    signal = 100 + 50 * np.random.randn(500)
    signal[100] = 1000
    signal[200] = -500

    clean, mask = zscore_outlier_removal(signal, 3)
    assert clean.shape == signal.shape
    assert np.sum(mask) >= 0

    smoothed = savitzky_golay_smooth(clean, 11, 3)
    assert smoothed.shape == signal.shape

    normalized = min_max_normalize(smoothed)
    assert np.min(normalized) >= 0
    assert np.max(normalized) <= 1

    result = preprocess_signal(signal, {'outlierRemoval': {'threshold': 3}, 'smoothing': {'frameLength': 11, 'polyOrder': 3}})
    assert 'normalized' in result


def test_vmd():
    t = np.linspace(0, 1, 256)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    modes, u_hat, omega = vmd(signal, alpha=2000, K=3)
    assert modes.shape[0] == 3
    assert modes.shape[1] == len(signal)

    modes2, omega2, score, best_K = feedback_vmd(signal, [3, 5])
    assert modes2.shape[0] == best_K
    assert 3 <= best_K <= 5


def test_teo():
    signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 256))
    teo = teager_energy_operator(signal)
    assert teo.shape == signal.shape
    assert np.all(teo >= 0)


def test_feature_extraction():
    t = np.linspace(0, 1, 256)
    signal = np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 15 * t)
    modes, _, _, _ = feedback_vmd(signal, [3, 4])
    teo = np.zeros_like(modes)
    for k in range(modes.shape[0]):
        teo[k] = teager_energy_operator(modes[k])

    features, names = extract_features(modes, teo)
    assert len(features) > 0
    assert len(names) > 0
    assert len(features) == len(names)


def test_fuzzy_logic():
    state, idx = evaluate_fuzzy_logic(0.5, 0.3, 0.7, 0.6)
    assert state in ['OFF', 'PARTIAL', 'ON']
    assert 0 <= idx <= 2

    fis = FuzzyLogicSystem()
    assert len(fis.rules) == 81


def test_deep_learning():
    model = VAEDCCNNAtt({'numClasses': 5, 'latentDim': 16, 'encoderChannels': [32, 64], 'decoderChannels': [64, 32]})
    x = np.random.randn(4, 1, 256)
    result = model.forward(x)
    assert result['predictions'].shape == (4,)
    assert result['probabilities'].shape == (4, 5)

    preds, probs = predict_with_model(model, np.random.randn(8, 256))
    assert preds.shape == (8,)


def test_federated_learning():
    updates = [[np.random.randn(10, 5) for _ in range(3)] for _ in range(10)]
    agg = multi_krum(updates, 2)
    assert agg is not None
    assert len(agg) == 3

    normal, suspicious = detect_byzantine_gat(updates, 0.5)
    assert len(normal) + len(suspicious) == 10


def test_evaluation():
    eval_fw = EvaluationFramework()
    preds = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    gt = np.array([0, 1, 2, 0, 2, 1, 0, 1])
    metrics = eval_fw.evaluate(preds, gt)
    assert 'accuracy' in metrics
    assert 'perClass' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_signal_quality():
    assessor = SignalQualityAssessor()
    signal = 100 + 20 * np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
    result = assessor.assess(signal)
    assert 'qualityScore' in result
    assert 0 <= result['qualityScore'] <= 1


def test_attention():
    attn = AttentionMechanism(dim=64, num_heads=8)
    x = np.random.randn(2, 10, 64)
    out = attn.forward(x)
    assert out.shape == (2, 10, 64)


def test_edge_cases():
    signal = np.array([])
    clean, mask = zscore_outlier_removal(signal)
    assert len(clean) == 0

    signal = np.array([5.0])
    clean, mask = zscore_outlier_removal(signal)
    assert len(clean) == 1

    signal = np.ones(100) * 42.0
    clean, mask = zscore_outlier_removal(signal)
    assert np.all(clean == 42.0)

    signal = np.random.randn(50)
    modes, _, _, _ = feedback_vmd(signal, [2, 3])
    assert modes.shape[0] >= 2


def run_all_tests():
    runner = NILMTestRunner()

    runner.run_test("Preprocessing", test_preprocessing)
    runner.run_test("VMD", test_vmd)
    runner.run_test("TEO", test_teo)
    runner.run_test("Feature Extraction", test_feature_extraction)
    runner.run_test("Fuzzy Logic", test_fuzzy_logic)
    runner.run_test("Deep Learning", test_deep_learning)
    runner.run_test("Federated Learning", test_federated_learning)
    runner.run_test("Evaluation", test_evaluation)
    runner.run_test("Signal Quality", test_signal_quality)
    runner.run_test("Attention", test_attention)
    runner.run_test("Edge Cases", test_edge_cases)

    print(runner.report())
    return runner


if __name__ == '__main__':
    run_all_tests()
