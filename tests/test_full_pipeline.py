import sys, os, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results = {}

print("\n" + "#" * 60)
print("# PyNILM Full End-to-End Pipeline Test")
print("#" * 60)
t_total = time.time()

# TEST 1
print("\n" + "=" * 60)
print("TEST 1: Data Loading (UK-DALE)")
print("=" * 60)
try:
    from data.data_loader import load_ukdale
    t0 = time.time()
    d = load_ukdale(1, {"samplingRate": 0.1667, "windowSize": 256, "overlap": 128, "trainRatio": 0.8, "duration": 7200})
    elapsed = time.time() - t0
    print("  Train:", d["trainData"].shape)
    print("  Test:", d["testData"].shape)
    print("  Appliances:", d["applianceNames"])
    print("  Time: %.2fs" % elapsed)
    results["Data Loading"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Data Loading"] = "FAIL"
    sys.exit(1)

# TEST 2
print("\n" + "=" * 60)
print("TEST 2: Preprocessing")
print("=" * 60)
try:
    from preprocessing.preprocessor import preprocess_signal
    config = {"outlierRemoval": {"threshold": 3}, "smoothing": {"frameLength": 11, "polyOrder": 3}}
    window = d["testData"][0]
    t0 = time.time()
    preprocessed = preprocess_signal(window, config)
    elapsed = time.time() - t0
    print("  Normalized:", preprocessed["normalized"].shape)
    print("  Range: [%.3f, %.3f]" % (preprocessed["normalized"].min(), preprocessed["normalized"].max()))
    print("  Outliers:", preprocessed["nOutliers"])
    print("  Time: %.2fs" % elapsed)
    results["Preprocessing"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Preprocessing"] = "FAIL"

# TEST 3
print("\n" + "=" * 60)
print("TEST 3: Feature Extraction (FVMD + TEO)")
print("=" * 60)
try:
    from feature_extraction.vmd import feedback_vmd
    from feature_extraction.teo import teager_energy_operator
    from feature_extraction.extract_features import extract_features
    t0 = time.time()
    modes, omega, score, best_K = feedback_vmd(preprocessed["normalized"], [3, 9])
    elapsed_vmd = time.time() - t0
    t1 = time.time()
    teo_energy = np.zeros_like(modes)
    for k in range(modes.shape[0]):
        teo_energy[k] = teager_energy_operator(modes[k])
    elapsed_teo = time.time() - t1
    features, feature_names = extract_features(modes, teo_energy)
    print("  Best K: %d, Score: %.4f" % (best_K, score))
    print("  Modes:", modes.shape)
    print("  Features: %d dims" % len(features))
    print("  VMD: %.2fs, TEO: %.2fs" % (elapsed_vmd, elapsed_teo))
    results["Feature Extraction"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Feature Extraction"] = "FAIL"

# TEST 4
print("\n" + "=" * 60)
print("TEST 4: Fuzzy Logic")
print("=" * 60)
try:
    from fuzzy_logic.fuzzy_system import evaluate_fuzzy_logic
    t0 = time.time()
    states = set()
    for dp in [0.2, 0.5, 0.8]:
        for sg in [0.2, 0.5, 0.8]:
            state, idx = evaluate_fuzzy_logic(dp, sg, 0.5, 0.5)
            states.add(state)
    elapsed = time.time() - t0
    print("  States:", sorted(states))
    print("  Time: %.2fs" % elapsed)
    results["Fuzzy Logic"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Fuzzy Logic"] = "FAIL"

# TEST 5
print("\n" + "=" * 60)
print("TEST 5: Event Detection")
print("=" * 60)
try:
    from event_detection.detect_events import detect_events, classify_events
    t0 = time.time()
    events = detect_events(preprocessed["normalized"], 1/6)
    n_events = len(events["indices"])
    elapsed = time.time() - t0
    print("  Events detected:", n_events)
    if n_events > 0:
        classifications = classify_events(events, preprocessed["normalized"])
        for i in range(min(n_events, 3)):
            print("    %s (%.0fW)" % (classifications["appliance"][i], classifications["powerDelta"][i]))
    print("  Time: %.2fs" % elapsed)
    results["Event Detection"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Event Detection"] = "FAIL"

# TEST 6
print("\n" + "=" * 60)
print("TEST 6: Deep Learning (Inference)")
print("=" * 60)
try:
    from deep_learning.model import VAEDCCNNAtt, predict_with_model
    t0 = time.time()
    model = VAEDCCNNAtt({"numClasses": 5, "latentDim": 16, "encoderChannels": [32, 64], "decoderChannels": [64, 32], "inputDim": 256})
    preds, probs = predict_with_model(model, features.reshape(1, -1))
    elapsed = time.time() - t0
    print("  Prediction:", preds)
    print("  Confidence: %.4f" % probs.max())
    print("  Time: %.2fs" % elapsed)
    results["DL Inference"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["DL Inference"] = "FAIL"

# TEST 7
print("\n" + "=" * 60)
print("TEST 7: Deep Learning (Training)")
print("=" * 60)
try:
    from deep_learning.model import train_vae_dccnn_model
    train_cfg = {
        "numClasses": 5, "latentDim": 16, "encoderChannels": [32, 64],
        "decoderChannels": [64, 32], "batchSize": 16, "epochs": 10,
        "learningRate": 0.001, "vaeBeta": 0.5, "inputDim": 256,
    }
    t0 = time.time()
    trained_model, history = train_vae_dccnn_model(d["trainData"], d["trainLabels"], train_cfg)
    elapsed = time.time() - t0
    print("  Final loss: %.4f" % history["loss"][-1])
    print("  Final accuracy: %.4f" % history["accuracy"][-1])
    print("  Training time: %.2fs" % elapsed)
    result = trained_model.predict(d["testData"][:4])
    print("  Test predictions:", result["predictions"])
    results["DL Training"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["DL Training"] = "FAIL"

# TEST 8
print("\n" + "=" * 60)
print("TEST 8: Evaluation Framework")
print("=" * 60)
try:
    from evaluation.evaluation_framework import EvaluationFramework
    t0 = time.time()
    eval_fw = EvaluationFramework()
    test_preds = trained_model.predict(d["testData"][:10])["predictions"]
    gt = d["testLabels"][:len(test_preds)]
    metrics = eval_fw.evaluate(test_preds, gt)
    elapsed = time.time() - t0
    print("  Accuracy: %.4f" % metrics["accuracy"])
    print("  Macro F1: %.4f" % metrics["macroF1"])
    print("  Weighted F1: %.4f" % metrics["weightedF1"])
    report = eval_fw.report()
    print("  Report lines:", len(report.splitlines()))
    print("  Time: %.2fs" % elapsed)
    results["Evaluation"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Evaluation"] = "FAIL"

# TEST 9
print("\n" + "=" * 60)
print("TEST 9: Federated Learning")
print("=" * 60)
try:
    from federated_learning.federated_sim import run_federated_learning
    fl_cfg = {"numClients": 10, "numRounds": 5, "byzantineFraction": 0.1, "localEpochs": 2, "learningRate": 0.01}
    t0 = time.time()
    fl_results = run_federated_learning(fl_cfg)
    elapsed = time.time() - t0
    print("  Rounds:", len(fl_results["history"]["loss"]))
    print("  Final acc: %.4f" % fl_results["history"]["accuracy"][-1])
    print("  Byzantine detected:", fl_results["nByzantineDetected"])
    print("  Time: %.2fs" % elapsed)
    results["Federated Learning"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Federated Learning"] = "FAIL"

# TEST 10
print("\n" + "=" * 60)
print("TEST 10: Signal Quality Assessment")
print("=" * 60)
try:
    from advanced.signal_quality import SignalQualityAssessor
    t0 = time.time()
    assessor = SignalQualityAssessor()
    sq_result = assessor.assess(d["aggregate"][:5000])
    elapsed = time.time() - t0
    print("  Quality: %.4f" % sq_result["qualityScore"])
    print("  SNR: %.1f dB" % sq_result["snr"])
    print("  Usable:", assessor.is_usable(d["aggregate"][:5000]))
    print("  Time: %.2fs" % elapsed)
    results["Signal Quality"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Signal Quality"] = "FAIL"

# TEST 11
print("\n" + "=" * 60)
print("TEST 11: Online Learning")
print("=" * 60)
try:
    from online_learning.online_learner import OnlineLearner
    t0 = time.time()
    learner = OnlineLearner(learning_rate=0.001, buffer_size=500)
    learner.initialize_model(256, 5)
    for i in range(10):
        X = np.random.randn(10, 256).astype(np.float64)
        y = np.random.randint(0, 5, 10)
        learner.update(X, y)
    perf = learner.get_performance()
    elapsed = time.time() - t0
    print("  Updates:", learner.total_updates)
    print("  Drifts:", learner.drifts_detected)
    if perf:
        print("  Final acc: %.4f" % perf[-1]["accuracy"])
    print("  Time: %.2fs" % elapsed)
    results["Online Learning"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Online Learning"] = "FAIL"

# TEST 12
print("\n" + "=" * 60)
print("TEST 12: Multi-Household Simulation")
print("=" * 60)
try:
    from multi_household.simulator import MultiHouseholdSimulator
    t0 = time.time()
    sim = MultiHouseholdSimulator(num_households=5, duration=3600)
    sim.generate_data()
    mh_results = sim.analyze()
    export_path = sim.export_to_grid()
    elapsed = time.time() - t0
    print("  Households:", mh_results["grid_level"]["num_households"])
    print("  Grid mean: %.1fW" % mh_results["grid_level"]["total_mean"])
    print("  Peak hour:", mh_results["demand_response"]["peak_hour"])
    print("  Time: %.2fs" % elapsed)
    results["Multi-Household"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Multi-Household"] = "FAIL"

# TEST 13
print("\n" + "=" * 60)
print("TEST 13: Export Results")
print("=" * 60)
try:
    from export.exporter import NILMExporter
    t0 = time.time()
    exporter = NILMExporter()
    exporter.export_all({"metrics": metrics})
    elapsed = time.time() - t0
    files = os.listdir(exporter.output_path)
    print("  Files:", files)
    print("  Time: %.2fs" % elapsed)
    results["Export"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Export"] = "FAIL"

# TEST 14
print("\n" + "=" * 60)
print("TEST 14: GUI Module Imports")
print("=" * 60)
try:
    t0 = time.time()
    from config.config_manager import load_config, save_config
    from streaming.data_stream import DataStream
    from transfer_learning.manager import TransferLearningManager
    from advanced.attention import AttentionMechanism
    elapsed = time.time() - t0
    print("  config_manager: OK")
    print("  DataStream: OK")
    print("  TransferLearningManager: OK")
    print("  AttentionMechanism: OK")
    print("  Time: %.2fs" % elapsed)
    results["GUI Imports"] = "PASS"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["GUI Imports"] = "FAIL"

# TEST 15
print("\n" + "=" * 60)
print("TEST 15: Full Test Suite (11 tests)")
print("=" * 60)
try:
    from tests.run_tests import run_all_tests
    t0 = time.time()
    runner = run_all_tests()
    elapsed = time.time() - t0
    print("  Passed: %d/%d" % (runner.passed, runner.passed + runner.failed))
    print("  Time: %.2fs" % elapsed)
    results["Test Suite"] = "PASS" if runner.failed == 0 else "FAIL"
except Exception as e:
    print("  FAIL:", e)
    traceback.print_exc()
    results["Test Suite"] = "FAIL"

# SUMMARY
total_time = time.time() - t_total
print("\n" + "#" * 60)
print("# FINAL RESULTS")
print("#" * 60)
passed = sum(1 for v in results.values() if v == "PASS")
total = len(results)
for name, status in sorted(results.items()):
    icon = "PASS" if status == "PASS" else "FAIL"
    print("  [%s] %s" % (icon, name))
print("")
print("  %d/%d tests passed" % (passed, total))
print("  Total time: %.1fs" % total_time)
if passed == total:
    print("\n  ALL TESTS PASSED!")
else:
    print("\n  %d TESTS FAILED" % (total - passed))
