# PyNILM - Non-Intrusive Load Monitoring System

## Overview
Python implementation of a comprehensive NILM simulation system based on FVMD-VAE-DCCNN-Att + Hybrid Federated Learning architecture with full GUI.

## Quick Start
```bash
cd PyNILM
pip install -r requirements.txt
python main.py
```

## Project Structure

```
PyNILM/
├── main.py                      # GUI Application (Tkinter)
├── requirements.txt             # Python dependencies
├── Config/                      # Configuration management
│   ├── config_manager.py
│   └── default_config.json
├── Data/                        # Dataset loading (UK-DALE, REFIT)
│   └── data_loader.py
├── Preprocessing/               # Signal preprocessing
│   └── preprocessor.py
├── FeatureExtraction/           # FVMD and TEO
│   ├── vmd.py
│   ├── teo.py
│   └── extract_features.py
├── FuzzyLogic/                  # Fuzzy Logic Controller (81 rules)
│   └── fuzzy_system.py
├── DeepLearning/                # VAE-DCCNN-Att model
│   └── model.py
├── FederatedLearning/           # Multi-Krum and GAT detection
│   ├── multi_krum.py
│   └── federated_sim.py
├── EventDetection/              # Event-based load detection
│   └── detect_events.py
├── Evaluation/                  # Evaluation framework
│   └── evaluation_framework.py
├── Advanced/                    # Attention mechanism, signal quality
│   ├── attention.py
│   └── signal_quality.py
├── Streaming/                   # Real-time data streaming
│   └── data_stream.py
├── OnlineLearning/              # Online learning + concept drift
│   ├── online_learner.py
│   └── concept_drift.py
├── MultiHousehold/              # Multi-household simulation
│   └── simulator.py
├── TransferLearning/            # Transfer learning capabilities
│   └── manager.py
├── Export/                      # Result export (LaTeX, CSV, JSON)
│   └── exporter.py
├── API/                         # REST API server
│   └── server.py
├── Tests/                       # Testing framework
│   └── run_tests.py
├── Utils/                       # Helper functions
│   └── helpers.py
└── Results/                     # Output plots and metrics
```

## Core Components

### 1. Preprocessing
- Z-score outlier removal
- Savitzky-Golay smoothing
- Min-Max normalization

### 2. Feature Extraction
- **FVMD**: Feedback Variational Mode Decomposition with automatic K selection
- **TEO**: Teager Energy Operator for transient detection
- **Features**: Time-domain, frequency-domain, TEO, cross-mode correlation

### 3. Fuzzy Logic Controller
- 4 inputs: DeltaP, Sigma, Duration, Frequency
- 3 membership functions per input (Mamdani FIS)
- 81 rules (3^4 combinations)
- Output: OFF / PARTIAL / ON state

### 4. Deep Learning (VAE-DCCNN-Att)
- Variational Autoencoder with dilated causal convolutions
- Multi-head self-attention mechanism
- Classification head for appliance identification
- PyTorch-compatible architecture

### 5. Federated Learning
- Multi-Krum aggregation (Byzantine-robust)
- GAT-based Byzantine detection
- Configurable number of clients and rounds

### 6. Event Detection
- Edge-based on/off event detection
- Adaptive thresholding
- Appliance classification by power/duration

### 7. Evaluation Framework
- Classification: accuracy, precision, recall, F1
- Regression: MAE, RMSE, MAPE
- Event-based metrics with tolerance matching
- Reconstruction: MSE, MAE, SNR, R2

### 8. GUI Dashboard
- 9-panel interactive visualization
- Real-time pipeline execution
- Federated learning results plotting
- Signal quality assessment

## Usage

### Run Full Pipeline
```python
from PyNILM.main import NILMApp
# Or run: python main.py
```

### Programmatic Usage
```python
from data.data_loader import load_ukdale
from preprocessing.preprocessor import preprocess_signal
from feature_extraction.vmd import feedback_vmd
from feature_extraction.teo import teager_energy_operator
from feature_extraction.extract_features import extract_features
from deep_learning.model import VAEDCCNNAtt
from evaluation.evaluation_framework import EvaluationFramework

# Load data
data = load_ukdale(home_id=1, config={'windowSize': 256})

# Preprocess
result = preprocess_signal(data['testData'][0], {})

# Extract features
modes, omega, _, _ = feedback_vmd(result['normalized'])
teo = teager_energy_operator(modes[0])
features, names = extract_features(modes, teo.reshape(1, -1))

# Classify
model = VAEDCCNNAtt({'numClasses': 5})
prediction = model.predict(features.reshape(1, -1))
```

### Run Tests
```python
from tests.run_tests import run_all_tests
run_all_tests()
```

### Federated Learning
```python
from federated_learning.federated_sim import run_federated_learning
config = {'numClients': 20, 'numRounds': 50, 'byzantineFraction': 0.1}
results = run_federated_learning(config)
```

## Configuration
Default configuration is in `config/default_config.json`. All parameters are validated on load.

## Requirements
- Python 3.8+
- numpy, scipy, scikit-learn
- matplotlib (optional, for visualization)
- Flask (optional, for REST API)
