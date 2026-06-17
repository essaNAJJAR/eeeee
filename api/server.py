import json
import os
import sys
import numpy as np
from preprocessing.preprocessor import preprocess_signal
from feature_extraction.vmd import feedback_vmd
from feature_extraction.teo import teager_energy_operator
from feature_extraction.extract_features import extract_features
from deep_learning.model import VAEDCCNNAtt


class NILMAPIServer:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.port = config.get('port', 8080)
        self.host = config.get('host', 'localhost')
        self.model = None
        self.stats = {'requests': 0, 'predictions': 0, 'errors': 0}

    def start(self):
        try:
            from flask import Flask, request, jsonify
            app = Flask(__name__)

            @app.route('/api/health', methods=['GET'])
            def health():
                return jsonify({'status': 'healthy', 'model_loaded': self.model is not None})

            @app.route('/api/predict', methods=['POST'])
            def predict():
                self.stats['requests'] += 1
                try:
                    data = request.get_json()
                    signal = np.array(data.get('signal', []))
                    if len(signal) == 0:
                        return jsonify({'error': 'No signal provided'}), 400

                    preprocessed = preprocess_signal(signal, {})
                    modes, omega, _, _ = feedback_vmd(preprocessed['normalized'])
                    teo = np.zeros_like(modes)
                    for k in range(modes.shape[0]):
                        teo[k] = teager_energy_operator(modes[k])
                    features, _ = extract_features(modes, teo)

                    if self.model is None:
                        self.model = VAEDCCNNAtt()

                    result = self.model.predict(features.reshape(1, -1))
                    self.stats['predictions'] += 1

                    return jsonify({
                        'predictions': result['predictions'].tolist(),
                        'probabilities': result['probabilities'].tolist(),
                    })
                except Exception as e:
                    self.stats['errors'] += 1
                    return jsonify({'error': str(e)}), 500

            @app.route('/api/status', methods=['GET'])
            def status():
                return jsonify({
                    'stats': self.stats,
                    'host': self.host,
                    'port': self.port,
                })

            print(f"API Server starting on {self.host}:{self.port}")
            app.run(host=self.host, port=self.port, debug=False)

        except ImportError:
            print("Flask not installed. Install with: pip install flask")
            return False

    def get_stats(self):
        return dict(self.stats)
