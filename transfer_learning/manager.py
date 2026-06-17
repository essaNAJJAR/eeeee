import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from deep_learning.model import VAEDCCNNAtt


class TransferLearningManager:
    def __init__(self, source_model=None):
        self.source_model = source_model
        self.target_model = None
        self.frozen_layers = []

    def load_source_model(self, model_path):
        self.source_model = VAEDCCNNAtt.load(model_path)
        return self.source_model

    def fine_tune(self, target_data, target_labels, config=None):
        if config is None:
            config = {}
        epochs = config.get('epochs', 20)
        batch_size = config.get('batchSize', 32)
        learning_rate = config.get('learningRate', 0.001)
        freeze_layers = config.get('freezeLayers', ['encoder'])

        if self.source_model is None:
            self.target_model = VAEDCCNNAtt(config)
        else:
            self.target_model = self.source_model

        self.frozen_layers = freeze_layers

        history = {'loss': [], 'accuracy': []}

        for epoch in range(epochs):
            indices = np.random.permutation(len(target_data))
            epoch_loss = 0
            epoch_correct = 0

            for start in range(0, len(target_data), batch_size):
                end = min(start + batch_size, len(target_data))
                batch_idx = indices[start:end]
                x_batch = target_data[batch_idx]
                y_batch = target_labels[batch_idx]

                result = self.target_model.forward(x_batch, training=True)
                probs = result['probabilities']

                n_classes = probs.shape[1]
                y_onehot = np.zeros((len(y_batch), n_classes))
                for i, y in enumerate(y_batch):
                    if 0 <= y < n_classes:
                        y_onehot[i, int(y)] = 1

                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=-1))
                preds = np.argmax(probs, axis=-1)

                if 'encoder' not in self.frozen_layers:
                    lr = learning_rate * 0.1
                    for conv in self.target_model.encoder.conv_layers:
                        conv.weight -= lr * 0.01 * conv.weight

                if 'classifier' not in self.frozen_layers:
                    self.target_model.classifier.fc1 -= learning_rate * 0.01 * self.target_model.classifier.fc1

                epoch_loss += loss
                epoch_correct += np.sum(preds == y_batch)

            n_batches = max(1, len(target_data) // batch_size)
            history['loss'].append(epoch_loss / n_batches)
            history['accuracy'].append(epoch_correct / len(target_data))

        return self.target_model, history

    def augment_data(self, data, noise_level=0.01, time_shift=5, scale_range=(0.9, 1.1)):
        augmented = []
        for signal in data:
            augmented.append(signal)
            noisy = signal + np.random.randn(len(signal)) * noise_level
            augmented.append(noisy)
            shifted = np.roll(signal, np.random.randint(-time_shift, time_shift))
            augmented.append(shifted)
            scaled = signal * np.random.uniform(*scale_range)
            augmented.append(scaled)
        return np.array(augmented)

    def extract_features(self, data):
        if self.source_model is None or self.target_model is None:
            return None
        source_features = self.source_model.predict(data)['mu']
        target_features = self.target_model.predict(data)['mu']
        return {'source': source_features, 'target': target_features}

    def save_model(self, path):
        if self.target_model:
            self.target_model.save(path)

    def load_model(self, path):
        self.target_model = VAEDCCNNAtt.load(path)
        return self.target_model
