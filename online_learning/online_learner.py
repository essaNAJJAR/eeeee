import numpy as np
from .concept_drift import ConceptDriftDetector


class OnlineLearner:
    def __init__(self, learning_rate=0.001, buffer_size=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_X = []
        self.buffer_y = []
        self.model_weights = None
        self.drift_detector = ConceptDriftDetector()
        self.performance_history = []
        self.total_updates = 0
        self.drifts_detected = 0

    def initialize_model(self, input_dim, num_classes):
        std = np.sqrt(2.0 / input_dim)
        self.model_weights = {
            'W1': np.random.randn(input_dim, 64) * std,
            'b1': np.zeros(64),
            'W2': np.random.randn(64, num_classes) * np.sqrt(2.0 / 64),
            'b2': np.zeros(num_classes),
        }

    def _forward(self, X):
        h = np.maximum(0, X @ self.model_weights['W1'] + self.model_weights['b1'])
        logits = h @ self.model_weights['W2'] + self.model_weights['b2']
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

    def update(self, X_new, y_new):
        if self.model_weights is None:
            self.initialize_model(X_new.shape[1], len(np.unique(y_new)))

        for i in range(len(X_new)):
            self.buffer_X.append(X_new[i])
            self.buffer_y.append(y_new[i])
        if len(self.buffer_X) > self.buffer_size:
            self.buffer_X = self.buffer_X[-self.buffer_size:]
            self.buffer_y = self.buffer_y[-self.buffer_size:]

        self.drift_detector.update(X_new)

        if self.drift_detector.detect():
            self.drifts_detected += 1
            self._retrain()

        self._mini_batch_update(X_new, y_new)
        self.total_updates += len(X_new)

    def _mini_batch_update(self, X, y):
        probs = self._forward(X)
        n_classes = probs.shape[1]
        y_onehot = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            if 0 <= label < n_classes:
                y_onehot[i, int(label)] = 1

        grad_logits = probs - y_onehot
        h = np.maximum(0, X @ self.model_weights['W1'] + self.model_weights['b1'])

        self.model_weights['W2'] -= self.learning_rate * (h.T @ grad_logits) / len(X)
        self.model_weights['b2'] -= self.learning_rate * np.mean(grad_logits, axis=0)

        grad_h = grad_logits @ self.model_weights['W2'].T
        grad_h[h <= 0] = 0
        self.model_weights['W1'] -= self.learning_rate * (X.T @ grad_h) / len(X)
        self.model_weights['b1'] -= self.learning_rate * np.mean(grad_h, axis=0)

        loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=-1))
        accuracy = np.mean(np.argmax(probs, axis=-1) == y)
        self.performance_history.append({'loss': loss, 'accuracy': accuracy})

    def _retrain(self):
        if len(self.buffer_X) < self.batch_size:
            return
        X = np.array(self.buffer_X)
        y = np.array(self.buffer_y)
        indices = np.random.permutation(len(X))
        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            idx = indices[start:end]
            self._mini_batch_update(X[idx], y[idx])

    def predict(self, X):
        if self.model_weights is None:
            return np.zeros(len(X), dtype=int)
        probs = self._forward(X)
        return np.argmax(probs, axis=-1)

    def get_performance(self):
        return self.performance_history
