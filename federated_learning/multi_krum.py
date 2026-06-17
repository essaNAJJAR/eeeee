import numpy as np


def multi_krum(client_updates, f):
    n = len(client_updates)
    if n <= f:
        return _average_updates(client_updates)

    flat_updates = []
    for update in client_updates:
        flat = np.concatenate([v.flatten() for v in update])
        flat_updates.append(flat)

    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(flat_updates[i] - flat_updates[j])
            distances[i, j] = d
            distances[j, i] = d

    k = max(1, n - f - 1)
    scores = np.zeros(n)
    for i in range(n):
        sorted_dists = np.sort(distances[i])
        scores[i] = np.sum(sorted_dists[1:k + 1])

    trusted_indices = np.argsort(scores)[:n - f]
    trusted_updates = [client_updates[i] for i in trusted_indices]
    return _average_updates(trusted_updates)


def _average_updates(updates):
    if not updates:
        return None
    avg = [np.zeros_like(v) for v in updates[0]]
    for update in updates:
        for i, v in enumerate(update):
            avg[i] += v
    for i in range(len(avg)):
        avg[i] /= len(updates)
    return avg


def detect_byzantine_gat(client_updates, threshold=0.5):
    n = len(client_updates)
    if n < 3:
        return list(range(n)), []

    flat_updates = []
    for update in client_updates:
        flat = np.concatenate([v.flatten() for v in update])
        flat_updates.append(flat)

    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            norm_i = np.linalg.norm(flat_updates[i])
            norm_j = np.linalg.norm(flat_updates[j])
            if norm_i > 0 and norm_j > 0:
                sim = np.dot(flat_updates[i], flat_updates[j]) / (norm_i * norm_j)
            else:
                sim = 0
            similarity[i, j] = sim
            similarity[j, i] = sim

    adj = (similarity > threshold).astype(float)
    np.fill_diagonal(adj, 1)

    centrality = np.zeros(n)
    for i in range(n):
        connected = np.sum(adj[i])
        centrality[i] = connected

    mean_c = np.mean(centrality)
    std_c = np.std(centrality)
    suspicious = []
    normal = []
    for i in range(n):
        if centrality[i] < mean_c - 2 * std_c:
            suspicious.append(i)
        else:
            normal.append(i)

    return normal, suspicious


class GATDetector:
    def __init__(self, threshold=0.5, num_heads=4, hidden_dim=64):
        self.threshold = threshold
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

    def detect(self, client_updates):
        return detect_byzantine_gat(client_updates, self.threshold)
