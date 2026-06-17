import numpy as np


class AttentionMechanism:
    def __init__(self, dim, num_heads):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        std = np.sqrt(2.0 / dim)
        self.W_q = np.random.randn(dim, dim) * std
        self.W_k = np.random.randn(dim, dim) * std
        self.W_v = np.random.randn(dim, dim) * std
        self.W_o = np.random.randn(dim, dim) * std

    def forward(self, x):
        batch, seq_len, _ = x.shape
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        Q = Q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)

        context = np.matmul(attn, V)
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.dim)
        return context @ self.W_o
