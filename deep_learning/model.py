import numpy as np
import json
import os


def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        return torch, nn, optim, DataLoader, TensorDataset
    except ImportError:
        return None


def _build_pytorch_classes(torch, nn):
    class PyTorchVAEDCCNN(nn.Module):
        def __init__(self, num_classes=5, latent_dim=32, enc_channels=None, dec_channels=None,
                     input_length=256, vae_beta=0.5):
            super().__init__()
            if enc_channels is None:
                enc_channels = [64, 128, 256]
            if dec_channels is None:
                dec_channels = [256, 128, 64]
            self.latent_dim = latent_dim
            self.num_classes = num_classes
            self.vae_beta = vae_beta
            self.input_length = input_length

            self.encoder_convs = nn.Sequential()
            in_ch = 1
            for i, out_ch in enumerate(enc_channels):
                self.encoder_convs.add_module(
                    f'conv{i}',
                    nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, dilation=1)
                )
                self.encoder_convs.add_module(f'relu{i}', nn.ReLU())
                in_ch = out_ch

            self.fc_mu = nn.Linear(enc_channels[-1], latent_dim)
            self.fc_logvar = nn.Linear(enc_channels[-1], latent_dim)

            self.attn_norm = nn.LayerNorm(enc_channels[-1])
            self.attn = nn.MultiheadAttention(enc_channels[-1], num_heads=8, batch_first=True)

            self.decoder_fc = nn.Linear(latent_dim, dec_channels[0])
            self.decoder_convs = nn.Sequential()
            for i, out_ch in enumerate(dec_channels):
                self.decoder_convs.add_module(
                    f'conv{i}',
                    nn.Conv1d(dec_channels[i - 1] if i > 0 else dec_channels[0],
                             out_ch, kernel_size=3, padding=1)
                )
                self.decoder_convs.add_module(f'relu{i}', nn.ReLU())
            self.decoder_out = nn.Conv1d(dec_channels[-1], 1, kernel_size=3, padding=1)

            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )

        def encode(self, x):
            h = self.encoder_convs(x)
            pooled = h.mean(dim=-1)
            mu = self.fc_mu(pooled)
            logvar = self.fc_logvar(pooled)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            h = self.decoder_fc(z).unsqueeze(-1)
            h = h.expand(-1, -1, self.input_length)
            h = self.decoder_convs(h)
            return self.decoder_out(h)

        def classify(self, z):
            return self.classifier(z)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            logits = self.classify(z)
            recon = self.decode(z)
            return logits, mu, logvar, recon

    class PyTorchModelWrapper:
        def __init__(self, pt_model, num_classes, device):
            self.pt_model = pt_model
            self.num_classes = num_classes
            self.device = device
            self.latent_dim = pt_model.latent_dim

        def predict(self, x):
            self.pt_model.eval()
            with torch.no_grad():
                mean = getattr(self, 'mean', 0.0)
                std = getattr(self, 'std', 1.0)
                if isinstance(x, np.ndarray):
                    x_norm = (x - mean) / std
                    x_tensor = torch.FloatTensor(x_norm)
                else:
                    x_arr = np.array(x)
                    x_norm = (x_arr - mean) / std
                    x_tensor = torch.FloatTensor(x_norm)
                if x_tensor.dim() == 2:
                    x_tensor = x_tensor.unsqueeze(1)
                x_tensor = x_tensor.to(self.device)
                logits, mu, logvar, recon = self.pt_model(x_tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                predictions = logits.argmax(dim=-1).cpu().numpy()
            return {
                'predictions': predictions,
                'probabilities': probs,
                'logits': logits.cpu().numpy(),
                'mu': mu.cpu().numpy(),
                'logvar': logvar.cpu().numpy(),
                'reconstruction': recon.cpu().numpy(),
            }

        def forward(self, x, training=True):
            return self.predict(x)

        def save(self, path):
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            state = {
                'state_dict': self.pt_model.state_dict(),
                'mean': getattr(self, 'mean', 0.0),
                'std': getattr(self, 'std', 1.0),
            }
            torch.save(state, path)

        @classmethod
        def load(cls, path, num_classes=5, latent_dim=32, enc_channels=None, dec_channels=None,
                 input_length=256):
            if enc_channels is None:
                enc_channels = [64, 128, 256]
            if dec_channels is None:
                dec_channels = [256, 128, 64]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = PyTorchVAEDCCNN(
                num_classes=num_classes, latent_dim=latent_dim,
                enc_channels=enc_channels, dec_channels=dec_channels,
                input_length=input_length,
            ).to(device)
            checkpoint = torch.load(path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                wrapper = cls(model, num_classes, device)
                wrapper.mean = checkpoint.get('mean', 0.0)
                wrapper.std = checkpoint.get('std', 1.0)
                return wrapper
            else:
                model.load_state_dict(checkpoint)
                return cls(model, num_classes, device)

    return PyTorchVAEDCCNN, PyTorchModelWrapper


class DilatedCausalConv1d:
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        receptive = (kernel_size - 1) * dilation
        fan_in = in_channels * kernel_size
        fan_out = out_channels * kernel_size
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.weight = np.random.randn(out_channels, in_channels, kernel_size) * std
        self.bias = np.zeros(out_channels)
        self.padding = receptive

    def forward(self, x):
        batch, channels, length = x.shape
        padded = np.zeros((batch, channels, length + self.padding))
        padded[:, :, self.padding:] = x
        out = np.zeros((batch, self.out_channels, length))
        for t in range(length):
            window = padded[:, :, t:t + self.kernel_size * self.dilation:self.dilation]
            for oc in range(self.out_channels):
                out[:, oc, t] = np.sum(window * self.weight[oc], axis=(1, 2)) + self.bias[oc]
        return out


class MultiHeadAttention:
    def __init__(self, dim, num_heads):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        std = np.sqrt(2.0 / dim)
        self.W_q = np.random.randn(dim, dim) * std
        self.W_k = np.random.randn(dim, dim) * std
        self.W_v = np.random.randn(dim, dim) * std
        self.W_o = np.random.randn(dim, dim) * std
        self.b_q = np.zeros(dim)
        self.b_k = np.zeros(dim)
        self.b_v = np.zeros(dim)
        self.b_o = np.zeros(dim)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v

        Q = Q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)

        context = np.matmul(attn, V)
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.dim)
        output = context @ self.W_o + self.b_o
        return output


class VAEEncoder:
    def __init__(self, input_dim, channels, latent_dim):
        self.conv_layers = []
        in_ch = 1
        for out_ch in channels:
            self.conv_layers.append(DilatedCausalConv1d(in_ch, out_ch, kernel_size=3, dilation=1))
            in_ch = out_ch

        self.fc_mu = np.random.randn(channels[-1], latent_dim) * 0.01
        self.fc_logvar = np.random.randn(channels[-1], latent_dim) * 0.01
        self.mu_bias = np.zeros(latent_dim)
        self.logvar_bias = np.zeros(latent_dim)

    def forward(self, x):
        h = x
        for conv in self.conv_layers:
            h = np.maximum(0, conv.forward(h))

        pooled = np.mean(h, axis=-1)
        mu = pooled @ self.fc_mu + self.mu_bias
        logvar = pooled @ self.fc_logvar + self.logvar_bias
        return mu, logvar, h

    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std


class VAEDecoder:
    def __init__(self, latent_dim, channels, output_dim):
        self.output_dim = output_dim
        self.fc = np.random.randn(latent_dim, channels[0]) * 0.01
        self.fc_bias = np.zeros(channels[0])
        self.conv_t_layers = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.conv_t_layers.append(DilatedCausalConv1d(in_ch, out_ch, kernel_size=3, dilation=1))
            in_ch = out_ch
        self.output_conv = DilatedCausalConv1d(channels[-1], 1, kernel_size=3, dilation=1)

    def forward(self, z):
        h = z @ self.fc + self.fc_bias
        h = h[:, :, np.newaxis]
        h = np.tile(h, (1, 1, self.output_dim))
        for conv in self.conv_t_layers:
            h = np.maximum(0, conv.forward(h))
        out = self.output_conv.forward(h)
        return out


class ClassificationHead:
    def __init__(self, input_dim, hidden_dims, num_classes):
        self.fc1 = np.random.randn(input_dim, hidden_dims[0]) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dims[0])
        self.fc2 = np.random.randn(hidden_dims[0], hidden_dims[1]) * np.sqrt(2.0 / hidden_dims[0])
        self.b2 = np.zeros(hidden_dims[1])
        self.fc3 = np.random.randn(hidden_dims[1], num_classes) * np.sqrt(2.0 / hidden_dims[1])
        self.b3 = np.zeros(num_classes)
        self.dropout_rate = 0.3

    def forward(self, x, training=True):
        h = np.maximum(0, x @ self.fc1 + self.b1)
        if training:
            mask = (np.random.rand(*h.shape) > self.dropout_rate).astype(float)
            h = h * mask / (1 - self.dropout_rate + 1e-12)
        h = np.maximum(0, h @ self.fc2 + self.b2)
        logits = h @ self.fc3 + self.b3
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)
        return logits, probs


class VAEDCCNNAtt:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.latent_dim = config.get('latentDim', 32)
        self.num_classes = config.get('numClasses', 5)
        enc_channels = config.get('encoderChannels', [64, 128, 256])
        dec_channels = config.get('decoderChannels', [256, 128, 64])
        input_dim = config.get('inputDim', 256)
        attn_dim = config.get('attention', {}).get('dim', 64)
        attn_heads = config.get('attention', {}).get('heads', 8)

        self.encoder = VAEEncoder(input_dim, enc_channels, self.latent_dim)
        self.attention = MultiHeadAttention(enc_channels[-1], attn_heads)
        self.decoder = VAEDecoder(self.latent_dim, dec_channels, input_dim)
        self.classifier = ClassificationHead(self.latent_dim, [128, 64], self.num_classes)
        self.mean = config.get('mean', 0.0)
        self.std = config.get('std', 1.0)

    def forward(self, x, training=True):
        x_norm = (x - self.mean) / self.std
        if x_norm.ndim == 2:
            x_norm = x_norm[:, np.newaxis, :]
        mu, logvar, enc_out = self.encoder.forward(x_norm)
        z = self.encoder.reparameterize(mu, logvar)
        logits, probs = self.classifier.forward(z, training)
        recon = self.decoder.forward(z)
        recon_unnorm = recon * self.std + self.mean
        return {
            'logits': logits,
            'probabilities': probs,
            'predictions': np.argmax(probs, axis=-1),
            'mu': mu,
            'logvar': logvar,
            'reconstruction': recon_unnorm,
        }

    def predict(self, x):
        return self.forward(x, training=False)

    def save(self, path):
        params = {
            'latent_dim': self.latent_dim,
            'num_classes': self.num_classes,
            'mean': self.mean,
            'std': self.std,
            'encoder_weights': [c.weight.tolist() for c in self.encoder.conv_layers],
            'encoder_biases': [c.bias.tolist() for c in self.encoder.conv_layers],
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            params = json.load(f)
        model = cls({
            'latentDim': params['latent_dim'],
            'numClasses': params['num_classes'],
            'mean': params.get('mean', 0.0),
            'std': params.get('std', 1.0)
        })
        return model


def predict_with_model(model, features):
    if isinstance(features, np.ndarray) and features.ndim == 1:
        features = features.reshape(1, -1)
    result = model.predict(features)
    return result['predictions'], result['probabilities']


def train_vae_dccnn_model(train_data, train_labels, config):
    torch_mod = _try_import_torch()

    if torch_mod is not None:
        return _train_with_pytorch(train_data, train_labels, config, torch_mod)
    else:
        return _train_numpy_fallback(train_data, train_labels, config)


def _train_with_pytorch(train_data, train_labels, config, torch_mod):
    torch, nn, optim, DataLoader, TensorDataset = torch_mod
    PyTorchVAEDCCNN, PyTorchModelWrapper = _build_pytorch_classes(torch, nn)

    num_classes = config.get('numClasses', 5)
    latent_dim = config.get('latentDim', 32)
    enc_channels = config.get('encoderChannels', [64, 128, 256])
    dec_channels = config.get('decoderChannels', [256, 128, 64])
    batch_size = config.get('batchSize', 32)
    epochs = config.get('epochs', 100)
    lr = config.get('learningRate', 0.001)
    vae_beta = config.get('vaeBeta', 0.5)
    weight_decay = config.get('weightDecay', 1e-5)

    input_length = train_data.shape[1]

    max_label = int(np.max(train_labels)) if len(train_labels) > 0 else 0
    num_classes = max(num_classes, max_label + 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training VAE-DCCNN-Att on {device} for {epochs} epochs")

    mean = float(np.mean(train_data))
    std = float(np.std(train_data))
    if std == 0:
        std = 1.0
    train_data_norm = (train_data - mean) / std

    X_tensor = torch.FloatTensor(train_data_norm).unsqueeze(1).to(device)
    y_tensor = torch.LongTensor(train_labels).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PyTorchVAEDCCNN(
        num_classes=num_classes, latent_dim=latent_dim,
        enc_channels=enc_channels, dec_channels=dec_channels,
        input_length=input_length, vae_beta=vae_beta,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce_loss_fn = nn.CrossEntropyLoss()

    history = {'loss': [], 'accuracy': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()

            logits, mu, logvar, recon = model(X_batch)

            ce_loss = ce_loss_fn(logits, y_batch)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
            recon_loss = nn.functional.mse_loss(recon.squeeze(1), X_batch.squeeze(1))
            loss = ce_loss + vae_beta * kl_loss + 0.1 * recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == y_batch).sum().item()
            epoch_total += y_batch.size(0)
            n_batches += 1

        scheduler.step()
        history['loss'].append(epoch_loss / max(n_batches, 1))
        history['accuracy'].append(epoch_correct / max(epoch_total, 1))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: loss={history['loss'][-1]:.4f}, "
                  f"acc={history['accuracy'][-1]:.4f}")

    wrapper = PyTorchModelWrapper(model, num_classes, device)
    wrapper.mean = mean
    wrapper.std = std
    return wrapper, history


def _train_numpy_fallback(train_data, train_labels, config):
    print("PyTorch not available. Using numpy fallback (limited training).")
    mean = float(np.mean(train_data))
    std = float(np.std(train_data))
    if std == 0:
        std = 1.0

    model = VAEDCCNNAtt(config)
    model.mean = mean
    model.std = std
    n_samples = train_data.shape[0]
    batch_size = config.get('batchSize', 32)
    epochs = config.get('epochs', 100)
    lr = config.get('learningRate', 0.001)
    vae_beta = config.get('vaeBeta', 0.5)

    history = {'loss': [], 'accuracy': []}
    n_classes = model.num_classes

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        epoch_loss = 0
        epoch_correct = 0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_idx = indices[start:end]
            x_batch = train_data[batch_idx]
            y_batch = train_labels[batch_idx]

            result = model.forward(x_batch, training=True)

            logits = result['logits']
            mu = result['mu']
            logvar = result['logvar']

            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            softmax = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-12)

            y_onehot = np.zeros((len(y_batch), n_classes))
            for i, y in enumerate(y_batch):
                if 0 <= y < n_classes:
                    y_onehot[i, int(y)] = 1

            ce_loss = -np.mean(np.sum(y_onehot * np.log(softmax + 1e-12), axis=-1))
            kl_loss = -0.5 * np.mean(np.sum(1 + logvar - mu ** 2 - np.log(np.exp(logvar) + 1e-12), axis=-1))
            loss = ce_loss + vae_beta * kl_loss

            preds = np.argmax(softmax, axis=-1)
            epoch_correct += np.sum(preds == y_batch)
            epoch_loss += loss

            grad_logits = softmax - y_onehot
            z = result['mu']

            h1 = np.maximum(0, z @ model.classifier.fc1 + model.classifier.b1)
            h2 = np.maximum(0, h1 @ model.classifier.fc2 + model.classifier.b2)

            model.classifier.fc3 -= lr * (h2.T @ grad_logits) / len(y_batch)
            model.classifier.b3 -= lr * np.mean(grad_logits, axis=0)

            grad_h2 = grad_logits @ model.classifier.fc3.T
            grad_h2[h2 <= 0] = 0
            model.classifier.fc2 -= lr * (h1.T @ grad_h2) / len(y_batch)
            model.classifier.b2 -= lr * np.mean(grad_h2, axis=0)

            grad_h1 = grad_h2 @ model.classifier.fc2.T
            grad_h1[h1 <= 0] = 0
            model.classifier.fc1 -= lr * (z.T @ grad_h1) / len(y_batch)
            model.classifier.b1 -= lr * np.mean(grad_h1, axis=0)

            n_batches += 1

        history['loss'].append(epoch_loss / max(n_batches, 1))
        history['accuracy'].append(epoch_correct / n_samples)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: loss={history['loss'][-1]:.4f}, "
                  f"acc={history['accuracy'][-1]:.4f}")

    return model, history
