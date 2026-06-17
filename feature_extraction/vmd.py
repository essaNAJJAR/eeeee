import numpy as np


def vmd(signal, alpha=2000, tau=0, K=6, DC=False, tol=1e-7, N_iter=500):
    T = len(signal)
    t = np.arange(T) / T
    T2 = T // 2

    freqs = t - 0.5 - 1.0 / T

    f_hat = np.fft.fftshift(np.fft.fft(signal))
    f_hat_plus = f_hat.copy()
    f_hat_plus[:T2 + 1] = 0

    u_hat_plus = np.zeros((N_iter, len(freqs), K), dtype=complex)
    omega_plus = np.zeros((N_iter, K))

    for k in range(K):
        omega_plus[0, k] = (0.5 / K) * k
    if DC:
        omega_plus[0, 0] = 0

    lambda_hat = np.zeros((N_iter, len(freqs)), dtype=complex)
    uDiff = tol + np.finfo(float).eps

    n = 0
    while uDiff > tol and n < N_iter - 1:
        for k in range(K):
            # Gauss-Seidel update scheme for component summation
            sum_of_ug = np.sum(u_hat_plus[n + 1, :, :k], axis=1) + np.sum(u_hat_plus[n, :, k + 1:], axis=1)
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_of_ug - lambda_hat[n, :]) / \
                (1 + alpha * (freqs - omega_plus[n, k]) ** 2)

            if not DC:
                numerator = np.sum(
                    freqs[T2 + 1:] * np.abs(u_hat_plus[n + 1, T2 + 1:, k]) ** 2
                )
                denominator = np.sum(np.abs(u_hat_plus[n + 1, T2 + 1:, k]) ** 2)
                if denominator > 0:
                    omega_plus[n + 1, k] = numerator / denominator
                else:
                    omega_plus[n + 1, k] = omega_plus[n, k]
            else:
                omega_plus[n + 1, k] = omega_plus[n, k]

        lambda_hat[n + 1, :] = lambda_hat[n, :] + tau * (
            np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus
        )

        n += 1
        uDiff = 0
        for k in range(K):
            uDiff += (1.0 / T) * np.sum(
                np.abs(u_hat_plus[n, :, k] - u_hat_plus[n - 1, :, k]) ** 2
            )
        uDiff = np.abs(uDiff)

    N_iter_actual = min(N_iter, n)
    omega = omega_plus[N_iter_actual - 1, :]

    u_hat = np.zeros((K, T), dtype=complex)
    mid = T2 + 1
    if mid < T:
        u_hat[:, mid:] = u_hat_plus[N_iter_actual - 1, mid:, :].T
    for k in range(K):
        for idx in range(1, mid):
            if T - idx >= 0 and T - idx < T:
                u_hat[k, idx] = np.conj(u_hat[k, T - idx])

    if T % 2 == 0:
        u_hat[:, mid - 1] = 0
    else:
        if mid < T:
            u_hat[:, mid - 1] = np.conj(u_hat[:, mid])

    u_hat[:, 0] = np.conj(u_hat[:, -1])

    u = np.zeros((K, T))
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[k, :])))

    return u, u_hat, omega


def calculate_wse(u_hat):
    T = u_hat.shape[1] if u_hat.ndim > 1 else len(u_hat)
    spectrum = np.abs(u_hat)
    total = np.sum(spectrum)
    if total == 0:
        return 0
    probs = spectrum / total
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = np.log2(len(probs) + 1e-12)
    return 1.0 - entropy / max_entropy if max_entropy > 0 else 0


def calculate_fhi(u_hat, fundamental_freq=0.1):
    T = u_hat.shape[1] if u_hat.ndim > 1 else len(u_hat)
    spectrum = np.abs(u_hat).flatten()
    freqs = np.fft.fftfreq(T)
    freqs = np.fft.fftshift(freqs)

    energy = np.sum(spectrum ** 2)
    if energy == 0:
        return 0

    harmonic_energy = 0
    for h in range(1, 6):
        target = fundamental_freq * h
        idx = np.argmin(np.abs(freqs - target))
        window = max(0, idx - 2), min(T, idx + 3)
        harmonic_energy += np.sum(spectrum[window[0]:window[1]] ** 2)

    return harmonic_energy / energy


def calculate_snr(signal, modes):
    reconstruction = np.sum(modes, axis=0)
    noise = signal - reconstruction
    signal_power = np.mean(reconstruction ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return 100.0
    return 10 * np.log10(signal_power / noise_power)


def feedback_vmd(signal, K_range=None, alpha=2000, tau=0, tol=1e-7, max_iter=500):
    if K_range is None:
        K_range = [3, 9]

    best_score = -np.inf
    best_K = K_range[0]
    best_modes = None
    best_omega = None

    for K in range(K_range[0], K_range[1] + 1):
        modes, u_hat, omega = vmd(signal, alpha, tau, K, False, tol, max_iter)

        wse = 0
        fhi_vals = []
        for k in range(K):
            wse += calculate_wse(u_hat[k, :])
            fhi_vals.append(calculate_fhi(u_hat[k, :]))
        wse /= K
        fhi = np.mean(fhi_vals)
        snr = calculate_snr(signal, modes)

        score = 0.4 * wse + 0.3 * fhi + 0.3 * (1.0 / (1.0 + np.exp(-snr / 10)))

        if score > best_score:
            best_score = score
            best_K = K
            best_modes = modes
            best_omega = omega

    return best_modes, best_omega, best_score, best_K
