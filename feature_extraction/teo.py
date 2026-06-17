import numpy as np


def teager_energy_operator(signal):
    n = len(signal)
    if n == 0:
        return np.zeros(0)
    teo = np.zeros(n)
    teo[0] = signal[0] ** 2
    for i in range(1, n - 1):
        teo[i] = signal[i] ** 2 - signal[i - 1] * signal[i + 1]
    if n > 1:
        teo[-1] = signal[-1] ** 2
    return np.abs(teo)
