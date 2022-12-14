import math
import numpy as np
import scipy.stats as st
import scipy.signal as si
from scipy.integrate import simps

# x_raw - raw signal data for 1 trial with shape N_ch x T
# el_pos_3d - matrix of 3D-coordinates of electrodes

# seed sf = 200
# mi sf = 250


def euclidean_dist_adjacency(el_pos_3d: np.array) -> np.array:
    n_ch = len(el_pos_3d)
    dist_matrix = np.zeros((n_ch, n_ch))

    for i in range(n_ch):
        for j in range(i):
            dist_matrix[i][j] = 1 / np.linalg.norm(el_pos_3d[i] - el_pos_3d[j])

    dist_matrix += dist_matrix.T
    dist_matrix /= dist_matrix.max()
    return dist_matrix


def geodesic_dist_adjacency(el_pos_3d: np.array) -> np.array:
    el_pos_3d /= np.linalg.norm(el_pos_3d.astype(float), axis=1, keepdims=True)
    n_ch = len(el_pos_3d)
    dist_matrix = np.zeros((n_ch, n_ch))

    r = 1
    for i in range(n_ch):
        for j in range(i):
            dist_matrix[i][j] = 1 / (r * math.acos(round(np.dot(el_pos_3d[i], el_pos_3d[j]) / (r ** 2), 2)))

    dist_matrix += dist_matrix.T
    dist_matrix /= dist_matrix.max()
    return dist_matrix


def pearson_corr_adjacency(x):
    n_ch = x.shape[0]
    corr_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i):
            corr_matrix[i][j] = st.pearsonr(x[i, :], x[j, :])[0]
    corr_matrix += corr_matrix.T
    corr_matrix = np.abs(corr_matrix)
    return corr_matrix


def calculate_coherence(x, y, sf, low_freq, hi_freq):
    nperseg = (2 / low_freq) * sf
    freqs, coh = si.coherence(x, y, fs=sf, nperseg=nperseg)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low_freq, freqs <= hi_freq)
    # Integral approximation of the coherence using Simpson's rule.
    bp = simps(coh[idx_band], dx=freq_res)
    return bp


def coherency_adjacency(x, sf, low_freq, hi_freq):
    n_ch = x.shape[0]
    coherence_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i):
            coherence_matrix[i][j] = calculate_coherence(x[i, :], x[j, :], sf, low_freq, hi_freq)
    coherence_matrix += coherence_matrix.T
    coherence_matrix /= coherence_matrix.max()
    return coherence_matrix


def calculate_plv(x, y):
    analytic_x = si.hilbert(x)
    analytic_y = si.hilbert(y)

    phase_x = np.unwrap(np.angle(analytic_x))
    phase_y = np.unwrap(np.angle(analytic_y))

    plv = np.mean([np.exp(1j*(phase_x - phase_y))])
    plv = np.sqrt(np.real(plv)**2 + np.imag(plv)**2)
    return plv


def plv_adjacency(x):
    n_ch = x.shape[0]
    plv_matrix = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i):
            plv_matrix[i][j] = calculate_plv(x[i, :], x[j, :])
    plv_matrix += plv_matrix.T
    return plv_matrix


