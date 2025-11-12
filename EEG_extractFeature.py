import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
import pywt


FS = 128
WINDOW = 50
OVERLAP = 5
EEG_START, EEG_END = 2, 16
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
    'gamma': (30, 50)
}


def bandpass_filter(x, low, high, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, x)

def iir_smoothing(x):
    return bandpass_filter(x, 0.5, 50, FS, order=2)


def hjorth_parameters(x):
    dx = np.diff(x)
    ddx = np.diff(dx)
    var_x, var_dx, var_ddx = np.var(x), np.var(dx), np.var(ddx)
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0
    complexity = np.sqrt(var_ddx / var_dx) / mobility if (var_dx > 0 and mobility > 0) else 0
    return mobility, complexity

def hurst_exponent(x):
    N = len(x)
    if N < 20:
        return 0.5
    Y = np.cumsum(x - np.mean(x))
    R, S = np.max(Y) - np.min(Y), np.std(x)
    if S == 0 or R == 0:
        return 0.5
    return np.log(R / S) / np.log(N)

def petrosian_fd(x):
    diff = np.diff(x)
    Ndelta = np.sum(diff[1:] * diff[:-1] < 0)
    N = len(x)
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * Ndelta + 1e-9)))

def katz_fd(x):
    n = len(x)
    L = np.sum(np.sqrt(np.diff(x) ** 2))
    d = np.max(np.abs(x - x[0]))
    if d == 0:
        return 0
    return np.log10(n) / (np.log10(n) + np.log10(d / 1.0 + 1e-9))

def dfa(x):
    X = np.cumsum(x - np.mean(x))
    coeffs = np.polyfit(np.arange(len(X)), X, 1)
    trend = np.polyval(coeffs, np.arange(len(X)))
    return np.sqrt(np.mean((X - trend) ** 2))

def dwt_features(x, wavelet='db4', level=3):
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    allc = np.hstack(coeffs)
    return np.mean(allc), np.median(allc), np.std(allc)

def power_spectral_density(x):
    f, Pxx = welch(x, fs=FS, nperseg=min(128, len(x)))
    return np.trapz(Pxx, f)

def extract_features(signal):
    step = WINDOW - OVERLAP
    features = []

    for start in range(0, len(signal) - WINDOW + 1, step):
        seg = signal[start:start + WINDOW]
        feat_vec = []

        for band, (low, high) in BANDS.items():
            try:
                x = bandpass_filter(seg, low, high, FS)
            except Exception:
                x = seg.copy()

            psd = power_spectral_density(x)
            power = np.mean(x ** 2)
            hurst = hurst_exponent(x)
            mob, comp = hjorth_parameters(x)
            petro = petrosian_fd(x)
            dwt_mean, dwt_med, dwt_std = dwt_features(x)
            katz = katz_fd(x)
            fluct = dfa(x)

            feat_vec.extend([
                psd, power, hurst, mob, comp, petro,
                dwt_mean, dwt_med, dwt_std, katz, fluct
            ])
        features.append(feat_vec)

    return np.array(features)

def process_eeg_file(path):
    df = pd.read_csv(path)
    eeg = df.iloc[:, EEG_START:EEG_END].to_numpy().T 
    eeg = np.array([iir_smoothing(ch[10:]) for ch in eeg])
    all_feats = [extract_features(ch) for ch in eeg]
    n = min(f.shape[0] for f in all_feats)
    all_feats = [f[:n] for f in all_feats]
    features = np.hstack(all_feats)
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-9
    features_norm = (features - mean) / std

    return features_norm

if __name__ == "__main__":
    input_path = "Test0_2019.08.18_14.15.17.csv"
    features = process_eeg_file(input_path)
    pd.DataFrame(features).to_csv("EEG_features_normalized.csv", index=False)

