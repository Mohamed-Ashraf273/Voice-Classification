import pandas as pd
import librosa
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


def plot_sig(signal, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(signal, sr=sr, color="royalblue")
    plt.title("Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_mse(sig1, sig2):
    return np.mean((sig1 - sig2) ** 2)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut=300, highcut=3400, fs=16000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y_filtered = lfilter(b, a, data)
    return y_filtered


def preprocess_audio(y, sr):
    y_filtered = bandpass_filter(y, fs=sr)
    y_filtered = y_filtered / np.max(np.abs(y_filtered))
    return y_filtered
