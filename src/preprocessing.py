import librosa
import librosa.display
import librosa.effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from imblearn.under_sampling import RandomUnderSampler
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def remove_silence(y, top_db=30):
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed


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
    y = remove_silence(y)
    y_filtered = bandpass_filter(y, fs=sr)
    y_filtered = y_filtered / np.max(np.abs(y_filtered))
    return y_filtered


def preprocessing_features(path):
    df = pd.read_csv(path)
    y = df["label"]
    x = df["features"]
    x = x.tolist()
    x = [np.asarray(x.split(","), np.float32) for x in x]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.5, random_state=42
    )
    under_sample = RandomUnderSampler()
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train.tolist())
    X_resampled, y_resampled = under_sample.fit_resample(x_train, y_train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_resampled)
    x_val = scaler.transform(x_val)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return (
        x_train,
        x_test,
        x_val,
        np.array(y_resampled).ravel(),
        np.array(y_test).ravel(),
        np.array(y_val).ravel(),
    )
