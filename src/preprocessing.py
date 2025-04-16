import librosa
import librosa.display
import librosa.effects
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import shutil
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
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


def get_test_dir(paths, accent_train):
    batches = [
        os.path.join("data/voice_project_data", f)
        for f in os.listdir("data/voice_project_data")
        if f.startswith("audio")
    ]

    all_files = [
        (os.path.join(batch, file), file)
        for batch in batches
        for file in os.listdir(batch)
    ]
    if accent_train:
        test_dir = "./data/test_accents"
    else:
        test_dir = "./data/test"
    os.makedirs(test_dir, exist_ok=True)
    copied = 0
    for file_name in paths:
        for full_path, fname in all_files:
            if fname == file_name:
                shutil.copy(full_path, os.path.join(test_dir, fname))
                copied += 1
                break


def balanced_undersampling_pipeline(
    X, y, min_samples=10, majority_ratio=3, random_state=42
):
    """
    More conservative undersampling that:
    1. Protects tiny classes (< min_samples)
    2. Only reduces true majority classes
    3. Never drops below min_samples
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_count = np.min(class_counts)

    protected_classes = [
        cls for cls, count in zip(unique_classes, class_counts) if count < min_samples
    ]

    sampling_strategy = {}
    for cls, count in zip(unique_classes, class_counts):
        if cls in protected_classes:
            continue
        target = int(minority_count * majority_ratio)
        sampling_strategy[cls] = min(count, max(target, min_samples))

    if not sampling_strategy:
        return X, y

    undersampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy, random_state=random_state
    )

    tomek = TomekLinks(sampling_strategy="majority")

    pipeline = make_pipeline(undersampler, tomek)
    return pipeline.fit_resample(X, y)


def preprocessing_features(path, save_test, accent_train, datapath):
    df = pd.read_csv(path)
    y = df["label"]
    x = df["features"]
    paths = df["path"]
    if save_test:
        assert (
            datapath is not None
        ), "you should provide a datapath so we can create your test dir"
        get_test_dir(paths, accent_train)
    x = x.tolist()
    x = [np.asarray(s.split(","), np.float32) for s in x]
    if accent_train:
        y = [
            np.asarray([int(float(i)) for i in s.split(",")], dtype=np.int8) for s in y
        ]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.5, random_state=42
    )

    # second plan
    # sampling_strategy = {0: 31779}
    # undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy)
    # X_resampled, y_resampled = undersampler.fit_resample(x_train, y_train)

    scaler = StandardScaler()
    x_train = pd.DataFrame(x_train)
    if accent_train:
        y_train = pd.DataFrame(y_train)
        y_test = pd.DataFrame(y_test)
        y_val = pd.DataFrame(y_val)
        y_train = y_train.values.argmax(axis=1)
        y_test = y_test.values.argmax(axis=1)
        y_val = y_val.values.argmax(axis=1)
    else:
        y_train = pd.DataFrame(y_train.tolist())

    # X_resampled, y_resampled = x_train[:80000], y_train[:80000]
    print(np.unique(y_train, return_counts=True))
    x_resampled, y_resampled = balanced_undersampling_pipeline(
        x_train, y_train, min_samples=9000, majority_ratio=3
    )
    print(np.unique(y_resampled, return_counts=True))

    x_train = scaler.fit_transform(x_resampled)
    x_val = scaler.transform(x_val)
    if accent_train:
        with open("./data/scaler_accents.pkl", "wb") as f:
            pickle.dump(scaler, f)
    else:
        with open("./data/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    return (
        x_train,
        x_test,
        x_val,
        np.array(y_resampled).ravel(),
        np.array(y_test).ravel(),
        np.array(y_val).ravel(),
    )
