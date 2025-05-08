import librosa
import librosa.display
import librosa.effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
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


def remove_silence(y, top_db=20, frame_length=256, hop_length=64):
    y_trimmed, _ = librosa.effects.trim(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    return y_trimmed


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut=80, highcut=4000, fs=48000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y_filtered = lfilter(b, a, data)
    return y_filtered


def preprocess_audio(y, sr):
    y = remove_silence(y)
    y_filtered = bandpass_filter(y, fs=sr)
    y_filtered = y_filtered / np.max(np.abs(y_filtered))
    return y_filtered


def balancing_pipeline(X, y, min_samples=10, majority_ratio=3, random_state=42):
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

    steps = []

    steps.append(
        RandomUnderSampler(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
    )

    steps.append(TomekLinks(sampling_strategy="majority"))

    pipeline = make_pipeline(*steps)
    return pipeline.fit_resample(X, y)


def preprocessing_features(path, gender, age, model_type):
    df = pd.read_csv(path)
    if gender:
        y = (df["label"] == 0) | (df["label"] == 2)  # 1 for male 0 for female
    elif age:
        y = (df["label"] == 0) | (df["label"] == 1)  # 1 for male 20 for 50
    else:
        y = df["label"]
    x = df["features"]
    x = x.tolist()
    x = [np.asarray(s.split(","), np.float32) for s in x]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.5, random_state=42
    )

    scaler = StandardScaler()
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train.tolist())
    x_val = pd.DataFrame(x_val)
    y_val = pd.DataFrame(y_val.tolist())

    print("Train data before balancing: ", np.unique(y_train, return_counts=True))
    x_resampled, y_resampled = balancing_pipeline(
        x_train, y_train, min_samples=13000, majority_ratio=2, random_state=42
    )
    print("Train data after balancing: ", np.unique(y_resampled, return_counts=True))
    print("Validation data before balancing: ", np.unique(y_val, return_counts=True))
    x_val, y_val = balancing_pipeline(
        x_val, y_val, min_samples=500, majority_ratio=1, random_state=42
    )
    print("Validation data after balancing: ", np.unique(y_val, return_counts=True))

    if gender:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_resampled)
        x_val = scaler.transform(x_val)
        with open(f"./model/scaler_gfas_{model_type}.pkl", "wb") as f:
            pickle.dump(scaler, f)
    elif age:
        try:
            with open(f"./model/scaler_gfas_{model_type}.pkl", "rb") as f:
                scaler = pickle.load(f)
        except:
            raise ValueError(
                "You should provide a scaler_gfas.pkl file to use the gfas model, by running train with gender first"
            )
        x_train = scaler.fit_transform(x_resampled)
        x_val = scaler.transform(x_val)
    else:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_resampled)
        x_val = scaler.transform(x_val)
        with open(f"./model/scaler_{model_type}.pkl", "wb") as f:
            pickle.dump(scaler, f)
    return (
        x_train,
        x_test,
        x_val,
        np.array(y_resampled).ravel(),
        np.array(y_test).ravel(),
        np.array(y_val).ravel(),
    )
