import csv
import librosa
import numpy as np
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import OneHotEncoder
from src import preprocessing
from threading import Lock


accent_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")


def extract_features(y, sr, preprocess=False, accent_df=None, file=None):
    if preprocess:
        y = preprocessing.preprocess_audio(y, sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    base_features = np.concatenate(
        (
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_centroid, axis=1),
        )
    )
    if accent_df is not None and file is not None:
        accent = accent_df[accent_df["path"] == file]["accent"]
        if not accent.empty:
            accent_label = accent.values[0]
            encoded = accent_encoder.transform(
                pd.DataFrame([[accent_label]], columns=["accent"])
            )
            return np.concatenate((base_features, encoded.flatten()))
        else:
            print(f"Accent not found for file: {file}")
            dummy_encoding = np.zeros(len(accent_encoder.categories_[0]))
            return np.concatenate((base_features, dummy_encoding))
    return base_features


def process_file(file_path, file, df, lock):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        features = extract_features(y, sr, preprocess=True, accent_df=df, file=file)
        with lock:
            match = df[df["path"] == file]
        if not match.empty:
            gender = match.iloc[0]["gender"]
            age = match.iloc[0]["age"]
            label = match.iloc[0]["label"]
            features_str = ",".join(map(str, features))
            return [gender, age, features_str, label, file]
        else:
            print(f"Metadata not found for {file}")
            return None
    except Exception as e:
        print(f"Failed to process {file}: {e}")
        return "FAIL"


def get_features(metadata_path, output_path, production, max_workers=12):
    if production:
        features_file = "/sampled_50k.tsv"
    else:
        features_file = "/filtered_data_labeled.tsv"
    df = pd.read_csv(metadata_path + features_file, sep="\t")
    df["accent"] = df["accent"].fillna("none")
    # df = df.dropna(subset=["accent"]) #dropping it
    accent_encoder.fit(df[["accent"]])
    data_rows = []
    lock = Lock()
    batches = [
        os.path.join(metadata_path, f)
        for f in os.listdir(metadata_path)
        if f.startswith("audio")
    ]

    all_files = []
    for batch in batches:
        for file in os.listdir(batch):
            all_files.append((os.path.join(batch, file), file))

    total_files = len(all_files)
    fail_to_load_count = 0

    print(f"Processing {total_files} files using {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, file_path, file, df, lock)
            for file_path, file in all_files
        ]

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result == "FAIL":
                fail_to_load_count += 1
            elif result:
                data_rows.append(result)

            if i % 100 == 0:
                print(f"Processed {i}/{total_files} files.")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gender", "age", "features", "label", "path"])
        writer.writerows(data_rows)

    print(
        f"Features saved to {output_path} with {(fail_to_load_count/total_files)*100:.2f}% failed to load percent"
    )
