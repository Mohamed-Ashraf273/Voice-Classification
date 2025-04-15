import csv
import librosa
import numpy as np
import os
import pandas as pd
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import OneHotEncoder
from src import preprocessing
from threading import Lock


def get_accent(accent_encoder, base_features):
    pass


def extract_features(
    y,
    sr,
    accent_feature_extraction,
    file,
    accent_encoder=None,
    accent_df=None,
    preprocess=False,
):
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
    if accent_feature_extraction:  # dev
        return base_features
    if accent_df is not None:  # dev
        accent = accent_df[accent_df["path"] == file]["accent"]
        accent_label = accent.values[0]
        encoded = accent_encoder.transform(
            pd.DataFrame([[accent_label]], columns=["accent"])
        )
    else:  # production
        encoded = get_accent(accent_encoder, base_features)
    return np.concatenate((base_features, encoded.flatten()))


def process_file_dev(
    file_path, file, accent_encoder, accent_feature_extraction, df, lock
):
    try:
        with lock:
            match = df[df["path"] == file]
        if not match.empty:
            y, sr = librosa.load(file_path, sr=16000)
            features = extract_features(
                y,
                sr,
                accent_feature_extraction,
                file=file,
                accent_encoder=accent_encoder,
                accent_df=df,
                preprocess=True,
            )
            if accent_feature_extraction:
                accent = match.iloc[0]["accent"]
                encoded_label = accent_encoder.transform(
                    pd.DataFrame([[accent]], columns=["accent"])
                )
                label = ",".join(map(str, encoded_label.flatten()))
            else:
                label = match.iloc[0]["label"]
            features_str = ",".join(map(str, features))
            return [features_str, label, file]
        else:
            print(f"Metadata not found for {file}")
            return None
    except Exception as e:
        print(f"Failed to process {file}: {e}")
        return "FAIL"


def process_file_prod(file_path, file, accent_encoder, accent_feature_extraction, *_):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        features = extract_features(
            y,
            sr,
            accent_feature_extraction,
            file=file,
            accent_encoder=accent_encoder,
            preprocess=True,
        )
        features_str = ",".join(map(str, features))
        return [features_str, file]
    except Exception as e:
        print(f"Failed to process {file}: {e}")
        return "FAIL"


def get_features(
    metadata_path,
    output_path,
    production,
    accent_feature_extraction,
    max_workers=12,
):
    data_rows = []
    lock = Lock()

    if production:
        files = sorted(os.listdir(metadata_path))
        all_files = [(os.path.join(metadata_path, file), file) for file in files]
        with open("./data/accent_encoder.pkl", "rb") as f:
            accent_encoder = pickle.load(f)
        process_file = process_file_prod
        df = None
    else:
        data_file = os.path.join(metadata_path, "filtered_data_labeled.tsv")
        df = pd.read_csv(data_file, sep="\t")
        df["accent"] = df["accent"].fillna("none")
        # df = df.dropna(subset=["accent"]) #dropping it

        if accent_feature_extraction:
            with open("./data/accent_encoder.pkl", "rb") as f:
                accent_encoder = pickle.load(f)
        else:
            accent_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            accent_encoder.fit(df[["accent"]])
            with open("./data/accent_encoder.pkl", "wb") as f:
                pickle.dump(accent_encoder, f)

        batches = [
            os.path.join(metadata_path, f)
            for f in os.listdir(metadata_path)
            if f.startswith("audio")
        ]

        all_files = [
            (os.path.join(batch, file), file)
            for batch in batches
            for file in os.listdir(batch)
        ]
        process_file = process_file_dev

    total_files = len(all_files)
    fail_to_load_count = 0

    print(f"Processing {total_files} files using {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_file,
                file_path,
                file,
                accent_encoder,
                accent_feature_extraction,
                df,
                lock,
            )
            for file_path, file in all_files
        ]

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result == "FAIL":
                fail_to_load_count += 1
            elif result:
                data_rows.append(result)

            if i % 100 == 0 or i == total_files:
                print(f"Processed {i}/{total_files} files.")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if production:
            writer.writerow(["features", "path"])
        else:
            writer.writerow(["features", "label", "path"])
        writer.writerows(data_rows)

    print(
        f"Features saved to {output_path} with {(fail_to_load_count / total_files) * 100:.2f}% failed to load percent"
    )
