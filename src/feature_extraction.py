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


def extract_features(
    y,
    sr,
    augment,
    file,
    preprocess=False,
):
    if preprocess:
        y = preprocessing.preprocess_audio(y, sr, augment)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    base_features = np.concatenate(
        (
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_centroid, axis=1),
        )
    )
    return base_features


def process_file_dev(
    file_path,
    file,
    augment,
    df,
    lock,
):
    try:
        with lock:
            match = df[df["path"] == file]
        if not match.empty:
            y, sr = librosa.load(file_path, sr=16000)
            features = extract_features(
                y,
                sr,
                augment,
                file=file,
                preprocess=True,
            )
            label = match.iloc[0]["label"]
            features_str = ",".join(map(str, features))
            return [features_str, label, file]
        else:
            print(f"Metadata not found for {file}")
            return None
    except Exception as e:
        print(f"Failed to process {file}: {e}")
        return "FAIL"


def process_file_prod(
    file_path,
    file,
    augment,
    *_,
):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        features = extract_features(
            y,
            sr,
            augment,
            file=file,
            preprocess=True,
        )
        features_str = ",".join(map(str, features))
        return [features_str, file]
    except Exception as e:
        print(f"Failed to process {file}: {e}")
        return "FAIL"


def get_features(
    augment,
    metadata_path,
    output_path,
    production,
    max_workers=12,
):
    data_rows = []
    lock = Lock()

    if production:
        files = sorted(os.listdir(metadata_path))
        all_files = [(os.path.join(metadata_path, file), file) for file in files]
        process_file = process_file_prod
        df = None
    else:
        data_file = os.path.join(metadata_path, "filtered_data_labeled.tsv")
        df = pd.read_csv(data_file, sep="\t")

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
                augment,
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
