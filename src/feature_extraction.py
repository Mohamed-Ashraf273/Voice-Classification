import csv
import librosa
import numpy as np
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.preprocessing import preprocess_audio
from threading import Lock


def extract_features(y, sr, preprocess=False):
    if preprocess:
        y = preprocess_audio(y, sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features = np.concatenate(
        (
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_centroid, axis=1),
        )
    )
    return features


def process_file(file_path, file, df, lock):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        features = extract_features(y, sr, preprocess=True)
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


def get_features(metadata_path, output_path, max_workers=12):
    df = pd.read_csv(metadata_path + "filtered_data_labeled.tsv", sep="\t")
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
