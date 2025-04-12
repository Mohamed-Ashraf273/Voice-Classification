import pandas as pd
import os
import librosa
import numpy as np
import librosa
import csv
from preprocessing import preprocess_audio


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


def get_features(metadata_path, output_path):
    df = pd.read_csv(metadata_path + "filtered_data_labeled.tsv", sep="\t")
    data_rows = []
    batches = [
        os.path.join(metadata_path, f)
        for f in os.listdir(metadata_path)
        if f.startswith("batch")
    ]
    data_len = 0
    fail_to_load_count = 0
    for batch in batches:
        data_len += len(os.listdir(batch))
    for batch in batches:
        print(batch)
        for file in os.listdir(batch):
            file_path = os.path.join(batch, file)
            try:
                y, sr = librosa.load(
                    file_path, sr=16000
                )  # or make it sr=22050 (librosa default)
            except:
                fail_to_load_count += 1
                continue
            features = extract_features(
                y, sr, True
            )  # change the thirs param to False if you need to extract with raw data
            match = df[df["path"] == file]
            if not match.empty:
                gender = match.iloc[0]["gender"]
                age = match.iloc[0]["age"]
                label = match.iloc[0]["label"]
                features_str = ",".join(map(str, features))
                data_rows.append([gender, age, features_str, label, file])
            else:
                print(f"Metadata not found for {file}")
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gender", "age", "features", "label", "path"])
        writer.writerows(data_rows)
    print(
        f"Features saved to {output_path} with {(fail_to_load_count/data_len)*100}% failed to load percent"
    )
