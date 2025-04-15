import csv
import librosa
import numpy as np
import os
import pandas as pd
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.preprocessing import OneHotEncoder
from src import preprocessing


def chunkify(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def extract_features(y, sr, accent_label, accent_encoder, preprocess=False):
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
    encoded = accent_encoder.transform(
        pd.DataFrame([[accent_label]], columns=["accent"])
    )

    return np.concatenate((base_features, encoded.flatten()))


def process_file_batch(batch, df, accent_encoder):
    results = []
    # Create a lookup dictionary from the DataFrame
    file_info_dict = df.set_index('path').to_dict('index')

    for file_path, file in batch:
        try:

            file_info = file_info_dict.get(file)
            if file_info is None:
                print(f"File not found in metadata: {file}")
                results.append("FAIL")
                continue

            y, sr = librosa.load(file_path, sr=16000)
            accent = file_info["accent"]
            gender = file_info["gender"]
            age = file_info["age"]
            label = file_info["label"]

            features = extract_features(
                y, sr, accent_label=accent, accent_encoder=accent_encoder, preprocess=True
            )
            features_str = ",".join(map(str, features))
            results.append([gender, age, features_str, label, file])
        except Exception as e:
            print(f"Failed to process {file}: {e}")
            results.append("FAIL")
    return results


def get_features(metadata_path, output_path, production, max_workers=12, chunk_size=500):
    if production:
        data_file = metadata_path
    else:
        data_file = os.path.join(metadata_path, "filtered_data_labeled.tsv")

    df = pd.read_csv(data_file, sep="\t")
    df["accent"] = df["accent"].fillna("none")

    if production:
        with open("./data/accent_encoder.pkl", "rb") as f:
            accent_encoder = pickle.load(f)
    else:
        accent_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        accent_encoder.fit(df[["accent"]])
        with open("./data/accent_encoder.pkl", "wb") as f:
            pickle.dump(accent_encoder, f)

    audio_dirs = [
        os.path.join(metadata_path, f)
        for f in os.listdir(metadata_path)
        if f.startswith("audio")
    ]

    all_files = []
    for audio_dir in audio_dirs:
        for file in os.listdir(audio_dir):
            all_files.append((os.path.join(audio_dir, file), file))

    total_files = len(all_files)
    fail_to_load_count = 0
    data_rows = []
    print(f"Processing {total_files} files using {max_workers} processes...")
    file_chunks = chunkify(all_files, chunk_size)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file_batch, chunk, df, accent_encoder)
                   for chunk in file_chunks]

        for i, future in enumerate(as_completed(futures), 1):
            results = future.result()
            for result in results:
                if result == "FAIL":
                    fail_to_load_count += 1
                elif result:
                    data_rows.append(result)
            if i % 12 == 0:
                print(f"Processed {i * chunk_size}/{total_files} files.")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gender", "age", "features", "label", "path"])
        writer.writerows(data_rows)

    print(
        f"Features saved to {output_path} with {(fail_to_load_count / total_files) * 100:.2f}% failed to load percent"
    )
