import csv
import librosa
import numpy as np
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from src import preprocessing


def chunkify(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def extract_features(y, sr, preprocess=False):
    if preprocess:
        y = preprocessing.preprocess_audio(y, sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.concatenate(
        (
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_centroid, axis=1),
        )
    )


def process_file_batch(batch, metadata_dict, production):
    results = []

    for file_path, file in batch:
        try:
            if not production:
                file_info = metadata_dict.get(file)
                if file_info is None:
                    print(f"Metadata not found for {file}")
                    results.append(None)
                    continue

            y, sr = librosa.load(file_path, sr=16000)
            features = extract_features(y, sr, preprocess=True)
            features_str = ",".join(map(str, features))

            if production:
                results.append([features_str, file])
            else:
                results.append([features_str, file_info["label"], file])

        except Exception as e:
            print(f"Failed to process {file}: {e}")
            results.append("FAIL")

    return results


def get_features(metadata_path, output_path, production, max_workers=12, chunk_size=50):
    if production:
        files = sorted(os.listdir(metadata_path))
        all_files = [(os.path.join(metadata_path, file), file) for file in files]
        metadata_dict = None
    else:
        data_file = os.path.join(metadata_path, "filtered_data_labeled.tsv")
        df = pd.read_csv(data_file, sep="\t")
        metadata_dict = df.set_index("path").to_dict("index")

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

    total_files = len(all_files)
    fail_to_load_count = 0
    data_rows = []

    print(f"Processing {total_files} files using {max_workers} processes...")

    file_chunks = chunkify(all_files, chunk_size)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file_batch, chunk, metadata_dict, production)
            for chunk in file_chunks
        ]

        for i, future in enumerate(as_completed(futures), 1):
            batch_results = future.result()
            for result in batch_results:
                if result == "FAIL":
                    fail_to_load_count += 1
                elif result:
                    data_rows.append(result)

            if i % max_workers == 0 or i == len(file_chunks):
                processed = min(i * chunk_size, total_files)
                print(f"Processed {processed}/{total_files} files")

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
