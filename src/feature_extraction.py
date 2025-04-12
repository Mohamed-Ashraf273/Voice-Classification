import shutil
import pandas as pd
import zipfile
import os
import librosa
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from scipy.signal import butter, lfilter, iirnotch
from IPython.display import Audio
from xyzservices.providers import data_path

data_path = "../data/Pattern Recognition Project Dataset V2/"

def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features = np.concatenate((np.mean(mfccs, axis=1), np.mean(chroma, axis=1), np.mean(spectral_centroid, axis=1)))
    return features

metadata_path = data_path+'filtered_data_labeled.tsv'
df = pd.read_csv(metadata_path, sep='\t')
data_rows = []
batches = [os.path.join( data_path+'uncompressed', f) for f in os.listdir( data_path+'uncompressed') if f.startswith('audio')]
data_len = 0
fail_to_load_count = 0
for batch in batches:
    data_len += len(os.listdir(batch)[:2000])
for batch in batches:
    print(batch)
    for file in os.listdir(batch)[:2000]:
        file_path = os.path.join(batch, file)
        try:
          y, sr = librosa.load(file_path, sr=16000) # or make it sr=22050 (librosa default)
        except:
          fail_to_load_count += 1
          continue
        features = extract_features(y, sr)
        match = df[df['path'] == file]
        if not match.empty:
            gender = match.iloc[0]['gender']
            age = match.iloc[0]['age']
            label = match.iloc[0]['label']
            #path = match.iloc[0]['path']
            features_str = ','.join(map(str, features))
            data_rows.append([gender, age, features_str, label, file])
        else:
            print(f"Metadata not found for {file}")
output_path = '/content/extracted_features.csv'
with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([ 'gender', 'age', 'features', 'label', 'path'])
    writer.writerows(data_rows)
print(f"Features saved to {output_path} with {(fail_to_load_count/data_len)*100}% failed to load percent")