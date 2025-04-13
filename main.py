import time
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from src import feature_extraction, get_metrics, train_classifier
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def extract_features(datapath):
    start = time.time()
    metadata_path = datapath
    output_path = Path(__file__).resolve().parent / "features_multi.csv"
    feature_extraction.get_features(metadata_path, output_path, max_workers=12)
    print(f"Time taken: {time.time() - start:.2f} seconds")


def plot_data_distribution(path):
    get_metrics.plot_data_distribution(path)


def preprocessing_feature(path):
    df = pd.read_csv(path)
    y = df["label"]
    x = df["features"]
    x = x.tolist()
    x = [np.asarray(x.split(","), np.float32) for x in x]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    under_sample = RandomUnderSampler()
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train.tolist())
    X_resampled, y_resampled = under_sample.fit_resample(x_train, y_train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_resampled)
    x_test = scaler.transform(x_test)
    return x_train, x_test, np.array(y_resampled).ravel(), np.array(y_test).ravel()


def train_classifier_main():
    x_train, x_test, y_train, y_test = preprocessing_feature("data\\features.csv")
    best_classifier = train_classifier.train_classifier(x_train, y_train)
    y_pred = best_classifier.predict(x_test)
    pickle.dump(best_classifier, open("model.pkl", "wb"))
    print("Model saved as model.pkl")
    print("accuracy report:", accuracy_score(y_test, y_pred))
    print("classification report:", classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# datapath = "data/uncompressed/"
train_classifier_main()
