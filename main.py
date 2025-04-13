import json
import numpy as np
import pickle
import time
from src import feature_extraction, classifier, metrics
from pathlib import Path


def extract_features(datapath):
    start = time.time()
    metadata_path = datapath
    output_path = Path(__file__).resolve().parent / "features_multi.csv"
    feature_extraction.get_features(metadata_path, output_path, max_workers=12)
    print(f"Time taken: {time.time() - start:.2f} seconds")


def plot_data_distribution(path):
    metrics.plot_data_distribution(path)


def train_classifier(path):
    x_test, x_val, y_test, y_val = classifier.train(path)
    return x_test, x_val, y_test, y_val


def save_test_data(x_test, y_test, filename="test_data.json"):
    data = {
        "x_test": [x.tolist() for x in x_test],
        "y_test": y_test.tolist() if isinstance(y_test, np.ndarray) else list(y_test),
    }
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Test data saved to {filename}")


def get_metrics(model, x_test, y_test):
    metrics.show(model, x_test, y_test)


x_test, x_val, y_test, y_val = train_classifier("data\\features.csv")
save_test_data(x_test, y_test)
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)
get_metrics(loaded_model, x_val, y_val)
# x_train, x_test, x_val, y_train, y_test, y_val = classifier.preprocessing_feature("data\\features.csv")
# print(f"y_train -> {np.unique(y_train, return_counts=True)}")
# print(f"y_test -> {np.unique(y_test, return_counts=True)}")
# print(f"y_val -> {np.unique(y_val, return_counts=True)}")
