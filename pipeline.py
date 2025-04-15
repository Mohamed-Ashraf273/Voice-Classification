import json
import numpy as np
import pandas as pd
import pickle
import time
from src import feature_extraction, classifier, metrics
from pathlib import Path


def extract_features(datapath, production=False):
    start = time.time()
    metadata_path = datapath
    output_path = Path(__file__).resolve().parent / "data/features.csv"
    feature_extraction.get_features(
        metadata_path, output_path, production, max_workers=12
    )
    print(f"Time taken: {time.time() - start:.2f} seconds")


def plot_data_distribution(path):
    metrics.plot_data_distribution(path)


def train_classifier(path, model_type="svc"):
    x_test, x_val, y_test, y_val = classifier.train(path)
    return x_test, x_val, y_test, y_val


def save_test_data(x_test, y_test, filename="./data/test_data.json"):
    data = {
        "x_test": [x.tolist() for x in x_test],
        "y_test": y_test.tolist() if isinstance(y_test, np.ndarray) else list(y_test),
    }
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Test data saved to {filename}")


def get_metrics(model, x_test, y_test):
    metrics.show(model, x_test, y_test)


def dev(model, datapath=None, features_file_path=None, train=True):
    if train:
        if features_file_path is None:
            raise ValueError("features_file_path must be provided when train is True")
        print("Training mode")
        model_selected = model
        x_test, x_val, y_test, y_val = train_classifier(
            features_file_path, model_type=model_selected
        )
        save_test_data(x_test, y_test)
        with open(f"./data/model_{model_selected}.pkl", "rb") as file:
            loaded_model = pickle.load(file)
        get_metrics(loaded_model, x_val, y_val)
    else:
        print("Feature extracting mode")
        if datapath is None:
            raise ValueError("datapath must be provided in feature extracting mode")
        extract_features(datapath)


def predict_all(test_file_path, model_selected):
    print("This function will predict on our model, but if you trained a model it will override ours.")
    extract_features(test_file_path, production=True)
    df = pd.read_csv("features.csv")
    x_test = df["features"].values.tolist()
    x_test = [np.asarray(x.split(","), np.float32) for x in x_test]
    y_test = df["label"].values.tolist()
    with open(f"./data/model_{model_selected}.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    with open("./data/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    x_test = scaler.transform(x_test)
    get_metrics(loaded_model, x_test, y_test)