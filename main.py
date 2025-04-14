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
    output_path = Path(__file__).resolve().parent / "features.csv"
    feature_extraction.get_features(
        metadata_path, output_path, production, max_workers=12
    )
    print(f"Time taken: {time.time() - start:.2f} seconds")


def plot_data_distribution(path):
    metrics.plot_data_distribution(path)


def train_classifier(path, model_type="svc"):
    x_test, x_val, y_test, y_val = classifier.train(path, model_type=model_type)
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


def dev_phase(datapath, features_file_path, model):
    #extract_features(datapath)
    model_selected = "svc"
    x_test, x_val, y_test, y_val = train_classifier(
        features_file_path, model_type=model_selected
    )
    save_test_data(x_test, y_test)
    with open(f"model_{model_selected}.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    get_metrics(loaded_model, x_val, y_val)


#production_phase is not fully supported please support it
def production_phase(test_file_path, model_selected):
    raise ValueError(
        "Production phase is not fully supported yet. Please use the development phase."
        " The production phase is still under development and will be available in future releases."
    )
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


features_file_path = "./data/f_v3_best/features_V1.1.csv"
datapath = "data/voice_project_data"
dev_phase(datapath, features_file_path, model="svc")

# production_phase(
#     test_file_path="./data/voice_project_data",
#     model_selected="svc",
# )
