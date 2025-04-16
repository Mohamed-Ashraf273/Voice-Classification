import json
import numpy as np
import pandas as pd
import pickle
import time
from src import feature_extraction, classifier, metrics
from pathlib import Path


def extract_features(
    datapath,
    augment,
    add_accents_to_features,
    accent_feature_extraction=False,
    production=False,
):
    start = time.time()
    metadata_path = datapath
    if production:
        output_path = "./data/features_prod.csv"
    elif accent_feature_extraction:
        output_path = "./data/features_accents.csv"
    else:
        output_path = "data/features.csv"
    if augment:
        print("Audios will be augmented")
    if add_accents_to_features:
        print("Accents will be added to features")
    feature_extraction.get_features(
        augment,
        metadata_path,
        output_path,
        production,
        add_accents_to_features,
        accent_feature_extraction=accent_feature_extraction,
        max_workers=12,
    )
    print(f"Time taken: {time.time() - start:.2f} seconds")


def plot_data_distribution(path):
    metrics.plot_data_distribution(path)


def train_classifier(path, save_test, accent_train, datapath, model_type="svc"):
    x_test, x_val, y_test, y_val = classifier.train(
        path, save_test, accent_train, datapath, model_type=model_type
    )
    return x_test, x_val, y_test, y_val


def save_test_data(x_test, y_test, accent_train, filename="./data/test_data.json"):
    if accent_train:
        filename = "./data/test_data_accents.json"
    data = {
        "x_test": [x.tolist() for x in x_test],
        "y_test": y_test.tolist() if isinstance(y_test, np.ndarray) else list(y_test),
    }
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Test data saved to {filename}")


def get_metrics(model, x_test, y_test):
    metrics.show(model, x_test, y_test)


def predict_test_data(model, x_test):
    return metrics.get_predictions(model, x_test)


def dev(
    model,
    datapath=None,
    augment=False,
    features_file_path=None,
    accent_feature_extraction=False,
    add_accents_to_features=False,
    accent_train=False,
    train=False,
    save_test=False,
):
    if train:
        if features_file_path is None:
            raise ValueError("features_file_path must be provided when train is True")
        print("Training mode")
        model_selected = model
        x_test, x_val, y_test, y_val = train_classifier(
            features_file_path,
            save_test,
            accent_train,
            datapath,
            model_type=model_selected,
        )
        if save_test:
            save_test_data(x_test, y_test, accent_train)
        if accent_train:
            with open(f"./data/model_{model_selected}_accent.pkl", "rb") as file:
                loaded_model = pickle.load(file)
        else:
            with open(f"./data/model_{model_selected}.pkl", "rb") as file:
                loaded_model = pickle.load(file)
        get_metrics(loaded_model, x_val, y_val)
    else:
        print("Feature extracting mode")
        if datapath is None:
            raise ValueError("datapath must be provided in feature extracting mode")
        extract_features(
            datapath,
            augment,
            add_accents_to_features,
            accent_feature_extraction=accent_feature_extraction,
        )


def predict_all(test_file_path, accent_test=False, model_selected="svc"):
    print(
        "This function will predict on our model, but if you trained a model it will override ours."
    )
    with open(test_file_path, "r") as file:
        data = json.load(file)
    x_test = np.array(data["x_test"])
    y_test = np.array(data["y_test"])
    if accent_test:
        with open(f"./data/model_{model_selected}_accent.pkl", "rb") as file:
            loaded_model = pickle.load(file)
        with open("./data/scaler_accents.pkl", "rb") as f:
            scaler = pickle.load(f)
    else:
        with open(f"./data/model_{model_selected}.pkl", "rb") as file:
            loaded_model = pickle.load(file)
        with open("./data/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    x_test = scaler.transform(x_test)
    get_metrics(loaded_model, x_test, y_test)


def final_out(test_file_path, model_selected):
    raise ValueError(
        "This function is not fully supported yet. Please use the predict_all function instead."
    )
    print(
        "This function will predict on our model, but if you trained a model it will override ours."
    )
    extract_features(test_file_path, production=True)
    df = pd.read_csv("./data/features_prod.csv")
    x_test = df["features"].values.tolist()
    x_test = [np.asarray(x.split(","), np.float32) for x in x_test]
    with open(f"./data/model_{model_selected}.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    with open("./data/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    x_test = scaler.transform(x_test)
    predictions = predict_test_data(loaded_model, x_test)
    with open(f"./data/predictions_{model_selected}.txt", "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    print(f"Predictions saved to predictions_{model_selected}.txt")
