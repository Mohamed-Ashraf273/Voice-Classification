import json
import numpy as np
import pandas as pd
import pickle
import time
from src import feature_extraction, classifier, metrics
from pathlib import Path


def extract_features(
    datapath,
    production=False,
):
    start = time.time()
    metadata_path = datapath
    if production:
        output_path = "./data/features_prod.csv"
    else:
        output_path = "data/features.csv"
    feature_extraction.get_features(
        metadata_path,
        output_path,
        production,
        max_workers=12,
    )
    print(f"Time taken: {time.time() - start:.2f} seconds")


def plot_data_distribution(path):
    metrics.plot_data_distribution(path)


def train_classifier(path, gender, age, make_test_dir, datapath, model_type="xgboost"):
    x_test, x_val, y_test, y_val = classifier.train(
        path, gender, age, make_test_dir, datapath, model_type=model_type
    )
    return x_test, x_val, y_test, y_val


def save_test_data(x_test, y_test, filename="./data/test_data.json"):
    data = {
        "x_test": [x.tolist() for x in x_test],
        "y_test": y_test.tolist() if isinstance(y_test, np.ndarray) else list(y_test),
    }
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Test data saved to {filename}")


def get_metrics(model, x_test, y_test, gfas, gender_model=None, age_model=None):
    metrics.show(model, x_test, y_test, gfas, gender_model, age_model)


def predict_test_data(model, x_test):
    return metrics.get_predictions(model, x_test)


def dev(
    model="xgboost",
    datapath=None,
    features_file_path=None,
    train=False,
    gender=False,
    age=False,
    save_test=False,
    save_val=False,
    make_test_dir=False,
):
    if train:
        if features_file_path is None:
            raise ValueError("features_file_path must be provided when train is True")
        print("Training mode")
        if gender:
            print("Gender training")
        elif age:
            print("Age training")
        else:
            print("Normal training")
        model_selected = model
        x_test, x_val, y_test, y_val = train_classifier(
            features_file_path,
            gender,
            age,
            make_test_dir,
            datapath,
            model_type=model_selected,
        )
        if save_test:
            save_test_data(x_test, y_test)
        if save_val:
            save_test_data(x_val, y_val, filename="./data/test_val.json")
        if gender:
            with open(f"./data/model_{model_selected}_gender.pkl", "rb") as file:
                loaded_model = pickle.load(file)
        elif age:
            with open(f"./data/model_{model_selected}_age.pkl", "rb") as file:
                loaded_model = pickle.load(file)
        else:
            with open(f"./data/model_{model_selected}.pkl", "rb") as file:
                loaded_model = pickle.load(file)
        get_metrics(loaded_model, x_val, y_val, gfas=False)
    else:
        print("Feature extracting mode")
        if datapath is None:
            raise ValueError("datapath must be provided in feature extracting mode")

        extract_features(
            datapath,
        )


def predict_all(test_file_path, val=False, model_selected="xgboost", gfas=False):
    print(
        "This function will predict on our model, but if you trained a model it will override ours."
    )
    with open(test_file_path, "r") as file:
        data = json.load(file)
    x_test = np.array(data["x_test"])
    y_test = np.array(data["y_test"])
    print("Data distribution: ", np.unique(y_test, return_counts=True))
    if gfas:
        with open(f"data/model_{model_selected}_gender.pkl", "rb") as file:
            gender_model = pickle.load(file)
        with open(f"data/model_{model_selected}_age.pkl", "rb") as file:
            age_model = pickle.load(file)
        with open(f"./data/scaler_gfas_{model_selected}.pkl", "rb") as f:
            scaler = pickle.load(f)
        loaded_model = None
    else:
        gender_model = None
        age_model = None
        with open(f"./data/model_{model_selected}.pkl", "rb") as file:
            loaded_model = pickle.load(file)
        with open(f"./data/scaler_{model_selected}.pkl", "rb") as f:
            scaler = pickle.load(f)
    if not val:
        x_test = scaler.transform(x_test)
    get_metrics(loaded_model, x_test, y_test, gfas, gender_model, age_model)


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
