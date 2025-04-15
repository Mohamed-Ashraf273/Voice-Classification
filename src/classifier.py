import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def preprocessing_feature(path):
    df = pd.read_csv(path)
    print(len(df))
    y = df["label"]
    x = df["features"]
    x = x.tolist()
    x = [np.asarray(x.split(","), np.float32) for x in x]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.5, random_state=42
    )
    under_sample = RandomUnderSampler()
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train.tolist())
    X_resampled, y_resampled = under_sample.fit_resample(x_train, y_train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_resampled)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    return (
        x_train,
        x_test,
        x_val,
        np.array(y_resampled).ravel(),
        np.array(y_test).ravel(),
        np.array(y_val).ravel(),
    )


def svc_classifier(x_train, y_train):
    model = SVC(probability=True, C=10, kernel="rbf", gamma="scale")
    # param_grid = {
    #     "C": [0.1, 1],
    #     "kernel": ["linear", "rbf"],
    # }

    # grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=10,return_train_score=True)
    # grid_search.fit(x_train, y_train)
    # best_model = grid_search.best_estimator_
    # print("Best parameters found: ", grid_search.best_params_)
    # print("Best cross-validation score: ", grid_search.best_score_)
    return model.fit(x_train, y_train)


def train(path):
    x_train, x_test, x_val, y_train, y_test, y_val = preprocessing_feature(path)
    best_classifier = svc_classifier(x_train, y_train)
    pickle.dump(best_classifier, open("./data/model_svc.pkl", "wb"))
    print("Model saved as model.pkl")
    return x_test, x_val, y_test, y_val
