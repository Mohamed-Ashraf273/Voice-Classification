import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
from src import preprocessing


class GMMClassifier:
    def __init__(self, n_components=4, covariance_type="diag", random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.models = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
            )
            gmm.fit(X[y == cls])
            self.models[cls] = gmm
        return self

    def predict(self, X):
        scores = np.array([self.models[cls].score_samples(X) for cls in self.classes_])
        return self.classes_[np.argmax(scores, axis=0)]


def gmm_classifier(x_train, y_train):
    model = GMMClassifier(n_components=2000)
    return model.fit(x_train, y_train)


def svc_classifier(x_train, y_train):
    model = SVC(
        probability=True, C=10, kernel="poly", gamma="scale", degree=6, random_state=42
    )  # best: probability=True, C=10, kernel="poly", gamma="scale", degree=6, random_state=42
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


def train(path, save_test, datapath, model_type):
    x_train, x_test, x_val, y_train, y_test, y_val = (
        preprocessing.preprocessing_features(path, save_test, datapath)
    )

    if model_type == "svc":
        best_classifier = svc_classifier(x_train, y_train)
    elif model_type == "gmm":
        best_classifier = gmm_classifier(x_train, y_train)
    else:
        raise ValueError("Invalid model_type. Choose 'svc' or 'gmm'.")

    pickle.dump(best_classifier, open(f"./data/model_{model_type}.pkl", "wb"))
    print(f"Model saved as model_{model_type}.pkl")
    return x_test, x_val, y_test, y_val
