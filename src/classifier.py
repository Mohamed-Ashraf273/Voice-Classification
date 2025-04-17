import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.utils.class_weight import compute_class_weight
from src import preprocessing
from xgboost import XGBClassifier


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
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    # class_weights[2] *= 2
    model = SVC(
        probability=True,
        class_weight=class_weights,
        C=10,
        kernel="poly",
        gamma="scale",
        degree=6,
        random_state=42,
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


def logistic_classifier(x_train, y_train):
    model = LogisticRegression(
        multi_class="multinomial",  # or 'ovr'
        solver="lbfgs",  # recommended for multinomial
        max_iter=1000,
    )
    return model.fit(x_train, y_train)


def xgboost_classifier_grid_search(x_train, y_train):
    model = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=len(set(y_train)),
        use_label_encoder=False,
        random_state=42,
    )
    # Best parameters: {'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.2, 'max_depth': 6, 'n_estimators': 200, 'reg_alpha': 0.1, 'reg_lambda': 0, 'subsample': 0.8}
    # Best parameters: {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.2, 'max_depth': 9, 'n_estimators': 300, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'subsample': 0.9}
    # {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.2, 'max_depth': 9, 'n_estimators': 400, 'reg_alpha': 0.1, 'reg_lambda': 0, 'subsample': 0.8}
    # {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.2, 'max_depth': 9, 'n_estimators': 500, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'subsample': 0.9}
    # Best parameters: {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.2, 'max_depth': 9, 'n_estimators': 500, 'reg_alpha': 0, 'reg_lambda': 0, 'subsample': 0.9}
    param_grid = {
        "n_estimators": [400, 500],
        "max_depth": [9],
        "learning_rate": [0.2],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [1.0],
        "gamma": [0],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [0, 0.1],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=10,
    )

    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_

    print("Best parameters:", grid_search.best_params_)
    print("Best weighted F1 CV score:", grid_search.best_score_)

    return best_model


def xgboost_classifier(x_train, y_train):
    model = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=len(set(y_train)),
        use_label_encoder=False,
        random_state=42,
        colsample_bytree=1.0,
        gamma=0,
        learning_rate=0.2,
        max_depth=9,
        n_estimators=500,
        reg_alpha=0,
        reg_lambda=0,
        subsample=0.9,
    )
    return model.fit(x_train, y_train)


def train(path, gender, age, grid_search, model_type):
    x_train, x_test, x_val, y_train, y_test, y_val = (
        preprocessing.preprocessing_features(path, gender, age, model_type)
    )

    if grid_search and model_type != "xgboost":
        raise ValueError(
            f"Grid search is only implemented for 'xgboost', not {model_type}"
        )

    if model_type == "xgboost":
        if grid_search:
            best_classifier = xgboost_classifier_grid_search(x_train, y_train)
        else:
            best_classifier = xgboost_classifier(x_train, y_train)
    elif model_type == "logistic":
        best_classifier = logistic_classifier(x_train, y_train)
    elif model_type == "svc":
        best_classifier = svc_classifier(x_train, y_train)
    elif model_type == "gmm":
        best_classifier = gmm_classifier(x_train, y_train)
    else:
        raise ValueError(
            "Invalid model_type. Choose 'svc' or 'gmm' or 'xgboost' or logistic."
        )

    if gender:
        pickle.dump(
            best_classifier, open(f"./data/model_{model_type}_gender.pkl", "wb")
        )
        print(f"Model saved as model_{model_type}_gender.pkl")
    elif age:
        pickle.dump(best_classifier, open(f"./data/model_{model_type}_age.pkl", "wb"))
        print(f"Model saved as model_{model_type}_age.pkl")
    else:
        pickle.dump(best_classifier, open(f"./data/model_{model_type}.pkl", "wb"))
        print(f"Model saved as model_{model_type}.pkl")
    return x_test, x_val, y_test, y_val
