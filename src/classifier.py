import numpy as np
import pickle
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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
    model = SVC(
        probability=True,
        class_weight=class_weights,
        C=10,
        kernel="poly",
        gamma="scale",
        degree=6,
        random_state=42,
    )
    return model.fit(x_train, y_train)


def logistic_classifier(x_train, y_train):
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
    )
    return model.fit(x_train, y_train)


def xgboost_classifier_grid_search(x_train, y_train):
    print("Grid search on xgboost started: ")
    model = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=len(set(y_train)),
        use_label_encoder=False,
        random_state=42,
    )
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


def lgbm_classifier_grid_search(x_train, y_train):
    print("Grid search on lgbm started: ")
    model = LGBMClassifier(
        objective="multiclass", num_class=len(set(y_train)), random_state=42, verbose=-1
    )

    param_grid = {
        "n_estimators": [500, 600],
        "max_depth": [10, 11],
        "learning_rate": [0.3, 0.4],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [1.0],
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


def lgbm_classifier(x_train, y_train):
    model = LGBMClassifier(
        objective="multiclass",
        num_class=len(set(y_train)),
        random_state=42,
        n_estimators=500,
        max_depth=9,
        learning_rate=0.2,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_alpha=0,
        reg_lambda=0,
        verbose=-1,
    )
    return model.fit(x_train, y_train)


def stacking(x_train, y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    clf1 = LGBMClassifier(
        objective="multiclass",
        num_class=len(set(y_train)),
        random_state=42,
        n_estimators=500,
        max_depth=9,
        learning_rate=0.2,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_alpha=0,
        reg_lambda=0,
        verbose=-1,
        class_weight=class_weights,
    )
    clf2 = LGBMClassifier(
        objective="multiclass",
        num_class=len(set(y_train)),
        random_state=42,
        n_estimators=600,
        max_depth=9,
        learning_rate=0.3,
        subsample=0.9,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=0,
        verbose=-1,
        class_weight=class_weights,
    )
    clf3 = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=len(set(y_train)),
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
    clf4 = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=len(set(y_train)),
        random_state=42,
        colsample_bytree=1.0,
        gamma=0,
        learning_rate=0.3,
        max_depth=9,
        n_estimators=600,
        reg_alpha=0.1,
        reg_lambda=0,
        subsample=0.9,
    )
    clf5 = MLPClassifier(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=128,
        learning_rate="adaptive",
        random_state=42,
        max_iter=400,
        early_stopping=True,
    )
    clf6 = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42,
        learning_rate=0.2,
        depth=9,
        iterations=500,
        l2_leaf_reg=0,
        rsm=1.0,
        subsample=0.9,
        bootstrap_type="Bernoulli",
        verbose=0,
        class_weights=list(weights),
    )
    clf7 = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight=class_weights,
    )
    clf8 = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1)

    base_classifiers = [
        ("LGBM1", clf1),
        ("LGBM2", clf2),
        ("XGBOOST1", clf3),
        ("XGBOOST2", clf4),
        ("MLP", clf5),
        ("CATBOOST", clf6),
        ("RANDOMFOREST", clf7),
        ("KNN", clf8),
    ]

    final_est = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        num_class=len(set(y_train)),
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

    stacking_clf = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=final_est,
        stack_method="predict",
        passthrough=True,
        n_jobs=-1,
        verbose=10,
    )
    return stacking_clf.fit(x_train, y_train)


def mlp_classifier(x_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        solver="adam",
        batch_size=50,
        learning_rate="adaptive",
        random_state=42,
        max_iter=350,
        early_stopping=True,
    )
    return model.fit(x_train, y_train)


def mlp_classifier_grid_search(x_train, y_train):
    print("Grid search on mlp started: ")
    model = MLPClassifier(
        solver="adam",
        random_state=42,
        early_stopping=True,
    )
    param_grid = {
        "hidden_layer_sizes": [(128, 128), (256, 256), (512, 512)],
        "activation": ["relu", "tanh"],
        "alpha": [0.001, 0.01, 0.1],
        "learning_rate": ["constant", "adaptive"],
        "batch_size": [128, 256, 512],
        "max_iter": [200, 400, 500],
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


def train(path, gender, age, grid_search, model_type):
    x_train, x_test, x_val, y_train, y_test, y_val = (
        preprocessing.preprocessing_features(path, gender, age, model_type)
    )

    if grid_search and model_type not in ["xgboost", "lgbm", "mlp"]:
        raise ValueError(
            f"Grid search is only implemented for 'xgboost', 'lgbm' and 'mlp, not {model_type}"
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
    elif model_type == "lgbm":
        if grid_search:
            best_classifier = lgbm_classifier_grid_search(x_train, y_train)
        else:
            best_classifier = lgbm_classifier(x_train, y_train)
    elif model_type == "stacking":
        best_classifier = stacking(x_train, y_train)
    elif model_type == "mlp":
        if grid_search:
            best_classifier = mlp_classifier_grid_search(x_train, y_train)
        else:
            best_classifier = mlp_classifier(x_train, y_train)
    else:
        raise ValueError(
            "Invalid model_type. Choose 'svc', 'gmm', 'xgboost', 'lgbm', 'stacking', 'mlp' or 'logistic'."
        )

    if gender:
        pickle.dump(
            best_classifier, open(f"./model/model_{model_type}_gender.pkl", "wb")
        )
        print(f"Model saved as model_{model_type}_gender.pkl")
    elif age:
        pickle.dump(best_classifier, open(f"./model/model_{model_type}_age.pkl", "wb"))
        print(f"Model saved as model_{model_type}_age.pkl")
    else:
        pickle.dump(best_classifier, open(f"./model/model_{model_type}.pkl", "wb"))
        print(f"Model saved as model_{model_type}.pkl")
    return x_train, x_test, x_val, y_train, y_test, y_val
