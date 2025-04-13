from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def train_classifier(x_train, y_train):
    model = SVC(probability=True, C=10)
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
