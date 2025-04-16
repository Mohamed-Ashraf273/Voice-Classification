import json
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def plot_data_distribution(path: str) -> None:
    df = pd.read_csv(path)
    sns.countplot(x=df["label"])
    plt.show()


def show(model, x_test, y_test, gfas, gender_model, age_model):
    if gfas:
        assert (
            gender_model is not None and age_model is not None
        ), "you should provide gender model and age model"
        gender = gender_model.predict(x_test)
        age = age_model.predict(x_test)
        y_pred = 3 - (gender + 2 * age)
    else:
        y_pred = model.predict(x_test)
    print("accuracy report:", accuracy_score(y_test, y_pred))
    print("classification report:", classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def get_predictions(model, x_test):
    predictions = model.predict(x_test)
    return predictions
