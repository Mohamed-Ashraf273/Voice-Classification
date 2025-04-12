import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_data_distribution(path: str) -> None:
    df = pd.read_csv(path)
    sns.countplot(x=df["label"])
    plt.show()
