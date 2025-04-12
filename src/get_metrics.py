import pandas as pd
import seaborn as sns


def plot_data_distribution(metadata_path: str) -> None:
    df = pd.read_csv(metadata_path + "/extracted_features.csv")
    sns.countplot(x=df["label"])
