from src import feature_extraction, get_metrics
from pathlib import Path
import time


def extract_features(datapath):
    start = time.time()
    metadata_path = datapath
    output_path = Path(__file__).resolve().parent / "features_multi.csv"
    feature_extraction.get_features(metadata_path, output_path, max_workers=12)
    print(f"Time taken: {time.time() - start:.2f} seconds")


def plot_data_distribution(path):
    get_metrics.plot_data_distribution(path)


# datapath = "data/uncompressed/"
plot_data_distribution("data\\features_all.csv")
