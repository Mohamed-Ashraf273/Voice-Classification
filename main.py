from src import feature_extraction
from pathlib import Path
import time

start = time.time()
metadata_path = "data/uncompressed/"
output_path = Path(__file__).resolve().parent / "features_multi.csv"
feature_extraction.get_features(metadata_path, output_path,max_workers=12)
print(f"Time taken: {time.time() - start:.2f} seconds")
