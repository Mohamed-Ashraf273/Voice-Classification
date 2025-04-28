import multiprocessing
import pipeline
import sys


def infer(test_dir, model="stacking"):
    predictions = pipeline.final_out(test_file_path=test_dir, model_selected=model)
    with open(f"./results.txt", "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    infer(data_dir)
