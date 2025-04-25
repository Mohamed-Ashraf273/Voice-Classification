import multiprocessing
import pipeline
import time


def infer(test_dir, model="stacking"):
    start = time.time()
    predictions = pipeline.final_out(test_file_path=test_dir, model_selected=model)
    end = time.time()
    with open(f"./results.txt", "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    with open("./time.txt", "w") as f:
        f.write(str(round(end - start, 3)))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    infer(
        "./data"
    )  # this path could be changed, it has a default value "data" because the TA wanted that, recommend to change it to something else.
