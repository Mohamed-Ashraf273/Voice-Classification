import multiprocessing
import pipeline


def infer(test_dir, model="xgboost"):
    pipeline.final_out(test_file_path=test_dir, model_selected=model)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    infer(
        "./data"
    )  # this path could be changed, it has a default value "data" because the TA wanted that, recommend to change it to something else.
