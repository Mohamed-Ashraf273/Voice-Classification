import subprocess
import time


def infer(run_inference_script, test_dir, model="stacking"):
    command = ["python", run_inference_script, test_dir, model]
    start = time.time()
    subprocess.run(command, check=True)
    end = time.time()
    with open("./time.txt", "w") as f:
        f.write(str(round(end - start, 3)))


if __name__ == "__main__":
    infer("run_inference.py", "./data")
