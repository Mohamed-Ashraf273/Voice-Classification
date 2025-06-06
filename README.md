# WIT: Gender & Age Recognition

Who Is Talking (WIT) is a project aims to classify audio input and predict the **gender**, and **age group** using machine learning techniques.

## Project Overview

We built a pipeline that takes in voice data and classifies:

- **Speaker Gender** (Male / Female)
- **Age Group** (Twenties / Fifties)

## Installation

```bash
git clone https://github.com/Mohamed-Ashraf273/Who-Is-Talking.git
cd Who-Is-Talking
mkdir data
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```
## How to Prepare Your Data for Training

If you plan to train the model on your own dataset, follow these steps:

1. **Create a `data_train/` directory** in the root of the repository.
2. **Place your audio files** inside one or more subdirectories under `data_train/`, and make sure each folder name **starts with `audio`** (e.g., `audio_batch1`, `audio_train_set`, etc.).
3. **It must also contain a `.csv` file** that holds the corresponding labels for the audio files.

### Example Directory Structure
```
Who-Is-Talking/
├── data_train/
│   ├── audio_batch1/
│   │   ├── sample_001.wav
│   │   ├── sample_002.wav
│   │   ├── labels.csv
│   ├── filtered_data_labeled.tsv
├── model/
│   ├── model.pkl
│   │── scaler.pkl
├── src
│   infer.py
│   main.py
│   pipeline.py
├── README.md
```
## To train your own data run:
```bash
./wit train --features './path_to_features.csv' --model model_selected
```
1) model by default is "stacking".
2) if you want to grid search over "xgboost" or "lgbm" add "--grid_search" when running the command, grid_search is supported for "xgboost" and "lgbm" only.
3) for saving files like "test_val.json" or "test_data.json" add "--save_val" for test_val and "--save_test" for test_data.
## To Extract features run:
```bash
./wit features --datapath './data_train/audio_batches' 
```
1) your batches name should start with "audio".
2) in your audio_batches dir you should include "filtered_data_labeled.tsv" contains your data.

## To get performance run:
first check that the models in the ```model``` path.
```bash
./wit validate --testfile './test_val.json' --model model_selected --val
```
1) your validation file should be a json file.
2) model by default "stacking".

## To get predictions file run:
first check that the models in the ```model``` dir.
```bash
./wit predict --audiopath './path_to_the_dir_contains_audios'
```
## For inference you could also build our docker image then run it:
before you build the image make sure that the test path dir in ```infer.py``` is setted correctly.
```bash
docker build -t wit .
docker run -v "C:\path_to_the_project:/app" wit
```
**Note: make sure you installed docker**

**OR**   
run the infer.exe file, but make sure that the test audios are in **```./data```** dir.
## Note: for developers
we are on 'black' format.
```bash
pip install black
python black ./path_to the_file
```
