# Voice Classification: Gender & Age Recognition

This project aims to classify audio input and predict the **gender**, and **age group** using machine learning techniques.

## Project Overview

We built a pipeline that takes in voice data and classifies:

- **Speaker Gender** (Male / Female)
- **Age Group** (Twenties / Fifties)

## Installation

```bash
git clone https://github.com/Mohamed-Ashraf273/Voice-Classification.git
cd Voice-Classification
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
python main.py
```
**Note: for developers**
we are on 'black' format
```bash
pip install black
python black ./path_to the_file
```

## Docker Setup (Optional)

```bash
docker build -t Voice-Classification .
docker run -it voice-Classification
```
**Note: Make sure Docker is installed and running on your machine.**
