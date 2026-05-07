## Pothole detection using yolov8 models

Dataset used: https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022

## Setup

1. Copy `.env.example` to `.env` and fill in Kaggle credentials if you want to download with the API and install dependencies:
```bash
pip install -r requirements.txt
```
2. Download the raw dataset:
```bash
python download_dataset.py
```
3. Run preprocessing:
```bash
python preprocess.py
```