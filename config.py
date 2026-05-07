"""
Central configuration for the RDD-2022 → YOLOv8 preprocessing pipeline.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(env_file: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if it exists."""
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()

RAW_ROOT = Path(os.getenv("RDD2022_ROOT", "./dataset/raw/RDD2022"))

# Subfolder layout expected inside RAW_ROOT
RAW_IMAGES_DIR = RAW_ROOT / "train" / "images"
RAW_ANNOTATIONS_DIR = RAW_ROOT / "train" / "annotations"

# Intermediate workspace
WORK_DIR = Path(os.getenv("WORK_DIR", "./workspace"))
WORK_IMAGES_DIR = WORK_DIR / "images"        # flat copy of raw images
WORK_LABELS_DIR = WORK_DIR / "labels"        # converted YOLO .txt files
QUARANTINE_DIR = WORK_DIR / "quarantine"     # problematic pairs moved here

# Final YOLO dataset output
DATASET_ROOT = Path(os.getenv("YOLO_DATASET_ROOT", "./dataset"))

# Class filter
CLASS_MAP: dict[str, int] = {
    "D40": 0,   
}
# Train / Val / Test split ratios (must sum to 1.0)
SPLIT_RATIOS = {
    "train": 0.70,
    "val":   0.20,
    "test":  0.10,
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Duplicate detection
# Perceptual hash size (higher → more sensitive, slower)
PHASH_HASH_SIZE = 8
# Two images are considered duplicates if their hash distance ≤ this value
PHASH_DISTANCE_THRESHOLD = 5

# Logging
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_FILE = LOG_DIR / "preprocess.log"

KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "aliabdelmenam/rdd-2022")