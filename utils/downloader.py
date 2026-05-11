import logging
import os
from pathlib import Path

def has_kaggle_credentials() -> bool:
    return bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))

def download_dataset(dataset_id, download_path):
    """
    Downloads and extracts a Kaggle dataset.
    Supports Kaggle auth via `KAGGLE_USERNAME`/`KAGGLE_KEY` in `.env`,
    `kaggle auth login`, or `~/.kaggle/kaggle.json`.
    """
    if not has_kaggle_credentials():
        logging.error("Kaggle credentials are not configured.")
        logging.info("Set KAGGLE_USERNAME and KAGGLE_KEY in .env, run 'kaggle auth login', or place the dataset under the configured raw data folder.")
        return False

    try:
        import kaggle
    except ImportError:
        logging.error("Kaggle package not found. Please install it with 'pip install kaggle'.")
        return False
    except Exception as e:
        logging.error(f"Kaggle API error: {e}")
        return False

    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Downloading dataset '{dataset_id}' from Kaggle...")
    try:
        # Download
        kaggle.api.dataset_download_files(dataset_id, path=str(download_path), unzip=True)
        logging.info(f"Dataset downloaded and extracted to {download_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        return False
