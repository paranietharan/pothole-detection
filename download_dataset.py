import logging

import config
from utils.downloader import download_dataset


def setup_logging():
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ]
    )


def main():
    setup_logging()
    logging.info("Starting RDD-2022 download")
    logging.info(f"Raw dataset will be stored at {config.RAW_ROOT.resolve()}")

    if config.RAW_IMAGES_DIR.exists() and config.RAW_ANNOTATIONS_DIR.exists():
        logging.info("Raw dataset already exists. Nothing to download.")
        return

    success = download_dataset(config.KAGGLE_DATASET, config.RAW_ROOT)
    if not success:
        logging.error("Dataset download failed.")
        return

    logging.info("Dataset download completed successfully.")


if __name__ == "__main__":
    main()