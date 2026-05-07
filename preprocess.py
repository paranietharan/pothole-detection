import shutil
import logging
from tqdm import tqdm

import config
from utils.voc_to_yolo import voc_to_yolo
from utils.cleaner import clean_dataset
from utils.splitter import split_dataset

def setup_logging():
    config.LOG_DIR.mkdir(exist_ok=True)
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
    logging.info("Starting RDD-2022 Preprocessing Pipeline")

    # 1. Dataset Preparation - require the raw dataset to be downloaded first
    if not config.RAW_IMAGES_DIR.exists() or not config.RAW_ANNOTATIONS_DIR.exists():
        logging.error(f"Raw data not found at {config.RAW_ROOT}.")
        logging.info("Run 'python download_dataset.py' first, or copy the extracted Kaggle dataset into the raw data folder with train/images and train/annotations.")
        return

    # Create workspace directories
    config.WORK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    config.WORK_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Annotation Conversion (Pascal VOC -> YOLO)
    logging.info("Step 1: Converting Pascal VOC to YOLO format...")
    xml_files = list(config.RAW_ANNOTATIONS_DIR.glob("*.xml"))
    
    for xml_path in tqdm(xml_files, desc="Converting XMLs"):
        img_stem = xml_path.stem
        # Check if corresponding image exists (could be .jpg or .JPG)
        img_path = config.RAW_IMAGES_DIR / f"{img_stem}.jpg"
        if not img_path.exists():
            img_path = config.RAW_IMAGES_DIR / f"{img_stem}.JPG"
            
        if not img_path.exists():
            logging.warning(f"Image not found for {xml_path}. Skipping.")
            continue
            
        # Copy image to workspace
        shutil.copy(str(img_path), config.WORK_IMAGES_DIR / f"{img_stem}.jpg")
        
        # Convert XML to YOLO .txt
        output_txt = config.WORK_LABELS_DIR / f"{img_stem}.txt"
        voc_to_yolo(xml_path, output_txt, config.CLASS_MAP)

    # 3. Data Cleaning & Validation
    logging.info("Step 2: Cleaning dataset and detecting duplicates...")
    valid_images = clean_dataset(
        config.WORK_IMAGES_DIR, 
        config.WORK_LABELS_DIR, 
        config.QUARANTINE_DIR,
        num_classes=len(config.CLASS_MAP)
    )

    # 4. Train / Val / Test Split
    logging.info("Step 3: Splitting dataset...")
    split_dataset(
        valid_images, 
        config.WORK_LABELS_DIR, 
        config.DATASET_ROOT, 
        config.SPLIT_RATIOS,
        seed=config.RANDOM_SEED
    )

    logging.info("Pipeline completed successfully!")
    logging.info(f"Final dataset at: {config.DATASET_ROOT}")
    logging.info(f"Quarantined files (errors/duplicates) at: {config.QUARANTINE_DIR}")

if __name__ == "__main__":
    main()
