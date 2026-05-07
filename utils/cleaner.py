import cv2
import os
import shutil
import logging
from pathlib import Path
import imagehash
from PIL import Image
from tqdm import tqdm

def check_image_integrity(image_path):
    """Checks if the image can be opened by OpenCV."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        return True
    except Exception:
        return False

def validate_yolo_label(label_path, num_classes=1):
    """
    Validates YOLO label content:
    - 5 values per line
    - Class ID is within range
    - Coordinates are [0, 1]
    - Width/Height > 0
    """
    try:
        if not label_path.exists():
            return True # Empty labels are valid background
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            if cls_id < 0 or cls_id >= num_classes:
                return False
                
            for c in coords:
                if not (0 <= c <= 1):
                    return False
            
            # w and h (indices 2 and 3 in parts[1:])
            if coords[2] <= 0 or coords[3] <= 0:
                return False
        return True
    except Exception:
        return False

def find_duplicates(image_paths, hash_size=8, threshold=5):
    """
    Finds near-duplicate images using perceptual hashing.
    Returns a list of image paths to remove.
    """
    hashes = {}
    to_remove = []
    
    logging.info("Calculating perceptual hashes for duplicate detection...")
    for img_path in tqdm(image_paths, desc="Hashing images"):
        try:
            with Image.open(img_path) as img:
                h = imagehash.phash(img, hash_size=hash_size)
                
            is_dup = False
            for prev_path, prev_hash in hashes.items():
                if h - prev_hash <= threshold:
                    to_remove.append(img_path)
                    is_dup = True
                    break
            
            if not is_dup:
                hashes[img_path] = h
        except Exception as e:
            logging.warning(f"Could not hash {img_path}: {e}")
            
    return to_remove

def clean_dataset(images_dir, labels_dir, quarantine_dir, num_classes=1):
    """
    Main cleaning routine.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    quarantine_dir = Path(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(images_dir.glob("*.jpg"))
    valid_pairs = []

    logging.info(f"Starting cleaning on {len(all_images)} images...")

    for img_path in tqdm(all_images, desc="Cleaning"):
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"
        
        # 1. Check Integrity & Label Validity
        is_ok = True
        if not check_image_integrity(img_path):
            logging.warning(f"Integrity check failed: {img_path}")
            is_ok = False
        elif not validate_yolo_label(lbl_path, num_classes):
            logging.warning(f"Label validation failed: {lbl_path}")
            is_ok = False
            
        if not is_ok:
            # Move to quarantine
            shutil.move(str(img_path), quarantine_dir / img_path.name)
            if lbl_path.exists():
                shutil.move(str(lbl_path), quarantine_dir / lbl_path.name)
            continue
            
        valid_pairs.append(img_path)

    # 2. Duplicate detection among valid pairs
    to_remove = find_duplicates(valid_pairs)
    for img_path in to_remove:
        logging.info(f"Duplicate found: {img_path}. Moving to quarantine.")
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"
        
        shutil.move(str(img_path), quarantine_dir / img_path.name)
        if lbl_path.exists():
            shutil.move(str(lbl_path), quarantine_dir / lbl_path.name)
        
        valid_pairs.remove(img_path)

    return valid_pairs
