import random
import shutil
from pathlib import Path
import logging
from tqdm import tqdm

def split_dataset(image_paths, labels_dir, output_root, ratios, seed=42):
    """
    Splits images and labels into train/val/test folders.
    ratios: dict like {'train': 0.7, 'val': 0.2, 'test': 0.1}
    """
    random.seed(seed)
    random.shuffle(image_paths)
    
    total = len(image_paths)
    train_end = int(total * ratios['train'])
    val_end = train_end + int(total * ratios['val'])
    
    splits = {
        'train': image_paths[:train_end],
        'val': image_paths[train_end:val_end],
        'test': image_paths[val_end:]
    }
    
    output_root = Path(output_root)
    
    for split_name, paths in splits.items():
        if not paths: continue
        
        img_out = output_root / "images" / split_name
        lbl_out = output_root / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Copying {len(paths)} files to {split_name} split...")
        for img_path in tqdm(paths, desc=f"Splitting {split_name}"):
            # Copy image
            shutil.copy(str(img_path), img_out / img_path.name)
            
            # Copy label
            lbl_path = Path(labels_dir) / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy(str(lbl_path), lbl_out / lbl_path.name)
            else:
                # Create empty label if it doesn't exist (background)
                (lbl_out / f"{img_path.stem}.txt").touch()

    return splits
