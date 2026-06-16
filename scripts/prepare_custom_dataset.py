
import os
import shutil
import random
from pathlib import Path
import yaml

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Source directories for images
SRC_DATA_DIR = BASE_DIR / "data"
SRC_IMAGE_DIRS = [
    SRC_DATA_DIR / "auto_test",
    SRC_DATA_DIR / "auto_train",
    SRC_DATA_DIR / "bus_test",
    SRC_DATA_DIR / "bus_train",
    SRC_DATA_DIR / "cars_test",
    SRC_DATA_DIR / "cars_train",
    SRC_DATA_DIR / "images_gen",
    SRC_DATA_DIR / "lorry_test",
    SRC_DATA_DIR / "lorry_train",
]
SRC_LABELS_DIR = SRC_DATA_DIR / "labels_gen"

# Destination directory for the structured dataset
DEST_DATASET_DIR = BASE_DIR / "custom_dataset"
DEST_IMAGES_DIR = DEST_DATASET_DIR / "images"
DEST_LABELS_DIR = DEST_DATASET_DIR / "labels"

# Validation split percentage
VAL_SPLIT = 0.2

# Class information (from train.py and config.ini)
CLASSES = {
    0: 'auto_rickshaw', 1: 'truck', 2: 'bus', 3: 'motorcycle', 4: 'car',
    5: 'bicycle', 6: 'bull_cart', 7: 'cow', 8: 'dog', 9: 'goat',
    10: 'pedestrian', 11: 'traffic_sign', 12: 'pole'
}

def create_dataset():
    """
    Creates a structured dataset for YOLO training from multiple source
    directories.
    """
    print("Clearing and recreating dataset directory structure...")
    if DEST_DATASET_DIR.exists():
        shutil.rmtree(DEST_DATASET_DIR)
        
    for split in ["train", "val"]:
        (DEST_IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (DEST_LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    print("Reading image files from all source directories...")
    image_files = []
    for src_dir in SRC_IMAGE_DIRS:
        if src_dir.exists():
            image_files.extend(list(src_dir.glob("*.jpg")))
            image_files.extend(list(src_dir.glob("*.png")))
            image_files.extend(list(src_dir.glob("*.jpeg")))
            image_files.extend(list(src_dir.glob("*.JPG")))
            image_files.extend(list(src_dir.glob("*.PNG")))

    # Remove duplicates
    image_files = sorted(list(set(image_files)))
    random.shuffle(image_files)

    split_index = int(len(image_files) * (1 - VAL_SPLIT))
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    print(f"Total images found: {len(image_files)}")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")

    def copy_files(file_list, split_name):
        print(f"Copying {split_name} files...")
        copied_count = 0
        for img_path in file_list:
            label_path = SRC_LABELS_DIR / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy(img_path, DEST_IMAGES_DIR / split_name / img_path.name)
                shutil.copy(label_path, DEST_LABELS_DIR / split_name / label_path.name)
                copied_count += 1
        print(f"Copied {copied_count} images and labels for {split_name}.")

    copy_files(train_files, "train")
    copy_files(val_files, "val")

    print("Creating custom_dataset.yaml...")
    dataset_yaml_path = DEST_DATASET_DIR / "custom_dataset.yaml"
    dataset_yaml_content = {
        'path': str(DEST_DATASET_DIR.absolute()),
        'train': str(DEST_IMAGES_DIR / 'train'),
        'val': str(DEST_IMAGES_DIR / 'val'),
        'nc': len(CLASSES),
        'names': CLASSES
    }

    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_yaml_content, f, default_flow_style=False)

    print(f"Dataset creation complete! YAML at: {dataset_yaml_path}")

if __name__ == "__main__":
    create_dataset()
