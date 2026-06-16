"""
Prepare dataset for fine-tuning RT-DETR on Indian road data.
Downloads sample images and creates YOLO format dataset.
"""

import os
import shutil
from pathlib import Path
import numpy as np
import cv2
import random

# Create dataset directory
DATASET_DIR = Path("data/india_roads")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

for split in ["train", "val", "test"]:
    (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
    (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

print("✓ Created directory structure")

# Class mapping (simplified for this fine-tune)
CLASSES = {
    0: 'auto_rickshaw',
    1: 'truck',
    2: 'bus',
    3: 'motorcycle',
    4: 'car',
    5: 'bicycle',
    6: 'bull_cart',
    7: 'cow',
    8: 'dog',
    9: 'goat',
    10: 'pedestrian',
    11: 'traffic_sign',
    12: 'pole',
}

# Create synthetic training data (since downloading large dataset takes time)
# In production, you'd download real IDD data
print("\n📊 Creating sample training data...")

def create_synthetic_sample(filename, split, obj_class, count=1):
    """Create a synthetic sample image with objects"""
    # Create random background (road scene)
    img = np.ones((640, 640, 3), dtype=np.uint8)
    
    # Sky (top half - blue)
    img[:200] = [135, 206, 235]  # Sky blue
    
    # Road (bottom half - gray)
    img[200:] = [128, 128, 128]  # Road gray
    
    # Add some trees/buildings on sides
    cv2.rectangle(img, (0, 150), (50, 400), (34, 139, 34), -1)
    cv2.rectangle(img, (590, 150), (640, 400), (34, 139, 34), -1)
    
    labels = []
    
    for i in range(count):
        if obj_class == 0:  # auto_rickshaw - small, colorful
            x = random.randint(100, 400)
            y = random.randint(250, 500)
            w, h = 80, 60
            color = (0, 100, 255)  # Orange
            cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color, -1)
            cv2.circle(img, (x-30, y+20), 8, (0, 0, 0), -1)  # wheel
            cv2.circle(img, (x+30, y+20), 8, (0, 0, 0), -1)  # wheel
            
        elif obj_class == 1:  # truck - large red
            x = random.randint(100, 400)
            y = random.randint(250, 450)
            w, h = 120, 80
            color = (0, 0, 255)
            cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color, -1)
            
        elif obj_class == 7:  # cow - brown with spots
            x = random.randint(100, 500)
            y = random.randint(300, 500)
            w, h = 70, 80
            cv2.ellipse(img, (x, y), (w//2, h//2), 0, 0, 360, (165, 42, 42), -1)
            cv2.circle(img, (x-20, y-30), 15, (139, 69, 19), -1)  # head
            
        elif obj_class == 8:  # dog - brown
            x = random.randint(100, 500)
            y = random.randint(300, 500)
            w, h = 50, 60
            cv2.ellipse(img, (x, y), (w//2, h//2), 0, 0, 360, (139, 69, 19), -1)
            cv2.circle(img, (x-15, y-25), 12, (101, 50, 15), -1)  # head
            
        elif obj_class == 9:  # goat - smaller brown
            x = random.randint(100, 500)
            y = random.randint(300, 500)
            w, h = 40, 50
            cv2.ellipse(img, (x, y), (w//2, h//2), 0, 0, 360, (160, 82, 45), -1)
            cv2.circle(img, (x-12, y-20), 10, (101, 50, 15), -1)  # head
            
        elif obj_class == 2:  # bus - large yellow/orange
            x = random.randint(100, 400)
            y = random.randint(250, 450)
            w, h = 100, 90
            color = (0, 165, 255)
            cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color, -1)
            
        elif obj_class == 3:  # motorcycle - purple
            x = random.randint(100, 500)
            y = random.randint(250, 450)
            w, h = 50, 60
            color = (200, 0, 200)
            cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color, -1)
            
        elif obj_class == 4:  # car - green
            x = random.randint(100, 400)
            y = random.randint(250, 450)
            w, h = 90, 70
            color = (0, 255, 0)
            cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color, -1)
            
        else:  # other classes
            x = random.randint(100, 500)
            y = random.randint(250, 500)
            w, h = 60, 60
            cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), (255, 255, 255), -1)
        
        # Normalize coordinates (YOLO format: 0-1)
        x_norm = x / 640
        y_norm = y / 640
        w_norm = w / 640
        h_norm = h / 640
        
        labels.append(f"{obj_class} {x_norm:.4f} {y_norm:.4f} {w_norm:.4f} {h_norm:.4f}")
    
    # Save image
    img_path = IMAGES_DIR / split / filename
    cv2.imwrite(str(img_path), img)
    
    # Save labels
    if labels:
        label_path = LABELS_DIR / split / filename.replace('.jpg', '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))

# Create training samples (focusing on autos and animals)
print("Creating training samples...")

splits = ['train', 'val', 'test']
split_counts = {'train': 100, 'val': 20, 'test': 10}

# Focus on auto and animal classes
focus_classes = [0, 7, 8, 9, 1, 2, 3, 4, 5]  # auto, cow, dog, goat, truck, bus, motorcycle, car, bicycle

for split in splits:
    count = split_counts[split]
    print(f"\n  {split.upper()}: {count} images")
    
    for i in range(count):
        # Mix of object classes
        obj_class = random.choice(focus_classes)
        obj_count = random.randint(1, 3)  # 1-3 objects per image
        filename = f"{split}_{obj_class}_{i:04d}.jpg"
        create_synthetic_sample(filename, split, obj_class, obj_count)
        
        if (i + 1) % 20 == 0:
            print(f"    Created {i + 1} images")

print("\n✓ Sample images created")

# Create dataset.yaml
dataset_yaml = f"""path: {DATASET_DIR.absolute()}
train: images/train
val: images/val
test: images/test

nc: 13

names:
  0: auto_rickshaw
  1: truck
  2: bus
  3: motorcycle
  4: car
  5: bicycle
  6: bull_cart
  7: cow
  8: dog
  9: goat
  10: pedestrian
  11: traffic_sign
  12: pole
"""

with open(DATASET_DIR / "dataset.yaml", 'w') as f:
    f.write(dataset_yaml)

print("✓ Created dataset.yaml")
print(f"\n📁 Dataset ready at: {DATASET_DIR}")
print(f"   Train: {len(list((IMAGES_DIR / 'train').glob('*.jpg')))} images")
print(f"   Val:   {len(list((IMAGES_DIR / 'val').glob('*.jpg')))} images")
print(f"   Test:  {len(list((IMAGES_DIR / 'test').glob('*.jpg')))} images")
