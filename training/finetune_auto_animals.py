"""
Fast Fine-tuning Script for Auto-rickshaws and Animals Detection
Uses GPU for training
"""

import os
import sys
import shutil
from pathlib import Path
import yaml
import requests
import cv2
import numpy as np
from ultralytics import YOLO

# Create dataset directory
dataset_dir = Path("data/auto_animals")
dataset_dir.mkdir(parents=True, exist_ok=True)

# Create subdirectories
for split in ['train', 'val', 'test']:
    (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

print("✓ Dataset directory created")

# Create synthetic training data (auto-rickshaws and animals)
# This generates training examples programmatically

def create_synthetic_image(img_type, num=50):
    """Create synthetic training images"""
    for i in range(num):
        # Create image
        img = np.ones((640, 640, 3), dtype=np.uint8) * 200  # Light gray background
        
        if img_type == 'auto':
            # Draw auto-rickshaw (orange vehicle)
            x1, y1 = np.random.randint(50, 300), np.random.randint(50, 300)
            cv2.rectangle(img, (x1, y1), (x1+200, y1+150), (0, 165, 255), -1)  # Orange
            cv2.circle(img, (x1+50, y1+130), 30, (0, 0, 0), -1)  # Wheel
            cv2.circle(img, (x1+150, y1+130), 30, (0, 0, 0), -1)  # Wheel
            cv2.rectangle(img, (x1+40, y1+30), (x1+160, y1+90), (200, 200, 255), -1)  # Cabin
            
            # Add road texture
            cv2.line(img, (0, 400), (640, 400), (100, 100, 100), 5)
            cv2.line(img, (0, 410), (640, 410), (255, 255, 255), 2)
            
            # Annotation: auto_rickshaw at (x_center, y_center, width, height) normalized
            x_center = (x1 + 100) / 640
            y_center = (y1 + 75) / 640
            width = 200 / 640
            height = 150 / 640
            
            split = np.random.choice(['train', 'val'], p=[0.8, 0.2])
            img_path = dataset_dir / 'images' / split / f'auto_{i:04d}.jpg'
            label_path = dataset_dir / 'labels' / split / f'auto_{i:04d}.txt'
            
            cv2.imwrite(str(img_path), img)
            with open(label_path, 'w') as f:
                f.write(f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")
        
        elif img_type == 'cow':
            # Draw cow (brown)
            x1, y1 = np.random.randint(50, 350), np.random.randint(50, 350)
            cv2.ellipse(img, (x1+100, y1+80), (90, 70), 0, 0, 360, (42, 42, 165), -1)  # Body
            cv2.circle(img, (x1+150, y1+50), 40, (42, 42, 165), -1)  # Head
            cv2.circle(img, (x1+60, y1+140), 25, (0, 0, 0), -1)  # Leg
            cv2.circle(img, (x1+130, y1+145), 25, (0, 0, 0), -1)  # Leg
            cv2.circle(img, (x1+170, y1+140), 25, (0, 0, 0), -1)  # Leg
            
            # Add road
            cv2.line(img, (0, 400), (640, 400), (100, 100, 100), 5)
            
            x_center = (x1 + 100) / 640
            y_center = (y1 + 90) / 640
            width = 180 / 640
            height = 160 / 640
            
            split = np.random.choice(['train', 'val'], p=[0.8, 0.2])
            img_path = dataset_dir / 'images' / split / f'cow_{i:04d}.jpg'
            label_path = dataset_dir / 'labels' / split / f'cow_{i:04d}.txt'
            
            cv2.imwrite(str(img_path), img)
            with open(label_path, 'w') as f:
                f.write(f"1 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")
        
        elif img_type == 'dog':
            # Draw dog
            x1, y1 = np.random.randint(50, 350), np.random.randint(50, 350)
            cv2.ellipse(img, (x1+80, y1+80), (60, 50), 0, 0, 360, (128, 0, 128), -1)  # Body
            cv2.circle(img, (x1+120, y1+60), 30, (128, 0, 128), -1)  # Head
            cv2.circle(img, (x1+50, y1+120), 15, (0, 0, 0), -1)  # Leg
            cv2.circle(img, (x1+110, y1+125), 15, (0, 0, 0), -1)  # Leg
            cv2.circle(img, (x1+130, y1+125), 15, (0, 0, 0), -1)  # Leg
            cv2.line(img, (x1+140, y1+70), (x1+155, y1+50), (128, 0, 128), 3)  # Tail
            
            x_center = (x1 + 80) / 640
            y_center = (y1 + 85) / 640
            width = 140 / 640
            height = 130 / 640
            
            split = np.random.choice(['train', 'val'], p=[0.8, 0.2])
            img_path = dataset_dir / 'images' / split / f'dog_{i:04d}.jpg'
            label_path = dataset_dir / 'labels' / split / f'dog_{i:04d}.txt'
            
            cv2.imwrite(str(img_path), img)
            with open(label_path, 'w') as f:
                f.write(f"2 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}\n")

print("\n🔄 Creating synthetic training data...")
print("  - Creating auto-rickshaws...")
create_synthetic_image('auto', num=40)
print("  - Creating cows...")
create_synthetic_image('cow', num=40)
print("  - Creating dogs...")
create_synthetic_image('dog', num=40)
print("✓ Generated 120 training images total")

# Create dataset.yaml
dataset_config = {
    'path': str(dataset_dir.absolute()),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 3,
    'names': {
        0: 'auto_rickshaw',
        1: 'cow',
        2: 'dog',
    }
}

with open(dataset_dir / 'dataset.yaml', 'w') as f:
    yaml.dump(dataset_config, f, default_flow_style=False)

print("\n✓ Created dataset.yaml")

# Fine-tune the model
def start_training():
    print("\n" + "="*70)
    print("STARTING FINE-TUNING ON GPU")
    print("="*70)

    model = YOLO('rtdetr-l.pt')

    print("\n🚀 Training configuration:")
    print("  Model: RT-DETR Large")
    print("  Dataset: Auto-rickshaws + Animals (Cows, Dogs)")
    print("  Training samples: 96 images")
    print("  Validation samples: 24 images")
    print("  Device: GPU (CUDA enabled)")
    print("  Epochs: 50")

    results = model.train(
        data=str(dataset_dir / 'dataset.yaml'),
        epochs=50,
        imgsz=640,
        batch=8,  # Reduced batch for GPU memory
        device=0,  # GPU device 0
        patience=10,
        save=True,
        cache=False,  # Changed to False for Windows
        workers=0,  # Changed to 0 for Windows multiprocessing
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        close_mosaic=10,
        resume=False,
    )

    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)

    best_weights = Path("runs/detect/train/weights/best.pt")
    if best_weights.exists():
        print(f"\n✓ Best weights saved to:")
        print(f"  {best_weights}")
        
        # Copy to custom_weights
        custom_weights_dir = Path("custom_weights")
        custom_weights_dir.mkdir(exist_ok=True)
        shutil.copy(best_weights, custom_weights_dir / "auto_animals_best.pt")
        print(f"\n✓ Copied to: custom_weights/auto_animals_best.pt")
        print("\n✓ To use these weights, update app.py:")
        print("  CUSTOM_WEIGHTS_PATH = 'custom_weights/auto_animals_best.pt'")
    else:
        print("✗ No weights found!")

    print("\n✓ Update app.py and restart to use new weights!")


if __name__ == '__main__':
    start_training()
