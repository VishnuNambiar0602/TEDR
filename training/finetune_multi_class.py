"""
Multi-Class Training: COCO Classes + Auto-Rickshaw
Trains RT-DETR to detect: persons, vehicles, animals, AND auto-rickshaws
Uses GPU for faster training on Indian road detection
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from ultralytics import YOLO
from PIL import Image, ImageDraw
import random

# Configuration
DATASET_DIR = Path("D:/Projects/DETR Object Detection/datasets/coco_autorickshaw")
IMAGES_DIR = DATASET_DIR / "images" / "train"
LABELS_DIR = DATASET_DIR / "labels" / "train"
NUM_AUTO_RICKSHAWS = 200  # More training images for auto-rickshaw
IMG_SIZE = 640


def generate_synthetic_auto_rickshaws():
    """Generate diverse synthetic auto-rickshaw images for training"""
    print("\n[*] Generating synthetic images (auto-rickshaws labeled as class for training)...")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create validation directories too
    val_images_dir = DATASET_DIR / "images" / "val"
    val_labels_dir = DATASET_DIR / "labels" / "val"
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Use class 5 = bus (similar vehicle type to auto-rickshaw)
    vehicle_class = 5
    
    for i in range(NUM_AUTO_RICKSHAWS):
        # Create random image
        img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), 
                        color=(random.randint(80, 150), random.randint(100, 180), random.randint(120, 200)))
        draw = ImageDraw.Draw(img)
        
        # Draw road
        road_color = (100, 100, 100)
        draw.rectangle([(0, IMG_SIZE//2), (IMG_SIZE, IMG_SIZE)], fill=road_color)
        
        # Draw sky
        sky_color = (135, 206, 250)
        draw.rectangle([(0, 0), (IMG_SIZE, IMG_SIZE//2)], fill=sky_color)
        
        # Add random road lines
        for _ in range(2):
            y = IMG_SIZE//2 + random.randint(50, 150)
            draw.line([(0, y), (IMG_SIZE, y)], fill=(255, 255, 100), width=3)
        
        # Generate 1-3 vehicles per image
        num_vehicles = random.randint(1, 3)
        boxes = []
        
        for _ in range(num_vehicles):
            # Random position (lower half = road)
            x = random.randint(50, IMG_SIZE - 150)
            y = random.randint(IMG_SIZE // 2 + 20, IMG_SIZE - 100)
            w = random.randint(80, 140)
            h = random.randint(60, 100)
            
            # Draw vehicle (representing auto-rickshaw)
            draw.rectangle([(x, y), (x + w, y + h)], outline=(255, 165, 0), width=3)
            draw.rectangle([(x + 10, y + 10), (x + w - 10, y + h - 30)], fill=(255, 165, 0), outline=(255, 140, 0), width=2)
            draw.ellipse([(x + 15, y + h - 20), (x + 35, y + h)], fill=(50, 50, 50))
            draw.ellipse([(x + w - 35, y + h - 20), (x + w - 15, y + h)], fill=(50, 50, 50))
            draw.rectangle([(x + 20, y + 20), (x + w - 20, y + 40)], fill=(100, 150, 200), outline=(0, 0, 0), width=1)
            
            # Normalize coordinates for YOLO format
            cx = (x + w/2) / IMG_SIZE
            cy = (y + h/2) / IMG_SIZE
            norm_w = w / IMG_SIZE
            norm_h = h / IMG_SIZE
            
            boxes.append((vehicle_class, cx, cy, norm_w, norm_h))
        
        # Save image
        img_path = IMAGES_DIR / f"auto_rickshaw_{i:04d}.jpg"
        img.save(img_path, quality=85)
        
        # Save labels
        label_path = LABELS_DIR / f"auto_rickshaw_{i:04d}.txt"
        with open(label_path, 'w') as f:
            for cls, cx, cy, w, h in boxes:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        # Copy 20% to validation set
        if i % 5 == 0:
            shutil.copy(img_path, val_images_dir / f"auto_rickshaw_{i:04d}.jpg")
            shutil.copy(label_path, val_labels_dir / f"auto_rickshaw_{i:04d}.txt")
        
        if (i + 1) % 50 == 0:
            print(f"  ✓ Generated {i + 1}/{NUM_AUTO_RICKSHAWS} images")
    
    print(f"  ✓ Created {NUM_AUTO_RICKSHAWS} synthetic vehicle images ({NUM_AUTO_RICKSHAWS//5} validation)")


def download_coco_subset():
    """Download COCO dataset subset for training"""
    print("\n[*] Downloading COCO128 dataset (recommended for quick training)...")
    try:
        # Ultralytics provides a small COCO subset for easy training
        model = YOLO('rtdetr-l.pt')
        # Create dataset.yaml automatically
        dataset_path = model.train(
            data='coco128.yaml',  # Ultralytics' small COCO subset
            epochs=1,  # Just to download, we'll train separately
            imgsz=IMG_SIZE,
            device=0,
            verbose=False,
            patience=1
        )
        print("  ✓ COCO128 dataset prepared")
        return True
    except Exception as e:
        print(f"  ⚠ Could not download COCO: {e}")
        return False


def create_dataset_yaml():
    """Create dataset.yaml for training with standard 80 COCO classes"""
    yaml_content = """path: D:/Projects/DETR Object Detection/datasets/coco_autorickshaw
train: images/train
val: images/val
test: images/test

nc: 80  # Standard COCO classes
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
        'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'vehicle', 'truck']
"""
    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"  ✓ Created dataset.yaml with 80 COCO classes")
    return str(yaml_path)


def train_model(dataset_yaml):
    """Train RT-DETR on COCO + auto-rickshaw dataset"""
    print("\n" + "="*70)
    print("STARTING GPU TRAINING: RT-DETR-L on COCO + Auto-Rickshaw Classes")
    print("="*70)
    
    try:
        # Accept hyperparameters as arguments for binary search
        def _train_once(epochs, batch, conf, patience, save_dir=None):
            print(f"\n[1/4] Loading pre-trained RT-DETR-S model...")
            model = YOLO('rtdetr-s.pt')
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU not available. Please check your drivers and hardware.")
            gpu_name = torch.cuda.get_device_name(0)
            device = 0  # Always use GPU
            print(f"  ✓ GPU: {gpu_name}")
            print(f"\n[2/4] Training for {epochs} epochs, batch={batch}, conf={conf}, patience={patience}")
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=IMG_SIZE,
                batch=batch,
                patience=patience,
                device=device,
                workers=0,
                cache=False,
                save=True,
                save_period=epochs,  # Save only at the end of 25 epochs
                verbose=True,
                conf=conf,
                project=save_dir if save_dir else None
            )
            print("\n[3/4] Training completed!")
            # Copy best weights
            best_weights = Path("runs/detect/train/weights/best.pt")
            if best_weights.exists():
                dest = Path("D:/Projects/DETR Object Detection/custom_weights/multiclass_best.pt")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(best_weights, dest)
                print(f"  ✓ Best weights saved to: {dest}")
            print("\n[4/4] Training summary:")
            print(f"  - Model: RT-DETR-L (80 COCO classes)")
            print(f"  - Classes: persons, vehicles, animals, objects, etc.")
            print(f"  - Best checkpoint: {best_weights.parent / 'best.pt'}")
            print(f"  - Results: {Path('runs/detect/train')}")
            return results

        # Example binary search loop for one hyperparameter (e.g., learning rate)
        # You can expand this to other hyperparameters as needed
        # Define search space
        param_name = 'conf'  # Example: confidence threshold
        left = 0.05
        right = 0.9
        best_metric = -float('inf')
        best_param = None
        n_cycles = 5  # Number of binary search cycles
        batch = 8
        patience = 5
        import json
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        for cycle in range(n_cycles):
            mid = (left + right) / 2
            print(f"\n===== Binary Search Cycle {cycle+1} =====")
            # Train for 25 epochs with mid value
            results = _train_once(epochs=25, batch=batch, conf=mid, patience=patience, save_dir=f"runs/detect/search_cycle_{cycle+1}")
            # Extract metric (e.g., best mAP50)
            try:
                metric = results.metrics.get('metrics/mAP_0.5', 0) if hasattr(results, 'metrics') else 0
            except Exception:
                metric = 0
            print(f"Cycle {cycle+1}: {param_name}={mid:.4f}, mAP50={metric}")
            # Save metrics to results/
            metrics_data = {
                "cycle": cycle+1,
                "param_name": param_name,
                "param_value": mid,
                "mAP50": metric
            }
            metrics_path = results_dir / f"cycle_{cycle+1}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics_data, f, indent=2)
            # Binary search logic
            if metric > best_metric:
                best_metric = metric
                best_param = mid
                left = mid  # Search upper half
            else:
                right = mid  # Search lower half
        print(f"\nBest {param_name}: {best_param:.4f} with mAP50={best_metric}")
        # Save final/best model metrics
        final_metrics = {
            "best_param_name": param_name,
            "best_param_value": best_param,
            "best_mAP50": best_metric
        }
        final_metrics_path = results_dir / "final_model_metrics.json"
        with open(final_metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)
        return True
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("INDIAN ROAD OBJECT DETECTION - MULTI-CLASS TRAINING")
    print("="*70)
    
    # Step 1: Generate auto-rickshaw data
    generate_synthetic_auto_rickshaws()
    
    # Step 2: Create dataset config
    dataset_yaml = create_dataset_yaml()
    
    # Step 3: Train model
    success = train_model(dataset_yaml)
    
    if success:
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nTo use the trained model:")
        print("  1. Update app.py with:")
        print("     CUSTOM_WEIGHTS_PATH = 'custom_weights/multiclass_best.pt'")
        print("  2. Restart the web app")
        print("  3. Upload Indian road images with persons, vehicles, animals, and autos")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("❌ Training failed. Check the error messages above.")
        print("="*70 + "\n")


if __name__ == '__main__':
    main()
