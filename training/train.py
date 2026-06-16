"""
Training Script Skeleton for Fine-tuning RT-DETR on Indian Road Dataset (IDD).

This script provides a template for training RT-DETR on custom Indian road data.

Dataset Setup:
==============
1. Download India Driving Dataset (IDD) from: 
   https://idd.is.iitd.ac.in/

2. Convert to YOLO format:
   - images/
     ├── train/
     ├── val/
     └── test/
   - labels/
     ├── train/
     ├── val/
     └── test/

   Each label file (txt) should contain:
   <class_id> <x_center> <y_center> <width> <height>  (normalized 0-1)

3. Create dataset.yaml:
   path: /path/to/dataset
   train: images/train
   val: images/val
   test: images/test
   
   nc: 13  # number of classes
   
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

Usage:
------
1. Set DATA_PATH and MODEL_NAME below
2. Run: python train.py
3. Best weights will be saved to 'runs/detect/trainX/weights/best.pt'
4. Use the .pt file in app.py by setting CUSTOM_WEIGHTS_PATH

For further fine-tuning, see:
https://docs.ultralytics.com/tasks/detect/#train
"""


import os
import sys
import yaml
from pathlib import Path
import torch

try:
    from ultralytics import YOLO, settings
except ImportError:
    print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

# Path to dataset.yaml
DATA_PATH = "custom_dataset/custom_dataset.yaml"

MODEL_NAME = "rtdetr-s.pt"  # Use the smallest model for minimal memory usage

# Fallback to YOLOv8n.pt if rtdetr-s.pt is missing
import os
if not os.path.exists(MODEL_NAME):
    print("Warning: rtdetr-s.pt not found, using yolov8n.pt as fallback.")
    MODEL_NAME = "yolov8n.pt"

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 25
BATCH_SIZE = 2
IMG_SIZE = 416  # Reduce image size for less memory usage
DEVICE = 0  # GPU device ID (0 for first GPU, or list for multiple GPUs)
PATIENCE = 20  # Early stopping patience

# Output directory
OUTPUT_DIR = "results"


# ============================================================================
# Utility Functions
# ============================================================================

def validate_dataset(dataset_path: str) -> bool:
    """
    Validate dataset structure and format.
    
    Args:
        dataset_path: Path to dataset.yaml
    
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset file not found: {dataset_path}")
        return False
    
    with open(dataset_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in dataset_config:
            print(f"✗ Missing required key in dataset.yaml: {key}")
            return False
    
    base_path = Path(dataset_config['path'])
    
    # Check directories exist
    for split in ['train', 'val']:
        image_dir = base_path / dataset_config[split]
        if not image_dir.exists():
            print(f"✗ Directory not found: {image_dir}")
            return False
    
    print(f"✓ Dataset validation passed")
    print(f"  - Path: {base_path}")
    print(f"  - Classes: {dataset_config['nc']}")
    print(f"  - Class names: {list(dataset_config['names'].values())[:5]}...")
    
    return True


def create_dataset_yaml(output_path: str, idd_root: str):
    """
    Create dataset.yaml for IDD dataset.
    
    Args:
        output_path: Where to save dataset.yaml
        idd_root: Root directory of IDD dataset
    """
    dataset_config = {
        'path': str(Path(idd_root).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 13,
        'names': {
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
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✓ Created dataset.yaml at: {output_path}")


# ============================================================================
# Training Function
# ============================================================================

def train_rtdetr(
    data_path: str,
    model_name: str = MODEL_NAME,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    img_size: int = IMG_SIZE,
    device: int = DEVICE,
    patience: int = PATIENCE,
    lr0: float = 0.005,
    mosaic: float = 1.0,
    mixup: float = 0.2,
    flipud: float = 0.5,
    fliplr: float = 0.5,
):
    """
    Train RT-DETR model on custom dataset.
    
    Args:
        data_path: Path to dataset.yaml
        model_name: Model architecture to train
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: GPU device ID
        patience: Early stopping patience
    """
    
    print("\n" + "="*70)
    print("RT-DETR Training on Indian Road Dataset")
    print("="*70)
    
    # Validate dataset
    if not validate_dataset(data_path):
        print("\n✗ Dataset validation failed. Please check your dataset setup.")
        return
    
    # Load model
    print(f"\n[1/4] Loading model: {model_name}...")
    model = YOLO(model_name)
    print(f"✓ Model loaded")
    
    # Configure training
    print(f"\n[2/4] Training configuration:")
    print(f"      Epochs: {epochs}")
    print(f"      Batch size: {batch_size}")
    print(f"      Image size: {img_size}")
    # Use GPU if available, else CPU
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"      Device: {device_str}")
    print(f"      Early stopping patience: {patience}")
    
    # Train model
    print(f"\n[3/4] Starting training...")
    print("      (This may take several hours depending on your hardware)\n")
    
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device_str,
        patience=patience,
        save=True,
        cache=False,
        workers=1,
        # Data augmentation
        mosaic=mosaic,
        mixup=mixup,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=flipud,
        fliplr=fliplr,
        # Learning rate
        lr0=lr0,
        lrf=lr0,
        close_mosaic=5,
        project=OUTPUT_DIR,
        name='exp',
        exist_ok=True
    )
    
    # Evaluate
    print(f"\n[4/4] Evaluating on validation set...")
    metrics = model.val()
    print(f"✓ Validation complete")
    
    # Print results summary
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"✓ Best weights saved to:")
    print(f"  {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")
    print(f"\n✓ To use these weights in the app, set:")
    print(f"  CUSTOM_WEIGHTS_PATH = '{Path(model.trainer.save_dir) / 'weights' / 'best.pt'}'")
    print(f"  in app.py")
    print("="*70)
    
    return results


def evaluate_model(weights_path: str, data_path: str):
    """
    Evaluate trained model on validation/test set.
    
    Args:
        weights_path: Path to trained weights (.pt file)
        data_path: Path to dataset.yaml
    """
    print(f"\nEvaluating model: {weights_path}")
    
    model = YOLO(weights_path)
    metrics = model.val(data=data_path)
    
    print(f"\n✓ Evaluation complete")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")


def predict_with_model(weights_path: str, image_path: str):
    """
    Test trained model on single image.
    
    Args:
        weights_path: Path to trained weights
        image_path: Path to test image
    """
    print(f"\nRunning inference with: {weights_path}")
    print(f"Image: {image_path}")
    
    model = YOLO(weights_path)
    results = model(image_path, conf=0.5)
    
    # Print detections
    for result in results:
        print(f"\nDetections:")
        for box in result.boxes:
            class_name = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            print(f"  - {class_name}: {confidence:.2f}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Quick start examples:
    
    1. Train from scratch:
       python train.py --mode train
    
    2. Create dataset.yaml:
       python train.py --mode create_dataset --idd_root /path/to/idd
    
    3. Evaluate trained model:
       python train.py --mode evaluate --weights runs/detect/trainX/weights/best.pt
    
    4. Run inference:
       python train.py --mode predict --weights runs/detect/trainX/weights/best.pt --image test_image.jpg
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RT-DETR Training on Indian Road Dataset"
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'evaluate', 'predict', 'create_dataset'],
        default='train',
        help='What to do'
    )
    parser.add_argument(
        '--data',
        default=DATA_PATH,
        help='Path to dataset.yaml'
    )
    parser.add_argument(
        '--model',
        default=MODEL_NAME,
        help='Model to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=BATCH_SIZE,
        help='Batch size'
    )
    parser.add_argument(
        '--weights',
        help='Path to trained weights (for eval/predict)'
    )
    parser.add_argument(
        '--image',
        help='Path to test image (for predict)'
    )
    parser.add_argument(
        '--idd_root',
        help='Root directory of IDD dataset'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Simple hyperparameter search
        best_map = 0
        best_params = None
        for batch_size in [4, 8, 16]:
            for lr0 in [0.001, 0.005, 0.01]:
                for mosaic in [0.5, 1.0]:
                    print(f"\nTrying batch_size={batch_size}, lr0={lr0}, mosaic={mosaic}")
                    results = train_rtdetr(
                        data_path=args.data,
                        model_name=args.model,
                        epochs=args.epochs,
                        batch_size=batch_size,
                        lr0=lr0,
                        mosaic=mosaic,
                    )
                    # Try to get mAP from results if possible
                    try:
                        map50 = results.metrics.box.map50 if hasattr(results, 'metrics') else 0
                    except Exception:
                        map50 = 0
                    if map50 > best_map:
                        best_map = map50
                        best_params = (batch_size, lr0, mosaic)
        print(f"Best params: batch_size={best_params[0]}, lr0={best_params[1]}, mosaic={best_params[2]}, mAP50={best_map}")
    
    elif args.mode == 'create_dataset':
        if not args.idd_root:
            print("ERROR: --idd_root required for create_dataset mode")
            sys.exit(1)
        create_dataset_yaml('dataset.yaml', args.idd_root)
    
    elif args.mode == 'evaluate':
        if not args.weights:
            print("ERROR: --weights required for evaluate mode")
            sys.exit(1)
        evaluate_model(args.weights, args.data)
    
    elif args.mode == 'predict':
        if not args.weights or not args.image:
            print("ERROR: --weights and --image required for predict mode")
            sys.exit(1)
        predict_with_model(args.weights, args.image)
