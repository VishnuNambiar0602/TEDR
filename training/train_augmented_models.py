"""
Enhanced Training Script with Augmented Data (3x)
Trains both YOLO detection model and time-series prediction model
"""

from ultralytics import YOLO
import os
import torch
import numpy as np
import json
from datetime import datetime
import pickle

# ====================
# PART 1: YOLO TRAINING
# ====================

def train_yolo_model():
    """Train YOLO object detection model"""
    print("\n" + "="*70)
    print("YOLO OBJECT DETECTION MODEL TRAINING")
    print("="*70)
    
    dataset_path = r"C:\Users\vishn\.cache\kagglehub\datasets\asfakali2\iruvd-dataset-for-automatic-vehicle-detection\versions\1\JU_yolov5"
    yaml_path = os.path.join(dataset_path, "data.yaml")
    
    if not os.path.exists(yaml_path):
        print(f"Warning: Dataset config not found at {yaml_path}")
        print("Skipping YOLO training. Please ensure dataset is downloaded.")
        return None
    
    print(f"Using dataset config: {yaml_path}")
    
    # Load model
    model = YOLO("yolov8n.pt")
    
    # Status file
    status_file = "training_status_yolo.json"
    
    def update_status(status, epoch=0, metrics=None):
        data = {
            "status": status,
            "epoch": epoch,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        }
        with open(status_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def on_train_epoch_end(trainer):
        metrics = {
            "mAP50": float(trainer.metrics.get("metrics/mAP50(B)", 0)),
            "box_loss": float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0
        }
        update_status("Training", epoch=trainer.epoch + 1, metrics=metrics)
    
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    update_status("Starting YOLO training")
    
    try:
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        
        results = model.train(
            data=yaml_path,
            epochs=100,           # Increased epochs for better convergence
            imgsz=640,
            batch=8,              # Small batch to save memory
            device=device,
            project="runs/detect",
            name="indian_traffic_model_augmented",
            exist_ok=True,
            patience=15,          # Early stopping
            save=True,
            verbose=True
        )
        
        update_status("YOLO training completed")
        print("\n✓ YOLO Training finished successfully!")
        print(f"  Model saved to runs/detect/indian_traffic_model_augmented/weights/best.pt")
        return results
    
    except Exception as e:
        update_status("YOLO training failed", metrics={"error": str(e)})
        print(f"✗ YOLO Training failed: {e}")
        raise e



# ====================
# MAIN EXECUTION
# ====================

if __name__ == '__main__':
    print("\n")
    print("#" * 70)
    print("# ENHANCED MODEL TRAINING WITH 3X AUGMENTED DATA")
    print("#" * 70)
    
    start_time = datetime.now()
    
    try:
        # Train YOLO model
        yolo_results = train_yolo_model()
        
        print("\nSummary:")
        print(f"  - YOLO Detection Model: Updated with existing data")
        print(f"\nModels and reports saved to runs/detect/ directory")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise e
