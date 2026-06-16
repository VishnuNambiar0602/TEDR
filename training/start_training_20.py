import os
import torch
from ultralytics import YOLO

# Configuration
DATA_PATH = "custom_dataset/custom_dataset.yaml"
# Using rtdetr-s.pt which will be downloaded by ultralytics if not present
MODEL_NAME = "rtdetr-s.pt"
EPOCHS = 20
BATCH_SIZE = 4 # Reduced batch for GPU memory
IMG_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

def start_training():
    print(f"Starting training on {DEVICE} for {EPOCHS} epochs...")
    print(f"Dataset config: {DATA_PATH}")
    print(f"Model architecture: {MODEL_NAME}")
    
    # Check if dataset config exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset config not found at {DATA_PATH}")
        return

    # Load model
    model = YOLO(MODEL_NAME)
    
    # Train
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        save=True,
        project="runs/detect",
        name="train_20_epochs",
        exist_ok=True,
        verbose=True
    )
    
    print("Training finished.")
    print(f"Best weights saved to runs/detect/train_20_epochs/weights/best.pt")

if __name__ == "__main__":
    start_training()
