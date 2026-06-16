from ultralytics import YOLO
import os
import torch

# Dataset path (Hardcoded based on download location)
# Pointing directly to the inner folder where data.yaml was found
dataset_path = r"C:\Users\vishn\.cache\kagglehub\datasets\asfakali2\iruvd-dataset-for-automatic-vehicle-detection\versions\1\JU_yolov5"
yaml_path = os.path.join(dataset_path, "data.yaml")

print(f"Using dataset config: {yaml_path}")

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Callback for monitoring
import json

status_file = "training_status.json"

def update_status(status, epoch=0, metrics=None):
    data = {
        "status": status,
        "epoch": epoch,
        "metrics": metrics or {}
    }
    with open(status_file, "w") as f:
        json.dump(data, f)

def on_train_epoch_end(trainer):
    metrics = {
        "mAP50": float(trainer.metrics.get("metrics/mAP50(B)", 0)),
        "box_loss": float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0
    }
    update_status("Training", epoch=trainer.epoch + 1, metrics=metrics)

# Train the model
# Using batch=8 to save RAM/GPU memory as requested
# Using device=0 for GPU
if __name__ == '__main__':
    # check for cuda
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    update_status("Starting")

    try:
        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        results = model.train(
            data=yaml_path,
            epochs=50,        # Increased to 50 for better accuracy
            imgsz=640,
            batch=8,          # Small batch size to save memory
            device=device,
            project="runs/detect",
            name="indian_traffic_model",
            exist_ok=True
        )

        update_status("Finished")
        print("Training finished.")
        print(f"Model saved to runs/detect/indian_traffic_model/weights/best.pt")
    
    except Exception as e:
        update_status("Failed", metrics={"error": str(e)})
        raise e
