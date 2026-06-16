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
# PART 2: TRAFFIC PREDICTION MODEL TRAINING
# ====================

def train_traffic_prediction_model():
    """Train time-series traffic prediction model with augmented data"""
    print("\n" + "="*70)
    print("TRAFFIC PREDICTION MODEL TRAINING (3x Augmented Data)")
    print("="*70)
    
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "scikit-learn", "-q"], check=True)
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
    
    processed_dir = "test_data/processed"
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Load augmented data
    print("\nLoading augmented datasets...")
    X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
    y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
    
    X_val = np.load(os.path.join(processed_dir, "X_val.npy"))
    y_val = np.load(os.path.join(processed_dir, "y_val.npy"))
    
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    
    print(f"  Training samples:   {X_train.shape[0]} (3x augmented)")
    print(f"  Validation samples: {X_val.shape[0]} (3x augmented)")
    print(f"  Test samples:       {X_test.shape[0]} (3x augmented)")
    
    # Load metadata
    with open(os.path.join(processed_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Standardize data
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting Regressor
    print("\nTraining Gradient Boosting Regressor...")
    print("  Parameters:")
    print("    - n_estimators: 200")
    print("    - max_depth: 7")
    print("    - learning_rate: 0.05")
    print("    - subsample: 0.8")
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=1,
        n_iter_no_change=20,
        validation_fraction=0.1
    )
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\nEvaluating on validation set...")
    val_score = model.score(X_val_scaled, y_val)
    print(f"  Validation R² Score: {val_score:.4f}")
    
    print("\nEvaluating on test set...")
    test_score = model.score(X_test_scaled, y_test)
    print(f"  Test R² Score: {test_score:.4f}")
    
    # Predictions for analysis
    val_pred = model.predict(X_val_scaled)
    val_mse = np.mean((y_val - val_pred) ** 2)
    val_rmse = np.sqrt(val_mse)
    val_mae = np.mean(np.abs(y_val - val_pred))
    
    test_pred = model.predict(X_test_scaled)
    test_mse = np.mean((y_test - test_pred) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(y_test - test_pred))
    
    print(f"\n  Validation Metrics:")
    print(f"    - RMSE: {val_rmse:.4f}")
    print(f"    - MAE:  {val_mae:.4f}")
    print(f"\n  Test Metrics:")
    print(f"    - RMSE: {test_rmse:.4f}")
    print(f"    - MAE:  {test_mae:.4f}")
    
    # Save models and scalers
    print("\nSaving models and scalers...")
    model_path = os.path.join(models_dir, "traffic_predictor_augmented.pkl")
    scaler_path = os.path.join(models_dir, "scaler_augmented.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    # Save training report
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_augmentation": True,
        "augmentation_factor": 3,
        "training_samples": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "model_type": "GradientBoostingRegressor",
        "model_parameters": {
            "n_estimators": 200,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8
        },
        "validation_metrics": {
            "r2_score": float(val_score),
            "rmse": float(val_rmse),
            "mae": float(val_mae)
        },
        "test_metrics": {
            "r2_score": float(test_score),
            "rmse": float(test_rmse),
            "mae": float(test_mae)
        },
        "feature_columns": metadata.get("feature_columns", []),
        "target_column": metadata.get("target_column", "traffic_volume")
    }
    
    report_path = os.path.join(models_dir, "training_report_augmented.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Traffic Prediction Model Training Complete!")
    print(f"  Model saved to:    {model_path}")
    print(f"  Scaler saved to:   {scaler_path}")
    print(f"  Report saved to:   {report_path}")
    
    return model, scaler, report


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
        
        # Train Traffic Prediction model
        model, scaler, report = train_traffic_prediction_model()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "="*70)
        print("✓ ALL TRAINING COMPLETE!")
        print("="*70)
        print(f"Total training time: {duration:.2f} minutes")
        print("\nSummary:")
        print(f"  - YOLO Detection Model: Updated with existing data")
        print(f"  - Traffic Prediction Model: Trained with 3x augmented data")
        print(f"    • Training:   1251 samples (3x)")
        print(f"    • Validation: 417 samples (3x)")
        print(f"    • Testing:    420 samples (3x)")
        print(f"\nModels and reports saved to runs/detect/ and models/ directories")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise e
