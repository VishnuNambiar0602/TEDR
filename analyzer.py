from ultralytics import YOLO
import cv2
import math
import os

class TrafficAnalyzer:
    def __init__(self, model_path="rtdetr-l.pt", cnn_fallback_path="yolov8n.pt"):
        # We prefer using RT-DETR (Transformer) for vehicle/object detection.
        # If it fails to load or isn't found, we fall back to the custom CNN model (YOLO)
        # or the standard CNN model as backup.
        self.model = None
        self.vehicle_classes = set()
        
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"TrafficAnalyzer initialized. Device: {self.device}")
        
        # Try to load RT-DETR Transformer first
        try:
            print(f"Loading RT-DETR Transformer model from {model_path}...")
            self.model = YOLO(model_path)
            if self.device == "cuda":
                self.model.to(self.device)
            # RT-DETR COCO classes for vehicles and relevant road objects
            self.vehicle_classes = {
                'car', 'bus', 'truck', 'motorcycle', 'motorbike', 'bicycle',
                'person', 'cow', 'dog'
            }
            print("✓ Loaded RT-DETR Transformer model successfully")
        except Exception as e:
            print(f"Warning: Failed to load RT-DETR Transformer model: {e}")
            print("Attempting to load custom CNN fallback model...")
            
            custom_model_path = "runs/detect/indian_traffic_model/weights/best.pt"
            if os.path.exists(custom_model_path):
                print(f"Loading custom Indian Traffic CNN model from {custom_model_path}")
                try:
                    self.model = YOLO(custom_model_path)
                    self.vehicle_classes = {
                        'trak', 'cyclist', 'bike', 'tempo', 'car', 'zeep', 'toto', 
                        'e-rickshaw', 'auto-rickshaw', 'bus', 'van', 'cycle-rickshaw', 
                        'taxi', 'motorcycle', 'truck'
                    }
                    print("✓ Loaded custom CNN model successfully")
                except Exception as ex:
                    print(f"Failed to load custom CNN model: {ex}")
                    self.model = None
            
            if self.model is None:
                print(f"Loading standard CNN fallback model from {cnn_fallback_path}")
                try:
                    self.model = YOLO(cnn_fallback_path)
                    self.vehicle_classes = {'car', 'bus', 'truck', 'motorbike', 'motorcycle', 'bicycle'}
                    print("✓ Loaded standard CNN fallback model successfully")
                except Exception as ex:
                    print(f"Error loading fallback CNN model: {ex}")
                    raise RuntimeError("No object detection model could be loaded.")
            
        self.low_congestion = 0.15
        self.high_congestion = 0.35

    def analyze_frame(self, frame):
        if frame is None:
            raise ValueError("Input frame is None")

        img = frame.copy()
        height, width, _ = img.shape
        image_area = height * width

        results = self.model(img, conf=0.4, device=self.device)

        vehicle_count = 0
        vehicle_area = 0
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]

                if class_name not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                vehicle_area += area
                vehicle_count += 1

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, class_name, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                detections.append({
                    "class": class_name,
                    "box": [x1, y1, x2, y2]
                })

        occupancy_ratio = vehicle_area / image_area if image_area > 0 else 0

        if occupancy_ratio < self.low_congestion:
            congestion_level = "LOW"
        elif occupancy_ratio < self.high_congestion:
            congestion_level = "MEDIUM"
        else:
            congestion_level = "HIGH"

        label = f"Vehicles: {vehicle_count} | Congestion: {congestion_level}"
        cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return {
            "vehicle_count": vehicle_count,
            "occupancy_ratio": round(occupancy_ratio, 3),
            "congestion_level": congestion_level,
            "processed_image": img,
            "detections": detections
        }

    def analyze(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        result = self.analyze_frame(img)
        return {
            "vehicle_count": result["vehicle_count"],
            "occupancy_ratio": result["occupancy_ratio"],
            "congestion_level": result["congestion_level"],
            "processed_image": result["processed_image"]
        }
