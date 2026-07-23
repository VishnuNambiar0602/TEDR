import sys
import os
import math
import cv2
import numpy as np

# ==============================================================================
# Python 3.14 Protobuf compatibility patch
# Intercepts imports of binary-extension modules which crash on Python 3.14
# ==============================================================================
class ProtobufImportPreventer:
    def find_spec(self, fullname, path, target=None):
        if fullname in ('google._upb._message', 'google.protobuf.pyext._message'):
            raise ImportError(f"Mocked import error for {fullname} on Python 3.14")
        return None

sys.meta_path.insert(0, ProtobufImportPreventer())

# ==============================================================================
# TensorRT 10.0+ / 11.0+ compatibility monkey-patch for Ultralytics
# Mocks deprecated flags/properties removed in TensorRT 10/11
# ==============================================================================
try:
    import tensorrt as trt
    # Mock deprecated NetworkDefinitionCreationFlag.EXPLICIT_BATCH
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 0
    # Mock deprecated platform_has_fast_fp16 and platform_has_fast_int8
    trt.Builder.platform_has_fast_fp16 = property(lambda self: True)
    trt.Builder.platform_has_fast_int8 = property(lambda self: True)
    # Mock deprecated BuilderFlag.FP16 and BuilderFlag.INT8
    if not hasattr(trt.BuilderFlag, 'FP16'):
        trt.BuilderFlag.FP16 = 'DUMMY_FP16'
    if not hasattr(trt.BuilderFlag, 'INT8'):
        trt.BuilderFlag.INT8 = 'DUMMY_INT8'
    # Wrap config.set_flag to ignore FP16/INT8 flags
    orig_set_flag = trt.IBuilderConfig.set_flag
    def custom_set_flag(self, flag):
        if flag in ('DUMMY_FP16', 'DUMMY_INT8'):
            return
        orig_set_flag(self, flag)
    trt.IBuilderConfig.set_flag = custom_set_flag
    print("[SUCCESS] Successfully applied TensorRT 10/11 compatibility patches")
except ImportError:
    pass

from ultralytics import YOLO, RTDETR

# ==============================================================================
# Tracker Helper Functions and Classes (Adaptive Temporal Tracking)
# ==============================================================================
def get_iou(box1, box2):
    """Calculates Intersection over Union (IoU) of two bounding boxes."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_area = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

class Track:
    """Represents a tracked vehicle bounding box with a Constant Velocity motion model."""
    def __init__(self, track_id, box, class_name, conf):
        self.id = track_id
        self.box = np.array(box, dtype=float)  # [x1, y1, x2, y2]
        self.class_name = class_name
        self.conf = conf
        self.velocity = np.zeros(2, dtype=float)  # [vx, vy]
        self.last_box = np.array(box, dtype=float)
        self.age = 0
        self.time_since_update = 0

    def predict(self, width, height):
        """Predicts position on intermediate frames using constant velocity."""
        self.last_box = self.box.copy()
        self.box[0] += self.velocity[0]
        self.box[2] += self.velocity[0]
        self.box[1] += self.velocity[1]
        self.box[3] += self.velocity[1]
        
        # Keep box coordinates within frame boundaries
        self.box[0] = max(0.0, min(self.box[0], width - 1))
        self.box[2] = max(0.0, min(self.box[2], width - 1))
        self.box[1] = max(0.0, min(self.box[1], height - 1))
        self.box[3] = max(0.0, min(self.box[3], height - 1))
        
        self.age += 1
        return self.box

    def update(self, new_box, conf):
        """Updates track state with a new detector measurement and updates velocity."""
        new_box = np.array(new_box, dtype=float)
        old_center = np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])
        new_center = np.array([(new_box[0] + new_box[2]) / 2, (new_box[1] + new_box[3]) / 2])
        
        # Calculate instantaneous velocity
        inst_velocity = new_center - old_center
        # Clip max velocity jump to prevent erratic bounding boxes
        max_vel = 40.0
        inst_velocity = np.clip(inst_velocity, -max_vel, max_vel)
        
        # Exponential moving average for velocity smoothing
        alpha = 0.5
        if np.all(self.velocity == 0):
            self.velocity = inst_velocity
        else:
            self.velocity = alpha * inst_velocity + (1 - alpha) * self.velocity
            
        self.box = new_box
        self.conf = conf
        self.time_since_update = 0

class ObjectTracker:
    """Manages active tracks and performs associations with new detections using Greedy IoU."""
    def __init__(self, max_age=5, min_iou=0.25):
        self.max_age = max_age
        self.min_iou = min_iou
        self.tracks = []
        self.next_id = 1

    def update(self, detections, width, height):
        # Predict positions
        for track in self.tracks:
            track.time_since_update += 1
            track.predict(width, height)
            
        matched_tracks = set()
        matched_detections = set()
        associations = []
        
        # Greedy matching based on same class and IoU
        for t_idx, track in enumerate(self.tracks):
            best_iou = -1
            best_d_idx = -1
            for d_idx, det in enumerate(detections):
                if d_idx in matched_detections:
                    continue
                if track.class_name != det["class"]:
                    continue
                iou = get_iou(track.box, det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = d_idx
            if best_iou >= self.min_iou:
                associations.append((t_idx, best_d_idx, best_iou))
                matched_detections.add(best_d_idx)
                
        # Update matched tracks
        for t_idx, d_idx, iou in associations:
            det = detections[d_idx]
            self.tracks[t_idx].update(det["box"], det["conf"])
            matched_tracks.add(t_idx)
            
        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_track = Track(self.next_id, det["box"], det["class"], det["conf"])
                self.next_id += 1
                self.tracks.append(new_track)
                
        # Filter dead tracks (not seen for max_age updates)
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        # Filter out tracks that have degenerated sizes
        valid_tracks = []
        for t in self.tracks:
            w = t.box[2] - t.box[0]
            h = t.box[3] - t.box[1]
            if w > 5 and h > 5:
                valid_tracks.append(t)
        self.tracks = valid_tracks

    def step_only(self, width, height):
        """Updates tracks on intermediate skipped frames without detector measurements."""
        for track in self.tracks:
            track.predict(width, height)


# ==============================================================================
# TrafficAnalyzer with Adaptive Temporal Detection
# ==============================================================================
class TrafficAnalyzer:
    def __init__(self, model_path="rtdetr-l.engine", cnn_fallback_path="yolov8n.pt"):
        self.model = None
        self.vehicle_classes = set()
        
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"TrafficAnalyzer initialized. Device: {self.device}")
        
        # Resolve model paths (prefer TensorRT engine only if CUDA is available)
        if model_path in ("rtdetr-l.engine", "rtdetr-l.pt"):
            if self.device == "cuda" and os.path.exists("rtdetr-l.engine"):
                model_path = "rtdetr-l.engine"
            elif os.path.exists("rtdetr-l.pt"):
                model_path = "rtdetr-l.pt"
        elif model_path.endswith(".engine") and self.device != "cuda":
            pt_fallback = model_path.replace(".engine", ".pt")
            if os.path.exists(pt_fallback):
                print(f"CUDA not available. Falling back from TensorRT engine {model_path} to PyTorch weights {pt_fallback}")
                model_path = pt_fallback

        # Try loading model
        try:
            print(f"Loading object detection model from {model_path}...")
            if "rtdetr" in model_path.lower():
                self.model = RTDETR(model_path)
            else:
                self.model = YOLO(model_path)
            # RT-DETR COCO classes for vehicles and relevant road objects
            self.vehicle_classes = {
                'car', 'bus', 'truck', 'motorcycle', 'motorbike', 'bicycle',
                'person', 'cow', 'dog'
            }
            print(f"[SUCCESS] Loaded model successfully: {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load preferred model: {e}")
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
                    print("[SUCCESS] Loaded custom CNN model successfully")
                except Exception as ex:
                    print(f"Failed to load custom CNN model: {ex}")
                    self.model = None
            
            if self.model is None:
                print(f"Loading standard CNN fallback model from {cnn_fallback_path}")
                try:
                    self.model = YOLO(cnn_fallback_path)
                    self.vehicle_classes = {'car', 'bus', 'truck', 'motorbike', 'motorcycle', 'bicycle'}
                    print("[SUCCESS] Loaded standard CNN fallback model successfully")
                except Exception as ex:
                    print(f"Error loading fallback CNN model: {ex}")
                    raise RuntimeError("No object detection model could be loaded.")
            
        # Congestion ratios
        self.low_congestion = 0.15
        self.high_congestion = 0.35
        
        # Adaptive Temporal Tracking variables
        self.frame_skip = 5
        self.adaptive = True
        self.frame_counter = 0
        self.tracker = ObjectTracker()

    def reset(self):
        """Resets tracker state and counter for a new video analysis session."""
        self.frame_counter = 0
        self.tracker = ObjectTracker()
        self.frame_skip = 5

    def adjust_frame_skip(self, active_tracks):
        """Dynamically adjusts detector invocation interval (frame_skip) based on scene speeds."""
        if not active_tracks:
            # Low dynamics / empty screen: check less frequently to save resources
            self.frame_skip = 20
            return
            
        speeds = []
        for t in active_tracks:
            speed = math.sqrt(t.velocity[0]**2 + t.velocity[1]**2)
            speeds.append(speed)
            
        avg_speed = sum(speeds) / len(speeds)
        
        # Adaptive thresholds
        if avg_speed > 18.0:
            # Fast movement: run detection more frequently (every 4th frame)
            self.frame_skip = 4
        elif avg_speed > 8.0:
            # Moderate movement: standard skipping (every 8th frame)
            self.frame_skip = 8
        else:
            # Static / Gridlock/ Slow movement: skip more frames (every 12th frame)
            self.frame_skip = 12

    def analyze_frame(self, frame, frame_skip=None, adaptive=None):
        if frame is None:
            raise ValueError("Input frame is None")
            
        if frame_skip is None:
            frame_skip = self.frame_skip
        if adaptive is None:
            adaptive = self.adaptive
            
        img = frame
        height, width, _ = img.shape
        image_area = height * width
        
        # Determine if this frame runs detection or temporal tracking projection
        is_detection_frame = (self.frame_counter % frame_skip == 0)
        
        detections = []
        
        if is_detection_frame:
            # Run deep learning detection (TensorRT engine or fallback)
            if self.device == "cuda":
                import torch
                import torch.nn.functional as F_torch
                # Transfer BGR frame to GPU (faster as uint8)
                tensor = torch.from_numpy(img).to(self.device)
                # Convert BGR to RGB, permute HWC -> CHW, convert to float, normalize, and unsqueeze to BCHW
                tensor = tensor[:, :, [2, 1, 0]].permute(2, 0, 1).float().div(255.0).unsqueeze(0)
                # GPU-based resize to 640x640
                tensor = F_torch.interpolate(tensor, size=(640, 640), mode='bilinear', align_corners=False)
                results = self.model(tensor, conf=0.4, verbose=False)
            else:
                results = self.model(img, conf=0.4, device=self.device, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    if class_name not in self.vehicle_classes:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class": class_name,
                        "conf": conf
                    })
            # Update tracker with detections
            self.tracker.update(detections, width, height)
        else:
            # Skip detection, update positions using motion models
            self.tracker.step_only(width, height)
            
        # Draw bounding boxes from active tracks
        active_tracks = self.tracker.tracks
        vehicle_count = 0
        vehicle_area = 0
        result_detections = []
        
        # Visual color palette for rich aesthetics
        class_colors = {
            'car': (0, 255, 0),        # Vibrant Green
            'truck': (0, 165, 255),    # Safety Orange
            'bus': (255, 0, 0),        # Deep Blue
            'motorcycle': (0, 255, 255),# Yellow
            'motorbike': (0, 255, 255),
            'bicycle': (255, 0, 255),  # Magenta
            'person': (255, 255, 0),   # Cyan
        }
        default_color = (0, 255, 0)
        
        for t in active_tracks:
            x1, y1, x2, y2 = map(int, t.box)
            class_name = t.class_name
            track_id = t.id
            
            area = (x2 - x1) * (y2 - y1)
            vehicle_area += area
            vehicle_count += 1
            
            color = class_colors.get(class_name, default_color)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Only draw label text if the box is large enough to save rendering overhead
            if area > 1200:
                label = f"{class_name} #{track_id}"
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                        
            result_detections.append({
                "class": class_name,
                "box": [x1, y1, x2, y2],
                "id": track_id
            })
            
        occupancy_ratio = vehicle_area / image_area if image_area > 0 else 0
        
        if occupancy_ratio < self.low_congestion:
            congestion_level = "LOW"
        elif occupancy_ratio < self.high_congestion:
            congestion_level = "MEDIUM"
        else:
            congestion_level = "HIGH"
            
        # Draw dynamic transparent HUD Overlay for stats (optimized blending on ROI)
        x_min, y_min, x_max, y_max = 10, 10, 340, 95
        hud_roi = img[y_min:y_max, x_min:x_max]
        hud_bg = np.zeros_like(hud_roi) # Dark overlay
        cv2.addWeighted(hud_roi, 0.4, hud_bg, 0.6, 0, img[y_min:y_max, x_min:x_max])
        
        cv2.putText(img, f"AT-AIR-DETR Active Tracks: {vehicle_count}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Congestion Level: {congestion_level}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if congestion_level != "HIGH" else (0, 0, 255), 2)
        cv2.putText(img, f"Mode: {'DETECTION' if is_detection_frame else 'TRACKING'} (Skip: {frame_skip})", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if is_detection_frame else (255, 165, 0), 1)
                    
        # Apply Adaptive Frame Skipping adjustment
        if adaptive:
            self.adjust_frame_skip(active_tracks)
            
        self.frame_counter += 1
        
        return {
            "vehicle_count": vehicle_count,
            "occupancy_ratio": round(occupancy_ratio, 3),
            "congestion_level": congestion_level,
            "processed_image": img,
            "detections": result_detections
        }

    def analyze(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        # Single image detection bypasses tracking and sets skip to 1
        self.reset()
        result = self.analyze_frame(img, frame_skip=1, adaptive=False)
        return {
            "vehicle_count": result["vehicle_count"],
            "occupancy_ratio": result["occupancy_ratio"],
            "congestion_level": result["congestion_level"],
            "processed_image": result["processed_image"]
        }
