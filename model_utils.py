"""
Model Utilities for Indian Road Object Detection using RT-DETR/DETR.

This module handles:
- Loading pre-trained RT-DETR or DETR models from Ultralytics or Hugging Face
- Processing images and performing inference
- Post-processing detection outputs
- Handling custom-trained weights

Why Transformers for Indian Roads:
====================================
1. Global Context Awareness: Unlike CNNs with local receptive fields, Transformers use 
   self-attention to simultaneously capture relationships between all objects in the image. 
   This is crucial for crowded Indian roads with multiple interacting vehicles, animals, 
   and pedestrians.

2. Handling Unstructured Scenes: Indian roads often have non-standard arrangements (cows 
   on highways, informal parking). Transformers excel at understanding these chaotic 
   patterns without excessive hand-crafted features.

3. Scale Invariance: Objects appear at various scales in Indian roads (distant trucks, 
   close auto-rickshaws). Transformers' attention mechanism naturally handles this better 
   than fixed anchor-based approaches.

4. Dense Object Detection: Transformers avoid the "one anchor per object" limitation, 
   enabling better detection in crowded scenes.

Target Detection Classes (Indian-Specific):
===========================================
Vehicles: auto_rickshaw, truck, bus, motorcycle, car, bicycle, bull_cart
Animals: cow, dog, goat
Pedestrians: pedestrian (including jaywalkers, traditional attire)
Others: traffic_sign, pole, building (context)
"""

import os
import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# Try to import ultralytics RT-DETR for real-time performance
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Try to import Hugging Face Transformers for advanced models
try:
    from transformers import AutoImageProcessor, RTDetrForObjectDetection
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


# Class mapping for Indian roads
INDIAN_ROAD_CLASSES = {
    # Vehicles
    'auto_rickshaw': (255, 100, 0),      # Orange
    'truck': (0, 0, 255),                 # Red
    'bus': (0, 165, 255),                 # Orange-red
    'motorcycle': (200, 0, 200),          # Purple
    'car': (0, 255, 0),                   # Green
    'bicycle': (255, 255, 0),             # Cyan
    'bull_cart': (165, 42, 42),           # Brown
    
    # Animals
    'cow': (255, 0, 255),                 # Magenta
    'dog': (128, 0, 128),                 # Dark Purple
    'goat': (255, 192, 203),              # Pink
    
    # Pedestrians
    'pedestrian': (255, 0, 0),            # Blue
    'jaywalker': (0, 255, 255),           # Yellow (for tracking)
    
    # Context
    'traffic_sign': (0, 128, 255),        # Dark Orange
    'pole': (128, 128, 128),              # Gray
}


class IndianRoadDetector:
    """
    RT-DETR/DETR based object detector for Indian road scenarios.
    
    Supports:
    - Pre-trained models from Ultralytics or Hugging Face
    - Custom fine-tuned weights
    - Batch processing
    - Confidence threshold adjustment
    """
    
    def __init__(
        self,
        model_name: str = 'rtdetr-l',
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the detector.
        
        Args:
            model_name: Model identifier
                - Ultralytics: 'rtdetr-s', 'rtdetr-m', 'rtdetr-l', 'rtdetr-x'
                - Hugging Face: 'PekingU/rtdetr_r50vd' or custom model ID
            weights_path: Path to custom .pt or .safetensors weights
            device: 'cuda' or 'cpu' (auto-detected if None)
            confidence_threshold: Detection confidence threshold (0-1)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None
        self.model_type = None  # 'ultralytics' or 'huggingface'
        
        self._load_model(model_name, weights_path)
    
    def _load_model(self, model_name: str, weights_path: Optional[str] = None):
        """Load model from Ultralytics or Hugging Face."""
        
        if weights_path and os.path.exists(weights_path):
            print(f"Loading custom weights from: {weights_path}")
            # Try Ultralytics first (custom .pt files)
            if weights_path.endswith('.pt') and ULTRALYTICS_AVAILABLE:
                self.model = YOLO(weights_path)
                self.model_type = 'ultralytics'
                print(f"✓ Loaded custom Ultralytics model from {weights_path}")
                return
        
        # Load from Ultralytics official models
        if ULTRALYTICS_AVAILABLE and model_name.startswith('rtdetr'):
            try:
                self.model = YOLO(model_name)
                self.model_type = 'ultralytics'
                print(f"✓ Loaded Ultralytics {model_name} model")
                return
            except Exception as e:
                print(f"Warning: Could not load {model_name} from Ultralytics: {e}")
        
        # Load from Hugging Face Transformers
        if HUGGINGFACE_AVAILABLE:
            try:
                model_id = model_name if '/' in model_name else 'PekingU/rtdetr_r50vd'
                self.processor = AutoImageProcessor.from_pretrained(model_id)
                self.model = RTDetrForObjectDetection.from_pretrained(model_id)
                self.model = self.model.to(self.device)
                self.model.eval()
                self.model_type = 'huggingface'
                print(f"✓ Loaded Hugging Face model: {model_id}")
                return
            except Exception as e:
                print(f"Warning: Could not load from Hugging Face: {e}")
        
        raise RuntimeError(
            "No model loaded! Please install either:\n"
            "  1. ultralytics: pip install ultralytics\n"
            "  2. transformers: pip install transformers\n"
        )
    
    def predict(
        self,
        image_path: str,
        confidence: Optional[float] = None,
    ) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image_path: Path to input image
            confidence: Override default confidence threshold
        
        Returns:
            Dictionary with detection results
        """
        conf = confidence or self.confidence_threshold
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        return self.predict_array(image, confidence=conf)
    
    def predict_array(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
    ) -> Dict:
        """
        Run inference on a numpy array image.
        
        Args:
            image: Image as numpy array (BGR format from OpenCV)
            confidence: Override default confidence threshold
        
        Returns:
            Dictionary with detection results
        """
        conf = confidence or self.confidence_threshold
        
        if self.model_type == 'ultralytics':
            return self._predict_ultralytics(image, conf)
        else:
            return self._predict_huggingface(image, conf)
    
    def _predict_ultralytics(self, image: np.ndarray, conf: float) -> Dict:
        """Ultralytics RT-DETR inference."""
        
        # Run inference
        results = self.model(image, conf=conf, device=self.device, verbose=False)
        
        detections = []
        class_counts = {}
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class_id': class_id,
                })
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'detections': detections,
            'class_counts': class_counts,
            'image_shape': image.shape,
            'model_type': 'ultralytics',
        }
    
    def _predict_huggingface(self, image: np.ndarray, conf: float) -> Dict:
        """Hugging Face RT-DETR inference."""
        
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=conf
        )
        
        detections = []
        class_counts = {}
        
        for result in results:
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]
            
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.cpu().numpy()
                class_name = self.model.config.id2label[int(label.cpu().numpy())]
                
                detections.append({
                    'class': class_name,
                    'confidence': float(score.cpu().numpy()),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class_id': int(label.cpu().numpy()),
                })
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'detections': detections,
            'class_counts': class_counts,
            'image_shape': image.shape,
            'model_type': 'huggingface',
        }
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image (BGR format)
            detections: List of detection dictionaries
            thickness: Box line thickness
        
        Returns:
            Image with drawn boxes
        """
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # Get color for class
            color = INDIAN_ROAD_CLASSES.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness
            )
            
            # Draw label with confidence
            label = f"{class_name} ({confidence:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_x = int(x1)
            text_y = int(y1) - 5
            
            # Background for text
            cv2.rectangle(
                annotated,
                (text_x, text_y - text_size[1] - 5),
                (text_x + text_size[0], text_y + 5),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        return annotated
    
    def set_confidence(self, confidence: float):
        """Update confidence threshold."""
        if 0 <= confidence <= 1:
            self.confidence_threshold = confidence
        else:
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")


# Convenience function for loading models
def load_detector(
    model_name: str = 'rtdetr-l',
    weights_path: Optional[str] = None,
    confidence: float = 0.5,
) -> IndianRoadDetector:
    """
    Quick loader for the detector.
    
    Args:
        model_name: Model identifier
        weights_path: Path to custom weights
        confidence: Confidence threshold
    
    Returns:
        IndianRoadDetector instance
    """
    return IndianRoadDetector(
        model_name=model_name,
        weights_path=weights_path,
        confidence_threshold=confidence,
    )
