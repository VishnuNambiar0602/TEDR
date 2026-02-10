"""DETR model wrapper for object detection."""
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np


class DETRModel:
    """Wrapper class for DETR object detection model."""
    
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        confidence_threshold: float = 0.7,
        device: str = None
    ):
        """Initialize DETR model.
        
        Args:
            model_name: Name of pretrained model from HuggingFace
            confidence_threshold: Minimum confidence score for detections
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        print(f"Loading DETR model on device: {self.device}")
        
        # Load processor and model
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully: {model_name}")
    
    def detect(self, image: Image.Image) -> Dict:
        """Perform object detection on an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing detections with labels, scores, and bounding boxes
        """
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get image size
        image_width, image_height = image.size
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs
        target_sizes = torch.tensor([[image_height, image_width]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.confidence_threshold
        )[0]
        
        # Format detections
        detections = []
        for score, label, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy()
        ):
            # Convert box from [x_min, y_min, x_max, y_max] to list
            bbox = box.tolist()
            
            detection = {
                "label": self.model.config.id2label[int(label)],
                "label_id": int(label),
                "confidence": float(score),
                "bbox": bbox  # [x1, y1, x2, y2]
            }
            detections.append(detection)
        
        return {
            "detections": detections,
            "num_detections": len(detections),
            "image_size": [image_width, image_height]
        }
    
    def detect_batch(self, images: List[Image.Image]) -> List[Dict]:
        """Perform object detection on a batch of images.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of detection dictionaries
        """
        results = []
        for image in images:
            result = self.detect(image)
            results.append(result)
        return results
    
    def save_model(self, path: str):
        """Save model to disk.
        
        Args:
            path: Directory path to save model
        """
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk.
        
        Args:
            path: Directory path to load model from
        """
        self.model = DetrForObjectDetection.from_pretrained(path)
        self.processor = DetrImageProcessor.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {path}")
