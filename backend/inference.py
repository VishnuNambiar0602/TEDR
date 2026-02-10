"""Inference logic for TEDR object detection."""
import time
from PIL import Image
from typing import Dict, List
import io
from models.detr_model import DETRModel
from backend.config import config


class ObjectDetector:
    """Main object detector class."""
    
    def __init__(self):
        """Initialize object detector with DETR model."""
        self.model = DETRModel(
            model_name=config.model_name,
            confidence_threshold=config.confidence_threshold,
            device=config.device
        )
    
    def process_image(self, image_bytes: bytes) -> Dict:
        """Process image and return detection results.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary containing detections and metadata
        """
        start_time = time.time()
        
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Run detection
        results = self.model.detect(image)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add processing time to results
        results['processing_time'] = round(processing_time, 3)
        
        return results
    
    def process_image_file(self, image_path: str) -> Dict:
        """Process image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing detections and metadata
        """
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        return self.process_image(image_bytes)


# Global detector instance (lazy loaded)
_detector = None


def get_detector() -> ObjectDetector:
    """Get or create global detector instance.
    
    Returns:
        ObjectDetector instance
    """
    global _detector
    if _detector is None:
        _detector = ObjectDetector()
    return _detector
