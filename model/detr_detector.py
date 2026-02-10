"""
DETR Object Detector Wrapper
Uses Hugging Face Transformers implementation of DETR
"""

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import numpy as np
from .config import Config
from .utils import process_detections, draw_boxes, get_detection_statistics


class DETRDetector:
    """
    DETR (DEtection TRansformer) wrapper for object detection
    """
    
    def __init__(self, model_name=None, confidence_threshold=None, device=None):
        """
        Initialize DETR detector
        
        Args:
            model_name: Hugging Face model name (default: from config)
            confidence_threshold: Minimum confidence for detections (default: from config)
            device: Device to run model on (default: auto-detect)
        """
        self.model_name = model_name or Config.MODEL_NAME
        self.confidence_threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD
        self.device = device or Config.DEVICE
        
        self.processor = None
        self.model = None
        self._model_loaded = False
    
    def load_model(self):
        """Load the DETR model and processor (lazy loading)"""
        if self._model_loaded:
            return
        
        print(f"Loading DETR model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = DetrImageProcessor.from_pretrained(self.model_name)
        self.model = DetrForObjectDetection.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self._model_loaded = True
        print("Model loaded successfully!")
    
    def preprocess_image(self, image):
        """
        Preprocess image for DETR model
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Preprocessed inputs
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Use processor to prepare inputs
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs, image
    
    def postprocess_outputs(self, outputs, target_sizes, original_image):
        """
        Post-process model outputs to get detections
        
        Args:
            outputs: Model outputs
            target_sizes: Target image sizes
            original_image: Original PIL image
        
        Returns:
            List of detection dictionaries
        """
        # Use processor to convert outputs to COCO format
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes,
            threshold=0.0  # We'll filter later
        )[0]
        
        detections = []
        
        # Extract boxes, scores, and labels
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        labels = results['labels'].cpu().numpy()
        
        for box, score, label_id in zip(boxes, scores, labels):
            # Get class name
            label_name = Config.COCO_CLASSES[label_id] if label_id < len(Config.COCO_CLASSES) else 'unknown'
            
            # Skip N/A classes
            if label_name == 'N/A':
                continue
            
            # Get category
            category = Config.get_category(label_name)
            
            detections.append({
                'box': box.tolist(),
                'score': float(score),
                'label': label_name,
                'label_id': int(label_id),
                'category': category
            })
        
        return detections
    
    def detect(self, image_input):
        """
        Perform object detection on an image
        
        Args:
            image_input: PIL Image, numpy array, or file path
        
        Returns:
            Dictionary with 'detections', 'annotated_image', and 'statistics'
        """
        # Load model if not already loaded
        self.load_model()
        
        # Load image if path provided
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
        
        # Preprocess
        inputs, original_image = self.preprocess_image(image)
        
        # Get image size for post-processing
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs
        raw_detections = self.postprocess_outputs(outputs, target_sizes, original_image)
        
        # Filter and apply NMS
        detections = process_detections(
            raw_detections,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=Config.NMS_THRESHOLD
        )
        
        # Draw boxes on image
        annotated_image = draw_boxes(original_image, detections)
        
        # Get statistics
        statistics = get_detection_statistics(detections)
        
        return {
            'detections': detections,
            'annotated_image': annotated_image,
            'statistics': statistics
        }
    
    def detect_batch(self, images):
        """
        Perform object detection on multiple images
        
        Args:
            images: List of PIL Images or file paths
        
        Returns:
            List of detection result dictionaries
        """
        results = []
        for image in images:
            results.append(self.detect(image))
        return results
