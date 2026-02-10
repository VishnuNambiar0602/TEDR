"""
Configuration settings for DETR Object Detector
"""

import torch
import os


class Config:
    """Configuration class for DETR model and application settings"""
    
    # Model Configuration
    MODEL_NAME = "facebook/detr-resnet-50"
    CONFIDENCE_THRESHOLD = 0.7
    NMS_THRESHOLD = 0.5
    
    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Image Configuration
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
    IMAGE_MAX_DIMENSION = 1333  # Max dimension for DETR input
    
    # Upload Configuration
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app', 'static', 'uploads')
    
    # COCO to Indian Road Category Mapping
    # COCO dataset has 91 classes, we map relevant ones to our categories
    COCO_CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Category to Color Mapping (for bounding boxes)
    CATEGORY_COLORS = {
        'vehicle': (59, 130, 246),      # Blue
        'pedestrian': (16, 185, 129),   # Green
        'animal': (249, 115, 22),       # Orange
        'traffic': (250, 204, 21),      # Yellow
        'other': (168, 85, 247)         # Purple
    }
    
    # Class to Category Mapping (COCO classes to Indian road categories)
    CLASS_TO_CATEGORY = {
        'person': 'pedestrian',
        'bicycle': 'vehicle',
        'car': 'vehicle',
        'motorcycle': 'vehicle',
        'bus': 'vehicle',
        'train': 'vehicle',
        'truck': 'vehicle',
        'traffic light': 'traffic',
        'stop sign': 'traffic',
        'bird': 'animal',
        'cat': 'animal',
        'dog': 'animal',
        'horse': 'animal',
        'sheep': 'animal',
        'cow': 'animal',
        'elephant': 'animal',
        'bear': 'animal',
        'zebra': 'animal',
        'giraffe': 'animal'
    }
    
    @staticmethod
    def allowed_file(filename):
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    
    @staticmethod
    def get_category_color(category):
        """Get RGB color for a category"""
        return Config.CATEGORY_COLORS.get(category, Config.CATEGORY_COLORS['other'])
    
    @staticmethod
    def get_category(class_name):
        """Get category for a class name"""
        return Config.CLASS_TO_CATEGORY.get(class_name, 'other')
