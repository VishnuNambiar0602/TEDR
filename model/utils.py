"""
Utility functions for object detection and visualization
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from .config import Config


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU score (float)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def apply_nms(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping boxes
    
    Args:
        detections: List of detection dictionaries with 'box', 'score', 'label'
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Filtered list of detections
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence score (descending)
    detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    keep = []
    
    while len(detections) > 0:
        # Keep the detection with highest confidence
        best = detections[0]
        keep.append(best)
        detections = detections[1:]
        
        # Remove detections with high IoU with the best detection
        detections = [
            det for det in detections 
            if calculate_iou(best['box'], det['box']) < iou_threshold
        ]
    
    return keep


def draw_boxes(image, detections, show_confidence=True):
    """
    Draw bounding boxes on image with labels and confidence scores
    
    Args:
        image: PIL Image
        detections: List of detection dictionaries
        show_confidence: Whether to show confidence scores
    
    Returns:
        PIL Image with drawn boxes
    """
    # Convert PIL to OpenCV format for better drawing
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for det in detections:
        box = det['box']
        label = det['label']
        score = det['score']
        category = det['category']
        
        # Get color for category (BGR format for OpenCV)
        color_rgb = Config.get_category_color(category)
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        # Draw rectangle
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color_bgr, 3)
        
        # Draw semi-transparent fill
        overlay = img_cv.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
        cv2.addWeighted(overlay, 0.1, img_cv, 0.9, 0, img_cv)
        
        # Prepare label text
        if show_confidence:
            text = f"{label}: {score:.2%}"
        else:
            text = label
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw label background
        label_y1 = max(y1 - text_height - 10, 0)
        label_y2 = label_y1 + text_height + 10
        cv2.rectangle(img_cv, (x1, label_y1), (x1 + text_width + 10, label_y2), color_bgr, -1)
        
        # Draw label text (white with shadow for readability)
        text_x = x1 + 5
        text_y = label_y1 + text_height + 5
        # Shadow
        cv2.putText(img_cv, text, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), thickness)
        # Main text
        cv2.putText(img_cv, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # Convert back to PIL Image
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def process_detections(raw_detections, confidence_threshold=0.7, nms_threshold=0.5):
    """
    Process raw model outputs into clean detection results
    
    Args:
        raw_detections: List of raw detection dictionaries
        confidence_threshold: Minimum confidence to keep detection
        nms_threshold: IoU threshold for NMS
    
    Returns:
        Filtered and processed list of detections
    """
    # Filter by confidence
    filtered = [
        det for det in raw_detections 
        if det['score'] >= confidence_threshold
    ]
    
    # Apply NMS
    filtered = apply_nms(filtered, nms_threshold)
    
    return filtered


def get_detection_statistics(detections):
    """
    Calculate statistics from detections
    
    Args:
        detections: List of detection dictionaries
    
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total': len(detections),
        'by_category': {},
        'by_class': {}
    }
    
    for det in detections:
        category = det['category']
        label = det['label']
        
        # Count by category
        if category not in stats['by_category']:
            stats['by_category'][category] = 0
        stats['by_category'][category] += 1
        
        # Count by class
        if label not in stats['by_class']:
            stats['by_class'][label] = 0
        stats['by_class'][label] += 1
    
    return stats
