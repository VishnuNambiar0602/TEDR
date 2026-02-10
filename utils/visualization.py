"""Visualization utilities for TEDR object detection."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Union
import random


# Color palette for bounding boxes
COLORS = [
    (59, 130, 246),   # Blue
    (239, 68, 68),    # Red
    (16, 185, 129),   # Green
    (245, 158, 11),   # Orange
    (139, 92, 246),   # Purple
    (236, 72, 153),   # Pink
    (20, 184, 166),   # Teal
    (249, 115, 22),   # Dark Orange
]


def get_color(index: int) -> Tuple[int, int, int]:
    """Get color for bounding box based on index.
    
    Args:
        index: Index of detection
        
    Returns:
        RGB color tuple
    """
    return COLORS[index % len(COLORS)]


def draw_bounding_boxes(
    image: Union[Image.Image, np.ndarray],
    detections: List[Dict],
    confidence_threshold: float = 0.0,
    show_labels: bool = True,
    show_confidence: bool = True,
    thickness: int = 3
) -> Union[Image.Image, np.ndarray]:
    """Draw bounding boxes on image.
    
    Args:
        image: PIL Image or numpy array
        detections: List of detection dictionaries with 'bbox', 'label', 'confidence'
        confidence_threshold: Minimum confidence to draw
        show_labels: Whether to show label text
        show_confidence: Whether to show confidence scores
        thickness: Line thickness for bounding boxes
        
    Returns:
        Image with bounding boxes drawn (same type as input)
    """
    is_pil = isinstance(image, Image.Image)
    
    if is_pil:
        # Convert to numpy for drawing
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image.copy()
    
    # Filter detections by confidence
    filtered_detections = [
        d for d in detections 
        if d.get('confidence', 0) >= confidence_threshold
    ]
    
    # Draw each detection
    for idx, detection in enumerate(filtered_detections):
        bbox = detection['bbox']
        label = detection.get('label', 'object')
        confidence = detection.get('confidence', 0)
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color
        color = get_color(idx)
        color_bgr = (color[2], color[1], color[0])  # Convert RGB to BGR
        
        # Draw rectangle
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color_bgr, thickness)
        
        # Draw label
        if show_labels:
            if show_confidence:
                text = f"{label} {confidence:.2f}"
            else:
                text = label
            
            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )
            
            # Draw text background
            cv2.rectangle(
                img_bgr,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color_bgr,
                -1
            )
            
            # Draw text
            cv2.putText(
                img_bgr,
                text,
                (x1 + 2, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
    
    # Convert back to original format
    if is_pil:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)
    else:
        return img_bgr


def create_detection_visualization(
    image_path: str,
    detections: List[Dict],
    output_path: str = None,
    **kwargs
) -> Image.Image:
    """Create visualization of detections and optionally save.
    
    Args:
        image_path: Path to input image
        detections: List of detection dictionaries
        output_path: Optional path to save visualization
        **kwargs: Additional arguments for draw_bounding_boxes
        
    Returns:
        Visualization as PIL Image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Draw bounding boxes
    result = draw_bounding_boxes(image, detections, **kwargs)
    
    # Save if output path provided
    if output_path:
        result.save(output_path)
    
    return result


def create_comparison_grid(
    images: List[Union[Image.Image, np.ndarray]],
    titles: List[str] = None,
    grid_size: Tuple[int, int] = None
) -> Image.Image:
    """Create grid of images for comparison.
    
    Args:
        images: List of images
        titles: Optional list of titles for each image
        grid_size: Optional grid size as (rows, cols). Auto-calculated if None
        
    Returns:
        Grid image as PIL Image
    """
    n_images = len(images)
    
    # Calculate grid size if not provided
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # Convert all images to PIL and get max dimensions
    pil_images = []
    max_width = 0
    max_height = 0
    
    for img in images:
        if isinstance(img, np.ndarray):
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        
        pil_images.append(img)
        max_width = max(max_width, img.width)
        max_height = max(max_height, img.height)
    
    # Create grid canvas
    grid_width = cols * max_width
    grid_height = rows * max_height
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # Paste images
    for idx, img in enumerate(pil_images):
        row = idx // cols
        col = idx % cols
        x = col * max_width
        y = row * max_height
        grid.paste(img, (x, y))
        
        # Add title if provided
        if titles and idx < len(titles):
            draw = ImageDraw.Draw(grid)
            # Try multiple font paths for cross-platform compatibility
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                "/System/Library/Fonts/Helvetica.ttc",  # macOS
                "C:\\Windows\\Fonts\\arial.ttf",  # Windows
            ]
            font = None
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, 20)
                    break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()
            draw.text((x + 10, y + 10), titles[idx], fill=(0, 0, 0), font=font)
    
    return grid


def save_detections_summary(
    detections: List[Dict],
    output_path: str
):
    """Save detections summary to text file.
    
    Args:
        detections: List of detection dictionaries
        output_path: Path to save summary file
    """
    with open(output_path, 'w') as f:
        f.write("Object Detection Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total detections: {len(detections)}\n\n")
        
        for idx, detection in enumerate(detections, 1):
            label = detection.get('label', 'unknown')
            confidence = detection.get('confidence', 0)
            bbox = detection.get('bbox', [0, 0, 0, 0])
            
            f.write(f"Detection {idx}:\n")
            f.write(f"  Label: {label}\n")
            f.write(f"  Confidence: {confidence:.4f}\n")
            f.write(f"  Bounding Box: {bbox}\n\n")
