"""
Example script demonstrating TEDR object detection usage.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.detr_model import DETRModel
from utils.visualization import draw_bounding_boxes, create_detection_visualization
from utils.preprocessing import resize_image
from PIL import Image
import argparse


def detect_objects_in_image(image_path: str, output_path: str = None, confidence: float = 0.7):
    """
    Detect objects in an image and optionally save visualization.
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save output visualization
        confidence: Confidence threshold for detections
    """
    print(f"Loading image: {image_path}")
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Initialize model
    print("Initializing DETR model...")
    model = DETRModel(
        model_name="facebook/detr-resnet-50",
        confidence_threshold=confidence
    )
    
    # Perform detection
    print("Running object detection...")
    results = model.detect(image)
    
    # Print results
    print(f"\nDetected {results['num_detections']} objects:")
    print("-" * 60)
    
    for idx, detection in enumerate(results['detections'], 1):
        label = detection['label']
        conf = detection['confidence']
        bbox = detection['bbox']
        print(f"{idx}. {label}: {conf:.2%} - BBox: {[int(x) for x in bbox]}")
    
    # Create visualization
    if output_path:
        print(f"\nCreating visualization...")
        viz_image = draw_bounding_boxes(
            image,
            results['detections'],
            show_labels=True,
            show_confidence=True
        )
        viz_image.save(output_path)
        print(f"Saved visualization to: {output_path}")
    
    return results


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description="TEDR Object Detection - Detect objects in images"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save output visualization"
    )
    parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=0.7,
        help="Confidence threshold (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    # Set default output path if not specified
    if args.output is None:
        input_path = Path(args.image)
        args.output = str(input_path.parent / f"{input_path.stem}_detected{input_path.suffix}")
    
    # Run detection
    detect_objects_in_image(args.image, args.output, args.confidence)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
