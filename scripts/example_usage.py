#!/usr/bin/env python3
"""
Example Usage of Indian Road Object Detector

This script demonstrates how to use the IndianRoadDetector in your own code.
Run with: python example_usage.py
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model_utils import IndianRoadDetector, load_detector
import cv2
import numpy as np


def example_1_basic_inference():
    """Example 1: Basic inference on a single image"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Inference on Single Image")
    print("="*70)
    
    # Initialize detector with default settings
    detector = load_detector(
        model_name='rtdetr-l',
        confidence=0.5
    )
    
    # Create a dummy image for demonstration
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_image[100:200, 100:200] = [0, 255, 0]  # Green rectangle
    
    # Save dummy image
    cv2.imwrite('dummy_test.jpg', dummy_image)
    
    # Run inference
    print("\n→ Running inference on image...")
    results = detector.predict('dummy_test.jpg', confidence=0.5)
    
    # Print results
    print(f"\n✓ Detection complete!")
    print(f"  Image shape: {results['image_shape']}")
    print(f"  Model type: {results['model_type']}")
    print(f"  Total detections: {len(results['detections'])}")
    print(f"\n  Detected classes:")
    for class_name, count in results['class_counts'].items():
        print(f"    - {class_name}: {count}")
    
    # Cleanup
    os.remove('dummy_test.jpg')


def example_2_confidence_threshold():
    """Example 2: Adjusting confidence threshold"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Adjusting Confidence Threshold")
    print("="*70)
    
    detector = load_detector()
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite('confidence_test.jpg', dummy_image)
    
    # Test with different confidence thresholds
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    for conf in thresholds:
        print(f"\n→ Testing with confidence threshold: {conf}")
        detector.set_confidence(conf)
        results = detector.predict('confidence_test.jpg', confidence=conf)
        print(f"  Detections: {len(results['detections'])}")
        print(f"  Classes: {list(results['class_counts'].keys())}")
    
    # Cleanup
    os.remove('confidence_test.jpg')


def example_3_batch_processing():
    """Example 3: Processing multiple images in batch"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Batch Processing Multiple Images")
    print("="*70)
    
    detector = load_detector()
    
    # Create dummy test directory
    test_dir = Path('test_images')
    test_dir.mkdir(exist_ok=True)
    
    # Create sample images
    print("\n→ Creating sample images...")
    for i in range(3):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(f'test_images/image_{i}.jpg', image)
    
    # Process all images
    print("→ Processing batch of images...")
    results_list = []
    
    for image_path in sorted(test_dir.glob('*.jpg')):
        print(f"\n  Processing: {image_path.name}")
        results = detector.predict(str(image_path))
        results_list.append({
            'image': image_path.name,
            'detections': len(results['detections']),
            'classes': results['class_counts']
        })
        print(f"    - Found {len(results['detections'])} objects")
        print(f"    - Classes: {results['class_counts']}")
    
    # Summary
    print("\n✓ Batch processing complete!")
    print(f"  Total images processed: {len(results_list)}")
    print(f"  Total detections: {sum(r['detections'] for r in results_list)}")
    
    # Cleanup
    for f in test_dir.glob('*.jpg'):
        f.unlink()
    test_dir.rmdir()


def example_4_using_custom_weights():
    """Example 4: Using custom trained weights"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Using Custom Trained Weights")
    print("="*70)
    
    # Path to custom trained weights
    custom_weights = "runs/detect/train1/weights/best.pt"
    
    if os.path.exists(custom_weights):
        print(f"\n→ Loading custom weights from: {custom_weights}")
        detector = load_detector(
            model_name='rtdetr-l',
            weights_path=custom_weights,
            confidence=0.5
        )
        print("✓ Custom weights loaded successfully!")
    else:
        print(f"\n⚠ Custom weights not found at: {custom_weights}")
        print("→ Train a model first using: python train.py --mode train --data dataset.yaml")


def example_5_drawing_annotations():
    """Example 5: Drawing annotations on images"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Drawing Annotated Boxes")
    print("="*70)
    
    detector = load_detector()
    
    # Create dummy image
    dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.putText(dummy_image, 'Sample Detection', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    # Save original
    cv2.imwrite('original.jpg', dummy_image)
    
    # Run inference
    print("\n→ Running inference...")
    results = detector.predict('original.jpg')
    
    # Draw detections
    print("→ Drawing annotations...")
    annotated = detector.draw_detections(dummy_image, results['detections'])
    
    # Save annotated
    cv2.imwrite('annotated.jpg', annotated)
    print("✓ Annotated image saved to: annotated.jpg")
    
    # Cleanup
    os.remove('original.jpg')
    os.remove('annotated.jpg')


def example_6_performance_analysis():
    """Example 6: Performance analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Performance Analysis")
    print("="*70)
    
    import time
    
    detector = load_detector()
    
    # Create test image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite('perf_test.jpg', dummy_image)
    
    # Warm up
    print("\n→ Warming up model...")
    detector.predict('perf_test.jpg')
    
    # Measure inference time
    print("→ Measuring performance (10 runs)...")
    times = []
    
    for i in range(10):
        start = time.time()
        results = detector.predict('perf_test.jpg')
        end = time.time()
        times.append(end - start)
    
    # Statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time
    
    print(f"\n✓ Performance metrics:")
    print(f"  Average inference time: {avg_time*1000:.2f} ms")
    print(f"  Min inference time: {min_time*1000:.2f} ms")
    print(f"  Max inference time: {max_time*1000:.2f} ms")
    print(f"  FPS: {fps:.1f} images/second")
    
    # Cleanup
    os.remove('perf_test.jpg')


def example_7_error_handling():
    """Example 7: Error handling and edge cases"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Error Handling")
    print("="*70)
    
    detector = load_detector()
    
    # Test 1: Non-existent image
    print("\n→ Test 1: Non-existent image")
    try:
        results = detector.predict('non_existent.jpg')
    except FileNotFoundError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    # Test 2: Confidence out of range
    print("\n→ Test 2: Invalid confidence threshold")
    try:
        detector.set_confidence(1.5)
    except ValueError as e:
        print(f"  ✓ Caught expected error: {e}")
    
    # Test 3: Valid confidence threshold
    print("\n→ Test 3: Valid confidence threshold")
    try:
        detector.set_confidence(0.5)
        print("  ✓ Confidence set successfully")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("INDIAN ROAD OBJECT DETECTION - USAGE EXAMPLES")
    print("="*70)
    
    try:
        # Run examples
        example_1_basic_inference()
        example_2_confidence_threshold()
        example_3_batch_processing()
        example_4_using_custom_weights()
        example_5_drawing_annotations()
        example_6_performance_analysis()
        example_7_error_handling()
        
        # Summary
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        
        print("\nNext steps:")
        print("  1. Run the web interface: python app.py")
        print("  2. Train on custom data: python train.py --mode train --data dataset.yaml")
        print("  3. Check README.md for more information")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
