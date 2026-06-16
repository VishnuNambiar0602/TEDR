# API Documentation

Complete API reference for the Indian Road Object Detector.

## Table of Contents
1. [Core Classes](#core-classes)
2. [Functions](#functions)
3. [Data Structures](#data-structures)
4. [Examples](#examples)
5. [Error Handling](#error-handling)

---

## Core Classes

### `IndianRoadDetector`

Main detector class for object detection.

#### Constructor

```python
IndianRoadDetector(
    model_name: str = 'rtdetr-l',
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    confidence_threshold: float = 0.5,
)
```

**Parameters:**
- `model_name` (str): Model identifier
  - Ultralytics: `'rtdetr-s'`, `'rtdetr-m'`, `'rtdetr-l'`, `'rtdetr-x'`
  - Hugging Face: Model ID like `'PekingU/rtdetr_r50vd'`
  - Default: `'rtdetr-l'`

- `weights_path` (str, optional): Path to custom .pt or .safetensors weights
  - If provided and exists, loads these weights
  - Supports both Ultralytics .pt and Hugging Face formats

- `device` (str, optional): Computation device
  - `'cuda'`: NVIDIA GPU (if available)
  - `'cpu'`: CPU (slower)
  - `None` (default): Auto-detect (CUDA if available, else CPU)

- `confidence_threshold` (float): Detection confidence threshold
  - Range: 0.0 to 1.0
  - Default: 0.5
  - Higher = fewer but more confident detections

**Example:**

```python
from model_utils import IndianRoadDetector

# Initialize with defaults
detector = IndianRoadDetector()

# With custom settings
detector = IndianRoadDetector(
    model_name='rtdetr-m',
    weights_path='runs/detect/train1/weights/best.pt',
    device='cuda',
    confidence_threshold=0.6
)

# With pre-trained Hugging Face model
detector = IndianRoadDetector(
    model_name='PekingU/rtdetr_r50vd',
    confidence_threshold=0.5
)
```

---

#### Methods

### `predict()`

Run inference on an image file.

```python
detector.predict(
    image_path: str,
    confidence: Optional[float] = None,
) -> Dict
```

**Parameters:**
- `image_path` (str): Path to input image file
  - Supported formats: JPEG, PNG, BMP, TIFF, etc.
  - Will raise `FileNotFoundError` if file doesn't exist

- `confidence` (float, optional): Override instance confidence threshold
  - If not provided, uses `self.confidence_threshold`

**Returns:** Dictionary with detection results (see [Detection Results](#detection-results))

**Raises:**
- `FileNotFoundError`: If image file doesn't exist
- `ValueError`: If image cannot be read

**Example:**

```python
# Load and detect
results = detector.predict('street_scene.jpg')

# With custom confidence
results = detector.predict('street_scene.jpg', confidence=0.7)
```

---

### `predict_array()`

Run inference on a numpy array image.

```python
detector.predict_array(
    image: np.ndarray,
    confidence: Optional[float] = None,
) -> Dict
```

**Parameters:**
- `image` (np.ndarray): Image as numpy array
  - Shape: (height, width, 3)
  - Format: BGR (OpenCV standard) or RGB
  - Data type: uint8 (0-255) or float32 (0-1)

- `confidence` (float, optional): Override threshold

**Returns:** Dictionary with detection results

**Example:**

```python
import cv2

# Load image
image = cv2.imread('street_scene.jpg')  # BGR format

# Detect
results = detector.predict_array(image, confidence=0.5)

# Process results
for detection in results['detections']:
    print(f"{detection['class']}: {detection['confidence']:.2f}")
```

---

### `draw_detections()`

Draw bounding boxes on image.

```python
detector.draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    thickness: int = 2,
) -> np.ndarray
```

**Parameters:**
- `image` (np.ndarray): Input image (BGR format)
  - Same format as returned by cv2.imread()

- `detections` (List[Dict]): List of detection dictionaries
  - Each dict should have: `'class'`, `'confidence'`, `'bbox'`

- `thickness` (int): Bounding box line thickness in pixels
  - Default: 2

**Returns:** Annotated image with drawn boxes (same format as input)

**Example:**

```python
# Get image and run detection
image = cv2.imread('street_scene.jpg')
results = detector.predict_array(image)

# Draw boxes
annotated = detector.draw_detections(image, results['detections'])

# Save result
cv2.imwrite('result_annotated.jpg', annotated)

# Display
cv2.imshow('Detections', annotated)
cv2.waitKey(0)
```

---

### `set_confidence()`

Update detection confidence threshold.

```python
detector.set_confidence(confidence: float) -> None
```

**Parameters:**
- `confidence` (float): New confidence threshold
  - Range: 0.0 to 1.0

**Raises:**
- `ValueError`: If confidence not in range [0, 1]

**Example:**

```python
# Start strict
detector.set_confidence(0.9)
results = detector.predict_array(image)

# Switch to lenient
detector.set_confidence(0.3)
results = detector.predict_array(image)
```

---

## Functions

### `load_detector()`

Convenience function to load detector.

```python
load_detector(
    model_name: str = 'rtdetr-l',
    weights_path: Optional[str] = None,
    confidence: float = 0.5,
) -> IndianRoadDetector
```

**Returns:** Initialized `IndianRoadDetector` instance

**Example:**

```python
from model_utils import load_detector

detector = load_detector(
    model_name='rtdetr-l',
    weights_path='custom_weights.pt',
    confidence=0.5
)
```

---

## Data Structures

### Detection Results Dictionary

Returned by `predict()` and `predict_array()`.

```python
{
    'detections': [
        {
            'class': str,           # Class name (e.g., 'auto_rickshaw')
            'confidence': float,    # Confidence score (0-1)
            'bbox': [x1, y1, x2, y2],  # Bounding box coordinates
            'class_id': int,        # Class ID (0-indexed)
        },
        # ... more detections
    ],
    'class_counts': {
        'auto_rickshaw': 2,
        'cow': 1,
        'pedestrian': 3,
        # ... count for each class
    },
    'image_shape': (height, width, 3),  # Input image shape
    'model_type': str,              # 'ultralytics' or 'huggingface'
}
```

### Detection Entry

```python
{
    'class': str,           # Detection class name
    'confidence': float,    # Confidence (0.0 to 1.0)
    'bbox': [float, float, float, float],  # [x1, y1, x2, y2] in pixels
    'class_id': int,        # Numeric class ID
}
```

---

## Class Names Map

```python
INDIAN_ROAD_CLASSES = {
    # Vehicles
    'auto_rickshaw': (255, 100, 0),
    'truck': (0, 0, 255),
    'bus': (0, 165, 255),
    'motorcycle': (200, 0, 200),
    'car': (0, 255, 0),
    'bicycle': (255, 255, 0),
    'bull_cart': (165, 42, 42),
    
    # Animals
    'cow': (255, 0, 255),
    'dog': (128, 0, 128),
    'goat': (255, 192, 203),
    
    # Pedestrians
    'pedestrian': (255, 0, 0),
    'jaywalker': (0, 255, 255),
    
    # Context
    'traffic_sign': (0, 128, 255),
    'pole': (128, 128, 128),
}
```

---

## Examples

### Example 1: Simple Detection

```python
from model_utils import load_detector
import cv2

# Load detector
detector = load_detector()

# Load and detect
image = cv2.imread('street.jpg')
results = detector.predict_array(image)

# Print results
print(f"Found {len(results['detections'])} objects")
for det in results['detections']:
    print(f"  - {det['class']}: {det['confidence']:.2f}")
```

### Example 2: Annotate and Save

```python
from model_utils import load_detector
import cv2

detector = load_detector()

# Load image
image = cv2.imread('street.jpg')

# Run detection
results = detector.predict_array(image)

# Annotate
output_image = detector.draw_detections(image, results['detections'])

# Save
cv2.imwrite('result.jpg', output_image)
```

### Example 3: Batch Processing

```python
from model_utils import load_detector
import cv2
from pathlib import Path

detector = load_detector()

image_dir = Path('images/')

for image_path in image_dir.glob('*.jpg'):
    image = cv2.imread(str(image_path))
    results = detector.predict_array(image)
    
    print(f"{image_path.name}: {results['class_counts']}")
```

### Example 4: Confidence Adjustment

```python
from model_utils import load_detector
import cv2

detector = load_detector()
image = cv2.imread('street.jpg')

# Try different thresholds
for conf in [0.3, 0.5, 0.7, 0.9]:
    detector.set_confidence(conf)
    results = detector.predict_array(image)
    print(f"Confidence {conf}: {len(results['detections'])} detections")
```

### Example 5: Custom Weights

```python
from model_utils import load_detector
import cv2

# Load with custom weights
detector = load_detector(
    model_name='rtdetr-l',
    weights_path='runs/detect/train1/weights/best.pt',
    confidence=0.5
)

image = cv2.imread('street.jpg')
results = detector.predict_array(image)

print(f"Detected: {results['class_counts']}")
```

### Example 6: Integration with OpenCV

```python
from model_utils import load_detector
import cv2

detector = load_detector()

# Video processing example
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect on each frame
    results = detector.predict_array(frame)
    
    # Draw annotations
    annotated = detector.draw_detections(frame, results['detections'])
    
    # Display
    cv2.imshow('Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Example 7: REST API (FastAPI)

```python
from fastapi import FastAPI, File, UploadFile
from model_utils import load_detector
import cv2
import numpy as np

app = FastAPI()
detector = load_detector()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = detector.predict_array(image, confidence=0.5)
    
    return results

# Run with: uvicorn api:app --reload
```

---

## Error Handling

### Common Errors

#### FileNotFoundError

```python
try:
    results = detector.predict('nonexistent.jpg')
except FileNotFoundError as e:
    print(f"Error: {e}")
```

#### ValueError (Invalid Confidence)

```python
try:
    detector.set_confidence(1.5)  # Out of range
except ValueError as e:
    print(f"Error: {e}")
```

#### CUDA Out of Memory

```python
# Solution 1: Use CPU
detector = IndianRoadDetector(device='cpu')

# Solution 2: Use smaller model
detector = IndianRoadDetector(model_name='rtdetr-s')

# Solution 3: Reduce image size
# (Implement image resizing before prediction)
```

#### Model Not Found

```python
# Ensure model_name is valid:
# Valid: 'rtdetr-s', 'rtdetr-m', 'rtdetr-l', 'rtdetr-x'
# Valid: 'PekingU/rtdetr_r50vd'

# Check internet connection (first run downloads weights)
```

---

## Performance Optimization

### Inference Speed Tips

1. **Use appropriate model size:**
   ```python
   # Fast inference
   detector = IndianRoadDetector(model_name='rtdetr-s')
   
   # Better accuracy
   detector = IndianRoadDetector(model_name='rtdetr-l')
   ```

2. **Batch processing:**
   ```python
   # Instead of detecting one image at a time
   results_list = []
   for image_path in image_paths:
       results = detector.predict(image_path)
       results_list.append(results)
   ```

3. **Use GPU:**
   ```python
   # Ensure CUDA is available
   detector = IndianRoadDetector(device='cuda')
   ```

### Memory Optimization

1. **Process images one at a time:**
   ```python
   # Good: Process sequentially
   for image_path in image_paths:
       results = detector.predict(image_path)
       # Process and clear memory
   ```

2. **Reduce image size:**
   ```python
   # Before detection
   small_image = cv2.resize(image, (640, 640))
   results = detector.predict_array(small_image)
   ```

---

## Advanced Usage

### Custom Model Architecture

To use a different architecture (e.g., YOLOv8):

```python
# Modify model_utils.py to load different models
from ultralytics import YOLO

model = YOLO('yolov8x.pt')  # Different model

# Use similar structure for integration
```

### Fine-tuning Documentation

See `train.py` for complete training and fine-tuning examples.

---

## Version Information

- **Version**: 1.0.0
- **PyTorch**: 2.1+
- **CUDA**: 11.8+ or 12.1+
- **Python**: 3.10+
- **Ultralytics**: 8.1+

---

## License

MIT License - See LICENSE file

---

## Support

For issues or questions:
1. Check examples in this file
2. See INSTALLATION.md for setup issues
3. See README.md for general usage
4. Check Ultralytics docs: https://docs.ultralytics.com/
