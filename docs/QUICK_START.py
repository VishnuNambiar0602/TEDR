"""
QUICK START GUIDE - Indian Road Object Detection with RT-DETR

This file contains quick reference commands and examples for common tasks.
"""

# ============================================================================
# INSTALLATION (Run these first)
# ============================================================================

"""
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install for specific GPU type
# For NVIDIA CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For NVIDIA CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio
"""


# ============================================================================
# RUNNING THE WEB APPLICATION
# ============================================================================

"""
# Start the Gradio web interface (main application)
python app.py

# Then open browser to: http://localhost:7860

# Tips:
# - First run downloads model weights (~300-800MB)
# - Model initializes on startup
# - Access from other machines: http://<your-ip>:7860
"""


# ============================================================================
# TRAINING ON CUSTOM DATA
# ============================================================================

"""
# 1. Prepare dataset in YOLO format (see README.md for details)

# 2. Create dataset.yaml (example below)

# 3. Basic training
python train.py --mode train \
  --data dataset.yaml \
  --model rtdetr-l.pt \
  --epochs 100 \
  --batch_size 16

# Training with custom image size
python train.py --mode train \
  --data dataset.yaml \
  --model rtdetr-l.pt \
  --epochs 100 \
  --batch_size 16
# (Modify IMG_SIZE variable in train.py for different sizes)

# Training with multiple GPUs
python train.py --mode train \
  --data dataset.yaml \
  --model rtdetr-l.pt \
  --epochs 100 \
  --batch_size 32 \
  --device 0

# Resume training from checkpoint
python train.py --mode train \
  --data dataset.yaml \
  --model runs/detect/train1/weights/last.pt \
  --epochs 50

# Use faster (but less accurate) model for quick iteration
python train.py --mode train \
  --data dataset.yaml \
  --model rtdetr-s.pt \
  --epochs 50 \
  --batch_size 32
"""


# ============================================================================
# EVALUATION AND TESTING
# ============================================================================

"""
# Evaluate trained model on validation set
python train.py --mode evaluate \
  --weights runs/detect/train1/weights/best.pt \
  --data dataset.yaml

# Run inference on single image
python train.py --mode predict \
  --weights runs/detect/train1/weights/best.pt \
  --image path/to/test_image.jpg

# Test with pre-trained model
python train.py --mode predict \
  --weights rtdetr-l.pt \
  --image path/to/test_image.jpg
"""


# ============================================================================
# USING CUSTOM TRAINED WEIGHTS
# ============================================================================

"""
# In app.py, modify line ~40:

CUSTOM_WEIGHTS_PATH = "runs/detect/train1/weights/best.pt"

# Then run:
python app.py

# Now the web interface will use your custom trained model
"""


# ============================================================================
# CONFIGURATION EXAMPLES
# ============================================================================

"""
# Example dataset.yaml for your dataset:

path: C:/Users/YourName/data/my_dataset
train: images/train
val: images/val
test: images/test

nc: 13

names:
  0: auto_rickshaw
  1: truck
  2: bus
  3: motorcycle
  4: car
  5: bicycle
  6: bull_cart
  7: cow
  8: dog
  9: goat
  10: pedestrian
  11: traffic_sign
  12: pole
"""


# ============================================================================
# PYTHON EXAMPLES (Use in your own scripts)
# ============================================================================

"""
# Example 1: Basic inference on image
from model_utils import load_detector
import cv2

detector = load_detector(model_name='rtdetr-l', confidence=0.5)
image = cv2.imread('test_image.jpg')
results = detector.predict_array(image, confidence=0.5)

print(f"Found {len(results['detections'])} objects")
for det in results['detections']:
    print(f"  - {det['class']}: {det['confidence']:.2f}")

# Draw and save result
image_annotated = detector.draw_detections(image, results['detections'])
cv2.imwrite('result.jpg', image_annotated)


# Example 2: Batch processing multiple images
from pathlib import Path

image_dir = Path('images/')
detector = load_detector()

for image_path in image_dir.glob('*.jpg'):
    results = detector.predict(str(image_path), confidence=0.5)
    print(f"{image_path.name}: {results['class_counts']}")


# Example 3: Adjusting confidence threshold
detector.set_confidence(0.7)  # More strict
results = detector.predict_array(image)

detector.set_confidence(0.3)  # More lenient
results = detector.predict_array(image)


# Example 4: Using custom trained weights
detector = load_detector(
    model_name='rtdetr-l',
    weights_path='runs/detect/train1/weights/best.pt',
    confidence=0.5
)
results = detector.predict_array(image)
"""


# ============================================================================
# COMMON ISSUES AND SOLUTIONS
# ============================================================================

"""
ISSUE: "CUDA out of memory"
SOLUTION:
  1. Use smaller model: DEFAULT_MODEL = 'rtdetr-s'
  2. Reduce batch size: --batch_size 8
  3. Use CPU: Set device='cpu' in model_utils.py

ISSUE: "No objects detected"
SOLUTION:
  1. Lower confidence threshold: DEFAULT_CONFIDENCE = 0.3
  2. Ensure image quality (clear, well-lit)
  3. Fine-tune model on your specific data

ISSUE: "Model loading fails"
SOLUTION:
  1. Reinstall PyTorch:
     pip install torch torchvision --force-reinstall
  2. Check internet (needs to download weights)
  3. Check disk space (needs ~5GB)

ISSUE: "Slow inference"
SOLUTION:
  1. Use GPU instead of CPU
  2. Use smaller model: rtdetr-s
  3. Reduce image size (modify IMG_SIZE)
  4. Enable image caching: enable_cache = True

ISSUE: "Gradio interface won't load"
SOLUTION:
  1. Check port 7860 is not blocked: 
     Check firewall settings
  2. Try different port:
     Modify server_port in config.ini
  3. Clear cache:
     Delete __pycache__ directories
"""


# ============================================================================
# USEFUL DIRECTORIES
# ============================================================================

"""
After running the application:

runs/detect/train1/              # Training output
├── weights/
│   ├── best.pt               # Best model weights
│   ├── last.pt               # Last epoch weights
│   └── epoch*.pt             # Intermediate weights
├── results.csv               # Training metrics
├── args.yaml                 # Training configuration
└── plots/                    # Training visualizations

results/                      # Inference results (if enabled)
├── predictions.txt          # Detection outputs
└── images/                  # Annotated images
"""


# ============================================================================
# PERFORMANCE TIPS
# ============================================================================

"""
For Faster Inference:
  1. Use GPU (NVIDIA CUDA recommended)
  2. Use smaller model (rtdetr-s instead of rtdetr-l)
  3. Reduce image size (modify IMG_SIZE in train.py)
  4. Use batch processing when possible

For Better Accuracy:
  1. Use larger model (rtdetr-x)
  2. Fine-tune on your specific data
  3. Use higher resolution images
  4. Lower confidence threshold
  5. Ensemble multiple models

For Faster Training:
  1. Use mixed precision (fp16)
  2. Increase batch size (if GPU memory allows)
  3. Reduce image size
  4. Use data augmentation
  5. Cache images
  6. Use multiple GPUs
"""


# ============================================================================
# DEPLOYMENT OPTIONS
# ============================================================================

"""
# 1. Gradio Web App (current)
python app.py

# 2. Alternative: Streamlit Web App
# (Create streamlit_app.py similar to app.py)

# 3. FastAPI REST API
# (Use model_utils.py with FastAPI for production)

# 4. Export to ONNX (faster inference)
from ultralytics import YOLO
model = YOLO('runs/detect/train1/weights/best.pt')
model.export(format='onnx')

# 5. Mobile deployment (TFLite, CoreML, etc.)
model.export(format='tflite')  # For mobile
"""


# ============================================================================
# GETTING HELP
# ============================================================================

"""
Documentation:
  - Ultralytics YOLO: https://docs.ultralytics.com/
  - Gradio: https://gradio.app/
  - Hugging Face Transformers: https://huggingface.co/

Datasets:
  - India Driving Dataset: https://idd.is.iitd.ac.in/
  - COCO Dataset: https://cocodataset.org/

Papers:
  - RT-DETR: https://arxiv.org/abs/2304.08069
  - DETR: https://arxiv.org/abs/2005.12139
"""
