# TEDR System Architecture

## Overview

TEDR is a full-stack web application for object detection on Indian roads using the DETR (DEtection TRansformer) model.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         TEDR System                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Frontend (Browser)                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  index.html (UI Template)                                │   │
│  │  - Drag & Drop Upload Area                              │   │
│  │  - Image Preview                                         │   │
│  │  - Detection Results Display                            │   │
│  │  - Statistics Dashboard                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  style.css (Modern Styling)                             │   │
│  │  - Gradient backgrounds                                  │   │
│  │  - Glassmorphism effects                                │   │
│  │  - Responsive design                                     │   │
│  │  - Animations & transitions                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  main.js (Frontend Logic)                               │   │
│  │  - File upload handling                                  │   │
│  │  - AJAX API calls                                        │   │
│  │  - Results rendering                                     │   │
│  │  - Download functionality                                │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────────┘
                        │ HTTP/REST API
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (Flask Server)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  app.py (Flask Application)                             │   │
│  │  - Route: GET /                                          │   │
│  │    → Serve main UI                                       │   │
│  │  - Route: POST /api/detect                              │   │
│  │    → Image upload & detection                           │   │
│  │  - Route: GET /api/health                               │   │
│  │    → Health check                                        │   │
│  │  - Error handling & validation                          │   │
│  │  - CORS support                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────────┘
                        │ Function Calls
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Model Package (AI Core)                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  config.py (Configuration)                              │   │
│  │  - Model settings (DETR ResNet-50)                      │   │
│  │  - Confidence threshold (0.7)                           │   │
│  │  - NMS threshold (0.5)                                   │   │
│  │  - Category mappings                                     │   │
│  │  - Color schemes                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  detr_detector.py (DETR Wrapper)                        │   │
│  │  - Load DETR model from HuggingFace                     │   │
│  │  - Preprocess images                                     │   │
│  │  - Run inference                                         │   │
│  │  - Postprocess outputs                                   │   │
│  │  - Batch processing support                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  utils.py (Helper Functions)                            │   │
│  │  - calculate_iou() - IoU calculation                    │   │
│  │  - apply_nms() - Non-Maximum Suppression                │   │
│  │  - draw_boxes() - Visualization                         │   │
│  │  - process_detections() - Filter results                │   │
│  │  - get_detection_statistics() - Stats                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────────┘
                        │ PyTorch
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              External Dependencies                               │
├─────────────────────────────────────────────────────────────────┤
│  - PyTorch (Deep Learning Framework)                            │
│  - Transformers (HuggingFace - DETR Model)                      │
│  - OpenCV (Image Processing)                                    │
│  - Pillow (Image I/O)                                           │
│  - NumPy (Numerical Operations)                                 │
│  - Flask (Web Framework)                                        │
│  - Flask-CORS (Cross-Origin Support)                            │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Image Upload Flow
```
User drags/selects image
    ↓
JavaScript validates file (type, size)
    ↓
Image preview shown
    ↓
User clicks "Detect Objects"
    ↓
FormData created with image
    ↓
POST /api/detect
```

### 2. Detection Processing Flow
```
Flask receives image upload
    ↓
Validate file (extension, size)
    ↓
Load image with Pillow
    ↓
Pass to DETRDetector.detect()
    ↓
Model preprocessing:
  - Resize image (max 1333px)
  - Normalize (ImageNet stats)
  - Convert to tensor
    ↓
DETR model inference (on GPU/CPU)
    ↓
Model postprocessing:
  - Extract boxes, scores, labels
  - Map to COCO classes
  - Filter by confidence (≥0.7)
  - Apply NMS (IoU ≤0.5)
  - Convert to pixel coordinates
    ↓
Visualization:
  - Draw colored bounding boxes
  - Add labels with confidence
  - Generate statistics
    ↓
Convert annotated image to base64
    ↓
Return JSON response
```

### 3. Results Display Flow
```
JavaScript receives JSON response
    ↓
Parse detections and statistics
    ↓
Display annotated image (base64)
    ↓
Render statistics cards by category:
  - Vehicles (Blue)
  - Pedestrians (Green)
  - Animals (Orange)
  - Traffic (Yellow)
  - Others (Purple)
    ↓
Show detailed detection list
    ↓
Enable download button
```

## Component Responsibilities

### Frontend Components

**index.html**
- User interface structure
- Upload area with drag-and-drop
- Results display sections
- Loading animations
- Responsive layout

**style.css**
- Modern visual design
- Gradient backgrounds
- Glassmorphism effects
- Color-coded categories
- Responsive breakpoints
- Animations

**main.js**
- File handling
- API communication
- Results rendering
- Download functionality
- Error handling
- Toast notifications

### Backend Components

**app.py**
- Flask application setup
- Route handling
- Request validation
- Response formatting
- Error middleware
- CORS configuration

**detr_detector.py**
- Model loading (lazy)
- Image preprocessing
- Inference execution
- Output postprocessing
- Batch support

**config.py**
- Centralized settings
- Model parameters
- Category mappings
- Color definitions
- File constraints

**utils.py**
- IoU calculation
- NMS algorithm
- Box visualization
- Statistics generation
- Detection filtering

## Technology Stack

### Backend
- **Python 3.8+**: Programming language
- **Flask 2.3+**: Web framework
- **PyTorch 2.0+**: Deep learning framework
- **Transformers 4.30+**: HuggingFace library for DETR
- **OpenCV 4.8+**: Image processing
- **Pillow 9.0+**: Image I/O
- **NumPy 1.24+**: Numerical operations

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling (with custom properties)
- **JavaScript (ES6+)**: Logic
- **Font Awesome 6.4**: Icons
- **Google Fonts (Poppins)**: Typography

### Model
- **DETR ResNet-50**: Pre-trained on COCO dataset
- **Input**: RGB images (max 1333px)
- **Output**: Bounding boxes, labels, confidence scores
- **Classes**: 80+ COCO classes (mapped to 5 categories)

## Deployment Options

### Development
```bash
python run.py
# Runs on http://localhost:5000
# Debug mode disabled for security
```

### Production
```bash
# Using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 run:app

# With Nginx reverse proxy (recommended)
# See nginx configuration in deployment docs
```

### Cloud Platforms
- **Heroku**: Deploy with Procfile
- **Google Cloud Run**: Containerized deployment
- **AWS EC2/ECS**: VM or container deployment
- **Azure App Service**: PaaS deployment

## Performance Characteristics

### Speed
- **GPU**: 100-500ms per image
- **CPU**: 2-10 seconds per image
- **Model Loading**: ~3 seconds (first request only)
- **Model Size**: ~160MB (cached after first download)

### Accuracy
- **Confidence Threshold**: 0.7 (configurable)
- **NMS Threshold**: 0.5 (configurable)
- **Precision**: High for common COCO classes
- **Recall**: Moderate for small/occluded objects

### Resource Usage
- **Memory**: ~1-2GB (with model loaded)
- **GPU Memory**: ~2-4GB (if using GPU)
- **Disk**: ~500MB (including dependencies)

## Security Features

- ✅ Debug mode disabled in production
- ✅ File type validation (JPG, PNG, JPEG, WebP)
- ✅ File size limits (10MB max)
- ✅ CORS properly configured
- ✅ Error messages don't leak sensitive info
- ✅ No hardcoded secrets
- ✅ Input sanitization

## Extensibility

### Easy to Add
- New object categories (edit config.py)
- Different DETR models (change MODEL_NAME)
- Custom confidence thresholds
- Additional file formats
- More API endpoints

### Requires More Work
- Video processing (need frame extraction)
- Real-time webcam (WebRTC integration)
- Fine-tuning on custom dataset
- Multi-language UI
- User authentication

## File Structure Summary

```
TEDR/
├── model/              # AI model package
├── app/               # Web application
│   ├── static/       # Frontend assets
│   └── templates/    # HTML templates
├── run.py            # Entry point
├── validate_setup.py # Setup checker
├── requirements.txt  # Dependencies
├── README.md         # Main documentation
├── QUICKSTART.md     # Quick start guide
└── .gitignore       # Git ignore rules
```

Total Lines of Code: ~2,100+ lines
- Python: ~1,600 lines
- HTML: ~130 lines
- CSS: ~450 lines
- JavaScript: ~270 lines
