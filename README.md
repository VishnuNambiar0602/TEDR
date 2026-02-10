# TEDR - Transformer-based Object Detection for Indian Roads

<div align="center">

🚗 **AI-Powered Object Detection System for Indian Road Scenarios** 🚗

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

TEDR (Transformer-based object DEtection for Indian Roads) is a state-of-the-art object detection system specifically designed to handle the unique challenges of Indian road scenarios. Built using the **DETR (DEtection TRansformer)** architecture from Facebook AI Research, this system provides accurate real-time detection of vehicles, pedestrians, animals, and traffic elements commonly found on Indian roads.

### Why TEDR?

Indian roads present unique challenges:
- **Dense, Mixed Traffic**: Cars, buses, trucks, auto-rickshaws, motorcycles, bicycles, and scooters all share the same space
- **Unpredictable Pedestrians**: People crossing at random points, not just at crosswalks
- **Street Animals**: Cows, dogs, goats, and other animals commonly found on roads
- **Chaotic Intersections**: Complex traffic patterns without strict lane discipline
- **Various Lighting Conditions**: From bright sunlight to poorly lit streets

TEDR is optimized to detect and track these diverse objects with high accuracy.

## ✨ Features

### 🤖 Advanced AI Model
- **DETR Architecture**: Leverages transformer-based detection from Hugging Face
- **Pre-trained Model**: Uses `facebook/detr-resnet-50` with COCO dataset weights
- **High Accuracy**: Optimized confidence thresholding (default 0.7)
- **Smart Filtering**: Non-Maximum Suppression (NMS) for cleaner results
- **GPU Accelerated**: Automatic GPU detection with CPU fallback

### 🎨 Modern Web Interface
- **Sleek Design**: Beautiful gradient backgrounds with glassmorphism effects
- **Drag & Drop**: Intuitive image upload with drag-and-drop support
- **Real-time Preview**: See your image before processing
- **Live Detection**: Animated loading states during processing
- **Rich Results**: Annotated images with color-coded bounding boxes
- **Statistics Dashboard**: Visual breakdown of detected objects by category
- **Download Results**: Save annotated images instantly
- **Responsive**: Works perfectly on desktop, tablet, and mobile

### 🚦 Object Detection Categories

**Vehicles** (Blue boxes)
- Cars, buses, trucks, motorcycles, bicycles, trains

**Pedestrians** (Green boxes)
- People walking, standing, crossing roads

**Animals** (Orange boxes)
- Cows, dogs, cats, horses, sheep, birds, elephants

**Traffic Elements** (Yellow boxes)
- Traffic lights, stop signs

**Others** (Purple boxes)
- Any other detected objects

### 🔧 Technical Capabilities
- **Batch Processing**: Support for multiple images
- **Flexible Input**: JPG, PNG, JPEG, WebP formats
- **Size Limits**: Up to 10MB per image
- **API Access**: RESTful API for integration
- **CORS Support**: Cross-origin requests enabled
- **Error Handling**: Comprehensive validation and error messages

## 🖼️ Demo

### Upload Interface
![Upload Interface](docs/images/upload.png)

### Detection Results
![Detection Results](docs/images/results.png)

### Statistics Dashboard
![Statistics](docs/images/statistics.png)

*Note: Screenshots to be added*

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Step 1: Clone the Repository
```bash
git clone https://github.com/VishnuNambiar0602/TEDR.git
cd TEDR
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The first time you run the application, it will download the DETR model (~160MB) from Hugging Face. This only happens once.

### Step 4: Run the Application
```bash
python run.py
```

The application will start on `http://localhost:5000`

Open your browser and navigate to the URL to start using TEDR!

## 📖 Usage

### Web Interface

1. **Upload Image**
   - Click the upload area or drag and drop an image
   - Supported formats: JPG, PNG, JPEG, WebP
   - Maximum size: 10MB

2. **Preview**
   - Review your uploaded image
   - Click "Detect Objects" to start processing

3. **View Results**
   - See annotated image with bounding boxes
   - Review statistics by category
   - Check detailed detection list with confidence scores

4. **Download**
   - Click "Download Result" to save the annotated image
   - Use "Upload Another" to process more images

### API Usage

#### Detect Objects
```bash
curl -X POST http://localhost:5000/api/detect \
  -F "image=@path/to/your/image.jpg"
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "box": [x1, y1, x2, y2],
      "score": 0.95,
      "label": "car",
      "label_id": 2,
      "category": "vehicle"
    }
  ],
  "statistics": {
    "total": 10,
    "by_category": {
      "vehicle": 5,
      "pedestrian": 3,
      "animal": 2
    },
    "by_class": {
      "car": 3,
      "motorcycle": 2,
      "person": 3,
      "cow": 2
    }
  },
  "annotated_image": "data:image/png;base64,..."
}
```

#### Health Check
```bash
curl http://localhost:5000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "model": "facebook/detr-resnet-50",
  "device": "cuda"
}
```

## 🔬 Technical Details

### Model Architecture

**DETR (DEtection TRansformer)**
- Developed by Facebook AI Research (FAIR)
- End-to-end object detection using transformers
- No need for hand-crafted components like NMS or anchor generation
- Backbone: ResNet-50
- Input: RGB images (resized to max 1333px)
- Output: Bounding boxes, class labels, and confidence scores

### Preprocessing Pipeline
1. Image loading and conversion to RGB
2. Resizing (maintaining aspect ratio, max dimension 1333px)
3. Normalization using ImageNet statistics
4. Tensor conversion for model input

### Post-processing Pipeline
1. Raw model output extraction
2. Confidence-based filtering (threshold: 0.7)
3. Non-Maximum Suppression (IoU threshold: 0.5)
4. Box coordinate conversion to pixel space
5. Class mapping to Indian road categories
6. Bounding box visualization with labels

### Supported Object Classes

The model detects 80+ object classes from COCO dataset. The most relevant for Indian roads include:

| Category | Classes |
|----------|---------|
| **Vehicles** | car, bus, truck, motorcycle, bicycle, train |
| **Pedestrians** | person |
| **Animals** | cow, dog, cat, horse, sheep, bird, elephant, bear, zebra, giraffe |
| **Traffic** | traffic light, stop sign |

## 📡 API Documentation

### Endpoints

#### `GET /`
Serves the main web interface.

#### `POST /api/detect`
Performs object detection on an uploaded image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `image` field containing the image file

**Response:**
- Status: `200 OK` on success
- Content-Type: `application/json`
- Body: JSON object with detections, statistics, and annotated image

**Error Responses:**
- `400 Bad Request`: Invalid file or missing image
- `413 Payload Too Large`: File exceeds 10MB
- `500 Internal Server Error`: Server-side error

#### `GET /api/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "facebook/detr-resnet-50",
  "device": "cuda"
}
```

## ⚙️ Configuration

### Model Settings

Edit `model/config.py` to customize:

```python
# Confidence threshold (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.7

# NMS threshold for overlapping boxes
NMS_THRESHOLD = 0.5

# Model selection
MODEL_NAME = "facebook/detr-resnet-50"
# Alternative: "facebook/detr-resnet-101" (more accurate, slower)

# Device selection
DEVICE = "cuda"  # or "cpu"
```

### Upload Settings

```python
# Maximum image size (bytes)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
```

### Color Customization

Modify bounding box colors in `model/config.py`:

```python
CATEGORY_COLORS = {
    'vehicle': (59, 130, 246),      # Blue
    'pedestrian': (16, 185, 129),   # Green
    'animal': (249, 115, 22),       # Orange
    'traffic': (250, 204, 21),      # Yellow
    'other': (168, 85, 247)         # Purple
}
```

## 📁 Project Structure

```
TEDR/
├── model/                          # AI Model Package
│   ├── __init__.py                # Package initialization
│   ├── detr_detector.py           # DETR model wrapper
│   ├── config.py                  # Configuration settings
│   └── utils.py                   # Helper functions (NMS, visualization)
│
├── app/                           # Flask Application
│   ├── __init__.py               # App package initialization
│   ├── app.py                    # Flask routes and API
│   ├── static/                   # Static assets
│   │   ├── css/
│   │   │   └── style.css        # Custom styles
│   │   ├── js/
│   │   │   └── main.js          # Frontend logic
│   │   └── uploads/             # Temporary upload folder
│   │       └── .gitkeep
│   └── templates/                # HTML templates
│       └── index.html           # Main UI page
│
├── run.py                        # Application entry point
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🔍 Troubleshooting

### Common Issues

**1. Model Download Fails**
```
Error: Connection timeout while downloading model
```
**Solution**: Ensure stable internet connection. The model will be cached after first download.

**2. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution**: Use CPU mode by setting `DEVICE = "cpu"` in `model/config.py`, or use smaller images.

**3. Port Already in Use**
```
OSError: [Errno 98] Address already in use
```
**Solution**: Change the port in `run.py` or kill the process using port 5000:
```bash
# Linux/macOS
lsof -ti:5000 | xargs kill -9

# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**4. Import Errors**
```
ModuleNotFoundError: No module named 'transformers'
```
**Solution**: Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**5. Slow Detection on CPU**
**Solution**: This is expected. GPU acceleration is recommended for real-time performance. Consider using Google Colab or cloud services with GPU support.

### Performance Tips

- **Use GPU**: Detection is 10-50x faster on GPU
- **Resize Large Images**: Images larger than 2000px may be slow to process
- **Batch Processing**: For multiple images, use the batch API endpoint
- **Adjust Threshold**: Lower confidence threshold for more detections (may include false positives)

## 🚧 Future Improvements

### Planned Features
- [ ] Video stream processing (real-time webcam/CCTV)
- [ ] Fine-tuning on Indian road-specific dataset
- [ ] Object tracking across frames
- [ ] Speed estimation for vehicles
- [ ] Lane detection and violation alerts
- [ ] Multi-language support for UI
- [ ] Mobile application (iOS/Android)
- [ ] Cloud deployment with Docker
- [ ] Advanced analytics dashboard
- [ ] Export detection data (CSV, JSON)
- [ ] Comparison mode (before/after)
- [ ] Custom model training interface

### Model Improvements
- Fine-tune on Indian road-specific dataset
- Add detection for auto-rickshaws (currently detected as generic vehicles)
- Improve small object detection (distant vehicles/pedestrians)
- Add scene understanding (intersection, highway, residential)
- Weather condition adaptation (rain, fog, night)

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add comments for complex logic
- Update documentation for new features
- Test your changes thoroughly
- Keep commits focused and atomic

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 TEDR

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- **Facebook AI Research (FAIR)** for the DETR model
- **Hugging Face** for the Transformers library
- **COCO Dataset** for training data
- **Flask** team for the excellent web framework
- All contributors and users of TEDR

## 📞 Contact

For questions, suggestions, or issues:
- **GitHub Issues**: [Create an issue](https://github.com/VishnuNambiar0602/TEDR/issues)
- **Email**: [Contact maintainer]

---

<div align="center">

**Made with ❤️ for Indian Roads**

⭐ Star this repo if you find it useful! ⭐

</div>