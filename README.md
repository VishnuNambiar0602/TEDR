# 🚗 TEDR - Transformer-based Object Detection for Indian Roads

AI-powered object detection system designed specifically for Indian road scenarios using DETR (DEtection TRansformer).

## ✨ Features

- **DETR Model Integration**: State-of-the-art transformer-based object detection using Facebook's DETR
- **Indian Road Focus**: Optimized for detecting vehicles, pedestrians, animals (cows), and traffic elements common on Indian roads
- **FastAPI Backend**: High-performance REST API with mobile-optimized responses
- **Modern Web UI**: Clean, responsive interface with drag-and-drop image upload
- **Real-time Detection**: Fast inference with bounding box visualization
- **Training Pipeline**: Ready-to-use training scripts for custom datasets
- **91 Object Classes**: Supports all COCO dataset classes including cars, motorcycles, trucks, buses, persons, bicycles, cows, and more

## 🎯 Detected Objects

Primary focus on Indian road scenarios:
- **Vehicles**: Auto rickshaw*, cars, motorcycles, trucks, buses, bicycles
- **Pedestrians**: People and crowds
- **Animals**: Cows, dogs, and other animals
- **Infrastructure**: Traffic lights, stop signs, parking meters
- **Plus 80+ additional COCO classes**

*Auto rickshaw detection available through fine-tuning (see Training section)

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster inference)
- 4GB+ RAM
- Modern web browser

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/VishnuNambiar0602/TEDR.git
cd TEDR

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
python backend/main.py
```

The API will start on `http://localhost:8000`

### 3. Access the Web UI

Open your browser and navigate to:
```
http://localhost:8000/static/index.html
```

Or use the API directly:
```
http://localhost:8000/docs  # Interactive API documentation
```

## 📚 API Documentation

### Endpoints

#### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model": "facebook/detr-resnet-50",
  "device": "cuda",
  "confidence_threshold": 0.7
}
```

#### 2. Object Detection
```bash
POST /detect
Content-Type: multipart/form-data

file: <image file>
```

Response:
```json
{
  "detections": [
    {
      "label": "car",
      "label_id": 3,
      "confidence": 0.95,
      "bbox": [100, 150, 400, 350]
    }
  ],
  "num_detections": 1,
  "image_size": [800, 600],
  "processing_time": 0.342
}
```

#### 3. Model Info
```bash
GET /models/info
```

### Using the API with cURL

```bash
# Detect objects in an image
curl -X POST "http://localhost:8000/detect" \
  -F "file=@/path/to/image.jpg"
```

### Using the API with Python

```python
import requests

# Upload image for detection
with open('image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/detect', files=files)
    results = response.json()

print(f"Detected {results['num_detections']} objects")
for detection in results['detections']:
    print(f"- {detection['label']}: {detection['confidence']:.2%}")
```

## 🎓 Training Custom Models

### Dataset Format

TEDR uses COCO format for training. Your dataset should have:

```
data/datasets/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

### COCO Annotation Format

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img1.jpg",
      "width": 800,
      "height": 600
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 250],
      "area": 50000
    }
  ],
  "categories": [
    {"id": 1, "name": "auto_rickshaw"},
    {"id": 2, "name": "car"}
  ]
}
```

### Training Script

```bash
# Edit config.yaml to set your dataset paths and parameters
python models/train.py
```

### Configuration

Edit `config.yaml` to customize:
- Model architecture and pretrained weights
- Training hyperparameters (batch size, learning rate, epochs)
- Dataset paths
- Device (CPU/GPU)
- Confidence thresholds

## 🏗️ Project Structure

```
TEDR/
├── backend/              # FastAPI backend
│   ├── main.py          # API server
│   ├── config.py        # Configuration management
│   └── inference.py     # Detection logic
├── frontend/            # Web UI
│   ├── index.html       # Main page
│   ├── styles.css       # Styling
│   └── app.js           # Frontend logic
├── models/              # Model definitions
│   ├── detr_model.py    # DETR wrapper
│   └── train.py         # Training pipeline
├── utils/               # Utility functions
│   ├── preprocessing.py # Image preprocessing
│   └── visualization.py # Detection visualization
├── data/                # Data directory
│   ├── sample_images/   # Test images
│   └── datasets/        # Training datasets
├── config.yaml          # Main configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 Configuration

### Model Settings

```yaml
model:
  name: "facebook/detr-resnet-50"
  confidence_threshold: 0.7
  image_size: 800
  device: "cuda"  # or "cpu"
```

### API Settings

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "*"
```

### Training Settings

```yaml
training:
  batch_size: 4
  num_epochs: 50
  learning_rate: 0.0001
  weight_decay: 0.0001
```

## 🎨 Web UI Features

- **Drag & Drop**: Simply drag images onto the upload area
- **Real-time Visualization**: See detections with color-coded bounding boxes
- **Detection List**: Detailed list of all detected objects with confidence scores
- **Statistics**: View object count, processing time, and image dimensions
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🔬 Model Architecture

TEDR uses the DETR (DEtection TRansformer) architecture:

1. **Backbone**: ResNet-50 CNN extracts image features
2. **Transformer Encoder**: Processes image features
3. **Transformer Decoder**: Generates object queries
4. **Prediction Heads**: Predicts bounding boxes and class labels

Key advantages:
- End-to-end training without NMS
- Global reasoning with self-attention
- Handles complex scenes well
- State-of-the-art accuracy

## 📊 Performance

- **Inference Speed**: ~0.3-0.5s per image (GPU) / ~2-3s (CPU)
- **Accuracy**: mAP 42+ on COCO val2017
- **Input Size**: 800x800 (configurable)
- **Confidence Threshold**: 0.7 (configurable)

## 🛠️ Development

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-cov

# Run tests (when available)
pytest tests/
```

### Code Style

```bash
# Format code
pip install black
black .

# Lint code
pip install flake8
flake8 .
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `config.yaml`
- Use smaller image sizes
- Switch to CPU mode

### Slow Inference
- Ensure CUDA is installed and configured
- Check that `device: "cuda"` in config.yaml
- Update PyTorch to latest version

### Model Download Issues
- Check internet connection
- Models are downloaded from HuggingFace Hub
- Use VPN if HuggingFace is blocked

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **DETR**: End-to-End Object Detection with Transformers (Facebook AI Research)
- **Hugging Face**: Transformers library
- **COCO Dataset**: Common Objects in Context

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review API docs at `/docs` endpoint

## 🗺️ Roadmap

Future enhancements:
- [ ] Video detection support
- [ ] Model quantization for mobile deployment
- [x] Docker containerization ✓
- [ ] Detection history and analytics
- [ ] Batch processing API
- [ ] Custom class training UI
- [ ] Multi-language support
- [ ] Mobile app integration

---

**Built with ❤️ for safer Indian roads**
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

### Production Deployment

For production environments, use a WSGI server like Gunicorn instead of Flask's built-in server:

**Using Gunicorn (included in requirements.txt):**
```bash
# Single worker
gunicorn -w 1 -b 0.0.0.0:5000 run:app

# Multiple workers (recommended for CPU-bound tasks)
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 run:app

# With access logging
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 --access-logfile - run:app
```

**Important Production Notes:**
- Debug mode is disabled by default in `run.py` for security
- Set `--timeout` high enough for model inference (default 30s may be too short)
- Use environment variables for sensitive configuration
- Consider using nginx as a reverse proxy
- Implement rate limiting for API endpoints
- Add authentication if deploying publicly

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
