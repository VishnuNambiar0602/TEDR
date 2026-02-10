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