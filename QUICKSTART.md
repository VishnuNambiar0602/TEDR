# 🚀 Quick Start Guide

Get TEDR up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- 4GB RAM (8GB recommended)
- Optional: CUDA-capable GPU for faster inference

## Installation

### Option 1: Automated Setup (Recommended)

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

### Option 2: Manual Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Docker

```bash
docker-compose up -d
```

## Running the API

```bash
# Activate virtual environment if not already active
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the server
python backend/main.py
```

The API will be available at:
- Web UI: http://localhost:8000/static/index.html
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## First Detection

### Using the Web UI

1. Open http://localhost:8000/static/index.html
2. Drag and drop an image or click to browse
3. View detection results with bounding boxes

### Using the API

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@your_image.jpg"
```

### Using Python

```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect',
        files={'file': f}
    )
    print(response.json())
```

### Using the Command Line Tool

```bash
python example_detect.py path/to/image.jpg
```

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  confidence_threshold: 0.7  # Adjust detection threshold
  image_size: 800           # Input image size
  device: "cuda"            # or "cpu"
```

## Troubleshooting

### Server won't start
- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip list`

### Slow inference
- Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce image_size in config.yaml
- Use GPU if available

### Out of memory
- Reduce batch_size in config.yaml
- Switch to CPU mode
- Close other applications

## Next Steps

- Check the main [README.md](README.md) for detailed documentation
- Explore the API at http://localhost:8000/docs
- Test with different images
- Train custom models (see Training section in README.md)

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Review API documentation at `/docs`
- Open an issue on GitHub for bugs

---

Happy detecting! 🚗🛺🚙
