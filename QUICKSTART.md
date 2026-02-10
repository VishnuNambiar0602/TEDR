# 🚀 Quick Start Guide

Get TEDR up and running in 5 minutes!
# Quick Start Guide

This guide will help you get TEDR up and running in minutes!

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
- pip (Python package installer)
- (Optional) CUDA-capable GPU for faster inference

## Installation Steps

### 1. Clone the Repository (if you haven't already)

```bash
git clone https://github.com/VishnuNambiar0602/TEDR.git
cd TEDR
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This will download approximately 2-3GB of packages including PyTorch and Transformers. Make sure you have a stable internet connection.

### 4. Validate Installation

```bash
python validate_setup.py
```

You should see all green checkmarks (✓) indicating the system is ready.

### 5. Run the Application

```bash
python run.py
```

You should see output like:
```
TEDR - Transformer-based Object Detection for Indian Roads

Starting server...
Open your browser and navigate to: http://localhost:5000

Press CTRL+C to stop the server
```

### 6. Open in Browser

Navigate to: **http://localhost:5000**

## First-Time Model Download

When you run object detection for the first time, DETR will automatically download the pre-trained model (~160MB) from Hugging Face. This only happens once and the model will be cached locally.

## Testing the System

1. **Upload an image** of an Indian road scene (or any image with vehicles, people, or animals)
2. Click **"Detect Objects"**
3. Wait for processing (usually 2-10 seconds depending on your hardware)
4. View the annotated results with bounding boxes
5. Download the result image if desired

## Common Issues

### Port 5000 Already in Use

If you see an error about port 5000 being in use, you can either:

1. Kill the process using port 5000
2. Edit `run.py` and change the port number:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
   ```

### Slow Performance on CPU

If you don't have a GPU, detection will be slower (10-30 seconds per image). This is normal. For faster performance:
- Use a machine with NVIDIA GPU
- Use Google Colab with free GPU
- Reduce image resolution before uploading

### Import Errors

Make sure your virtual environment is activated:
```bash
# Check if venv is activated - you should see (venv) in your prompt
# If not, activate it:
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the API endpoints for integration
- Customize detection thresholds in `model/config.py`
- Try different types of images

## Support

If you encounter any issues:
1. Check the [Troubleshooting](README.md#troubleshooting) section in README.md
2. Open an issue on GitHub
3. Make sure all dependencies are correctly installed

---

**Happy Detecting! 🚗🚦**
