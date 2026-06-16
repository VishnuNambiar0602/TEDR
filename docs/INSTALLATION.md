# INSTALLATION & SETUP GUIDE

Complete step-by-step instructions for installing and running the Indian Road Object Detection application.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Options](#installation-options)
3. [GPU Setup (Recommended)](#gpu-setup-recommended)
4. [Verification](#verification)
5. [Running the Application](#running-the-application)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.13+
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum (16GB+ recommended)
- **Disk Space**: 10GB (5GB for model weights + 5GB for dependencies)
- **CPU**: Intel i7/Ryzen 5 or better

### Recommended Setup (for 30+ FPS inference)
- **GPU**: NVIDIA GPU (GTX 1080 Ti, RTX 3090, etc.)
- **CUDA Toolkit**: 11.8 or 12.1
- **cuDNN**: 8.1 or higher
- **RAM**: 16GB+
- **Disk Space**: 20GB

### Supported GPUs
- NVIDIA: All CUDA-capable GPUs (optimized for RTX series)
- Apple: Metal acceleration (slower)
- CPU: Works but slow (~0.5 FPS)

---

## Installation Options

### Option 1: Windows with GPU (Recommended)

#### Step 1: Install NVIDIA Drivers (Skip if already installed)

```bash
# Check your GPU
GPU_name_and_version  # Run in PowerShell

# Download drivers from: https://www.nvidia.com/Download/driverDetails.aspx
# Or use GeForce Experience if you have NVIDIA GeForce GPU
```

#### Step 2: Install Project

```powershell
# Open PowerShell in project directory
cd D:\Projects\DETR Object Detection

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

#### Step 3: Run the Application

```powershell
# Make sure venv is activated
venv\Scripts\activate

# Start the app
python app.py

# Open browser to: http://localhost:7860
```

---

### Option 2: Windows with CPU Only

#### Step 1: Install Project

```powershell
cd D:\Projects\DETR Object Detection

# Create virtual environment
python -m venv venv

# Activate
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch for CPU
pip install torch torchvision torchaudio

# Install other requirements
pip install -r requirements.txt

# Remove transformers if you want (optional)
pip uninstall transformers -y
```

#### Step 2: Configure for CPU

Edit `app.py` (around line 240):

```python
# Change this line:
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# To this:
device = 'cpu'

# Or change in model_utils.py around line 140:
self.device = 'cpu'  # Force CPU
```

#### Step 3: Run

```powershell
venv\Scripts\activate
python app.py
# Note: Inference will be slow (~0.5-1 FPS)
```

---

### Option 3: Linux with GPU

#### Step 1: Install NVIDIA Drivers (if not already)

```bash
# Check current driver
nvidia-smi

# Install drivers (Ubuntu)
sudo apt update
sudo apt install nvidia-driver-525

# Reboot if drivers were installed
sudo reboot
```

#### Step 2: Install Project

```bash
cd ~/Projects/DETR Object Detection

# Create virtual environment
python3.10 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

#### Step 3: Run

```bash
source venv/bin/activate
python app.py
# Open: http://localhost:7860
```

---

### Option 4: macOS

#### Step 1: Install Dependencies

```bash
# Using Homebrew
brew install python@3.10 git

# Create project directory
mkdir ~/Projects/DETR
cd ~/Projects/DETR

# Clone or copy project files
```

#### Step 2: Install Project

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch for macOS (Metal acceleration)
pip install torch torchvision torchaudio

# Install other requirements
pip install -r requirements.txt
```

#### Step 3: Run

```bash
source venv/bin/activate
python app.py

# Note: macOS uses Metal acceleration (slower than NVIDIA GPU)
```

---

### Option 5: Docker (Advanced)

Create `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv \
    git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir -r requirements.txt && \
    python3.10 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY . .

EXPOSE 7860

CMD ["python3.10", "app.py"]
```

Build and run:

```bash
docker build -t india-road-detection .
docker run --gpus all -p 7860:7860 india-road-detection
```

---

## GPU Setup (Recommended)

### Installing CUDA Toolkit (Windows)

1. Download from: https://developer.nvidia.com/cuda-downloads
2. Select:
   - OS: Windows 10/11
   - Architecture: x86_64
   - Version: 12.1
   - Installer Type: exe (network)
3. Run installer and complete setup
4. Verify installation:
   ```powershell
   nvcc --version
   ```

### Installing cuDNN (Windows)

1. Visit: https://developer.nvidia.com/cudnn
2. Download cuDNN for CUDA 12.1
3. Extract to CUDA installation directory:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
   ```
4. Verify paths in System Environment Variables

### Check GPU is Detected

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## Verification

### Check Installation Completed Successfully

```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run verification script
python -c "
import sys
import torch
import cv2
import gradio
import transformers
from ultralytics import YOLO

print('✓ Python:', sys.version.split()[0])
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ OpenCV:', cv2.__version__)
print('✓ Gradio:', gradio.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ Ultralytics YOLO: Installed')
print('\n✓ All dependencies installed successfully!')
"
```

### Test Import of Custom Modules

```bash
python -c "from model_utils import load_detector; print('✓ model_utils imported successfully')"
```

---

## Running the Application

### First Run (Downloads Model Weights)

```bash
# Activate virtual environment
venv\Scripts\activate

# Run application (first time takes 3-5 minutes)
python app.py

# You should see:
# [1/2] Initializing model...
#   Loading rtdetr-l model...
#   ✓ Model loaded (downloading ~300-800 MB)
# 
# [2/2] Starting Gradio web interface...
# 
# ✓ Application ready!
# Open the web interface: http://localhost:7860
```

### Subsequent Runs

```bash
# Just run normally (faster, model already cached)
python app.py
```

### Access the Application

1. **Local Computer**: Open browser to `http://localhost:7860`
2. **From Another Computer**: Use `http://<your-computer-ip>:7860`
   - Find your IP: 
     - Windows: `ipconfig` → IPv4 Address
     - Linux: `hostname -I`

---

## Troubleshooting Installation Issues

### Issue 1: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
# Ensure virtual environment is activated
venv\Scripts\activate

# Reinstall PyTorch
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: "CUDA Error: out of memory"

**Solution:**
```python
# In app.py, use smaller model:
DEFAULT_MODEL = 'rtdetr-s'  # Instead of 'rtdetr-l'

# Or force CPU:
# In model_utils.py, line 140:
self.device = 'cpu'
```

### Issue 3: "FileNotFoundError: Could not find CUDA"

**Solution:**
```bash
# Reinstall PyTorch for CPU (remove CUDA requirement)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio

# Then force CPU in code
```

### Issue 4: "pip: command not found" or "python: command not found"

**Solution:**
- Ensure Python is in PATH
- Use full path: `C:\Python310\Scripts\pip install -r requirements.txt`
- Or reinstall Python with "Add Python to PATH" checked

### Issue 5: Port 7860 Already in Use

**Solution:**
```bash
# Kill process on port 7860:
# Windows:
netstat -ano | findstr :7860
taskkill /PID <PID> /F

# Or change port in app.py:
interface.launch(server_port=7861)

# Or in config.ini:
server_port = 7861
```

### Issue 6: Slow Installation (Dependencies Take Too Long)

**Solution:**
```bash
# Use faster index
pip install -r requirements.txt -i https://pypi.tsinghua.tsinghua.edu.cn/simple

# Or use PyTorch's pre-built wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Advanced Configuration

### Using Different CUDA Version

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision torchaudio

# AMD ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### Installing Specific PyTorch Version

```bash
# PyTorch 2.0.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### Create Startup Script

**Windows** (`run.bat`):
```batch
@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python app.py
pause
```

**Linux/Mac** (`run.sh`):
```bash
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python app.py
```

---

## Next Steps After Installation

1. **Test the App**: Open `http://localhost:7860` and upload a test image
2. **Read README.md**: Detailed documentation
3. **Try Examples**: Run `python example_usage.py`
4. **Train Custom Model**: Use `python train.py --data dataset.yaml`
5. **Deploy**: Use Gradio, FastAPI, or Docker for production

---

## Getting Help

If you encounter issues:

1. Check this file again
2. Read README.md
3. Check Ultralytics docs: https://docs.ultralytics.com/
4. Check Gradio docs: https://gradio.app/
5. Visit GitHub Issues (if applicable)

---

**Happy detecting! 🚀**
