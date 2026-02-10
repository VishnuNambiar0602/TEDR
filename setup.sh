#!/bin/bash

# TEDR System Installation and Setup Script

echo "========================================="
echo "TEDR - Object Detection System Setup"
echo "========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "To start the API server:"
echo "  source venv/bin/activate"
echo "  python backend/main.py"
echo ""
echo "Then open your browser to:"
echo "  http://localhost:8000/static/index.html"
echo ""
echo "For API documentation:"
echo "  http://localhost:8000/docs"
echo ""
