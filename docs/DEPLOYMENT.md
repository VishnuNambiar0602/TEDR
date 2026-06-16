# AIR-DETR Deployment & Docker Guide

This guide details the containerization, orchestration, automation, and CI/CD pipelines for deploying **AIR-DETR (v1)**.

---

## 🐳 1. Docker Containerization

We provide a production-grade `Dockerfile` that configures a GPU-accelerated environment with CUDA, PyTorch, and all required libraries (including OpenCV and Ultralytics).

### Dockerfile (`Dockerfile` in project root)
Create a `Dockerfile` with the following content:

```dockerfile
# Use official NVIDIA CUDA development base image
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-opencv \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Create app directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies (plus PyTorch with CUDA 12.1 support, ultralytics, and safetensors)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir ultralytics safetensors psutil matplotlib pandas scikit-learn \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start command
CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🐙 2. Docker Compose Orchestration

Use `docker-compose.yml` to define the container configurations, environment variables, and GPU resource allocations.

### Docker Compose Configuration (`docker-compose.yml`)
```yaml
version: '3.8'

services:
  air-detr-server:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: air_detr_api
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

---

## 🛠️ 3. Makefile Automation

A `Makefile` is provided in the project root to simplify common engineering workflows.

### Makefile Configuration (`Makefile`)
```makefile
.PHONY: setup test benchmark run clean docker-build docker-up docker-down

# Install requirements
setup:
	pip install -r requirements.txt
	pip install ultralytics safetensors psutil matplotlib pandas scikit-learn

# Run all unit and integration tests
test:
	python -m unittest discover -s tests -p "test_*.py"

# Run the benchmark suite
benchmark:
	python benchmarks/benchmark_streaming.py

# Run FastAPI dev server
run:
	python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

# Clean temporary files, caches, and shards
clean:
	rm -rf temp/benchmark_shards temp/test_air_detr_shards temp/test_shards
	rm -rf __pycache__ air_detr/__pycache__ tests/__pycache__
	rm -rf .pytest_cache

# Build Docker image
docker-build:
	docker build -t air-detr:latest .

# Launch container stack
docker-up:
	docker-compose up -d

# Stop container stack
docker-down:
	docker-compose down
```

---

## 🚀 4. CI/CD GitHub Actions Pipeline

Automate linting, unit testing, and Docker builds on pull requests and commits to the `main` branch.

### GitHub Actions Workflow (`.github/workflows/ci.yml`)
```yaml
name: AIR-DETR CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout Code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-style: '3.10'

    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        pip install ultralytics safetensors psutil matplotlib pandas scikit-learn
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

    - name: Run Unit Tests
      run: |
        python -m unittest discover -s tests -p "test_*.py"

  docker-build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
    - name: Checkout Code
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Build Docker Image
      run: |
        docker build -t air-detr:latest .
```
