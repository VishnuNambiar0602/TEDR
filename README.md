# AIR-DETR: Adaptive Inference RT-DETR

AIR-DETR (Adaptive Inference Real-Time DEtection TRansformer) is a high-performance computer vision systems framework designed to enable the execution of RT-DETR models under extremely tight GPU memory constraints. By utilizing **AirLLM-style layer-by-layer weight streaming** and **multi-level quantization**, AIR-DETR can run models significantly larger than the available GPU VRAM.

---

## 🚀 Key Features

* **AirLLM-inspired Weight Streaming**: Splits the 29 sequential layers of RT-DETR-L on disk. Only the active layer resides in GPU VRAM during execution, immediately evicting layers and reclaiming memory.
* **Double-Buffered Asynchronous Prefetching**: Overlaps disk I/O, CPU-to-GPU PCIe transfers, and GPU execution using pinned host memory and non-blocking copies to hide transfer latencies.
* **Post-Training Quantization (PTQ)**: Supports INT8 and INT4 channel-wise symmetric quantization.
* **INT4 Bit-Packing**: Compresses two 4-bit values into one `uint8` byte for a $4\times$ reduction in weight file size, unpacking on-the-fly directly on the GPU.
* **VRAM Manager & Guard**: Tracks and enforces active memory limits, preventing Out-Of-Memory (OOM) situations on edge hardware.
* **Zero-Code Integration**: Implemented as an in-place monkey-patch over the standard Ultralytics RT-DETR validation/inference pipelines, enabling immediate support without codebase rewrites.

---

## 📂 Project Structure

```
D:\Projects\TEDR\
├── air_detr/                  # Core library
│   ├── __init__.py            # Package entrypoint
│   ├── model.py               # StreamingRTDETR wrapper & forward patch
│   ├── quantization.py        # INT8/INT4 PTQ engine & bit-packing
│   ├── scheduler.py           # SSD -> RAM -> GPU prefetching queue
│   ├── shard_manager.py       # Layer weight sharder (safetensors/pt)
│   └── vram_manager.py        # GPU memory stats and limit checker
├── benchmarks/                # Performance evaluation scripts
│   └── benchmark_streaming.py # Suite running baseline vs streaming modes
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # Technical design details
│   ├── BENCHMARKS.md          # Measurement methodology
│   ├── DEPLOYMENT.md          # Docker & production guides
│   ├── FORMULAS.md            # Mathematical equations & derivations
│   └── RESEARCH_REPORT.md     # Findings & results discussion
├── tests/                     # Test suite
│   ├── test_quantization.py   # Unit tests for PTQ and packing
│   ├── test_shard_manager.py  # Unit tests for sharding
│   └── test_streaming.py      # Integration tests for streaming inference
├── Dockerfile                 # Container image configuration
├── docker-compose.yml         # Container stack orchestration
├── Makefile                   # Workflows automation utility
└── requirements.txt           # Python dependencies
```

---

## ⚙️ Installation

1. **Clone the Repository**:
   ```bash
   git clone <repo-url>
   cd TEDR
   ```

2. **Set up Environment**:
   Using the automated Makefile target:
   ```bash
   make setup
   ```
   Or manually:
   ```bash
   pip install -r requirements.txt
   pip install ultralytics safetensors psutil matplotlib pandas scikit-learn
   ```

---

## 🏃 Quick Start

To wrap an existing YOLO/RT-DETR model and enable streaming, use `StreamingRTDETR`:

```python
from ultralytics import YOLO
from air_detr import StreamingRTDETR

# 1. Load standard model
model = YOLO("rtdetr-l.pt")

# 2. Wrap it with AIR-DETR (offloads weights and patches forward pass)
# Supports: None (FP32), 'int8', or 'int4' quantization
stream_model = StreamingRTDETR(
    yolo_model=model,
    shard_dir="temp/air_detr_shards",
    quantization="int8",
    format="safetensors"
)

# 3. Run inference normally!
# Standard APIs are preserved (will stream and dequantize under the hood)
results = model("test_data/traffic_images/traffic_02.jpg")

# 4. Save visualization
results[0].save("output.jpg")

# 5. Restore original model when done
stream_model.restore()
```

---

## 🧪 Running Tests & Benchmarks

* **Run all tests**:
  ```bash
  make test
  ```

* **Run benchmark suite**:
  This runs baseline vs streaming modes on the validation dataset and generates `logs1.md`, plots, and LaTeX tables:
  ```bash
  make benchmark
  ```

---

## 🐳 Docker Deployment

To build and run the FastAPI server inside a GPU-accelerated Docker container:

```bash
# Build the image
make docker-build

# Start the service
make docker-up

# Stop the service
make docker-down
```
