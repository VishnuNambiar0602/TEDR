# pure_RT-DETR: PekingU/lyuwenyu Architecture Style Streaming & Quantization

`pure_RT-DETR` is a native PyTorch implementation of RT-DETR structured exactly as the original PekingU/lyuwenyu three-component architecture: **Backbone**, **Hybrid Encoder**, and **Transformer Decoder**. 

It uses layer-by-layer weight streaming (AIRLLM-style) and multi-level quantization (PTQ, AWQ, GPTQ) to execute inference under strict VRAM bounds.

---

## 📂 Subdirectory Structure

All code and utilities are fully self-contained inside the `pure_RT-DETR` directory:
```
pure_RT-DETR/
├── __init__.py          # Entrypoint exporting RTDETR & QuantizationManager
├── model.py             # Backbone, Encoder, Decoder component classes & RTDETR module
├── quantization.py      # QuantizationManager (INT8, INT4, AWQ, GPTQ)
├── scheduler.py         # Asynchronous SSD -> RAM -> GPU prefetch queue & thread
├── shard_manager.py     # Layer-wise model splitter and disk-saving utility
├── vram_manager.py      # VRAM memory tracking and limit guard
├── calibration.py       # Hook-based calibration engine
└── test_pure_rtdetr.py  # Unit & integration tests verifying execution & accuracy
```

---

## 🚀 Quick Start

To run streaming inference using the original three-component structure, import `RTDETR` from `pure_RT-DETR`:

```python
import cv2
import torch
import importlib

# 1. Load pure_RT-DETR dynamically (due to hyphen)
pure_RT_DETR = importlib.import_module("pure_RT-DETR")
RTDETR = pure_RT_DETR.RTDETR

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Initialize the three-component streaming module
model = RTDETR(
    model_path="rtdetr-l.pt",
    quantization=None,  # FP32 streaming
    shard_dir="temp/pure_rtdetr_shards",
    format="pt",
    vram_limit_mb=1000.0
)
model.eval()

# 3. Preprocess your image to [1, 3, 640, 640]
img = cv2.imread("test_data/traffic_images/traffic_02.jpg")
img_resized = cv2.resize(img, (640, 640))
img_rgb = img_resized[:, :, ::-1].copy()

# Query model dtype (FP16 or FP32)
dtype = next(model.original_model.parameters()).dtype
x = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
x = (x.float() / 255.0).to(dtype)

# 4. Run inference
with torch.no_grad():
    predictions = model(x)
    # Output shape: [1, 300, 84]

# 5. Restore original model parameters
model.restore()
```

---

## 🏗️ Architecture Design Details

The 29 sequential layers of the RT-DETR model loaded from `rtdetr-l.pt` are mapped and grouped as follows:

* **`self.backbone`**: Represents the **HGNetv2** backbone (layers 0 to 9). Takes the input image tensor and extracts multi-scale intermediate features.
* **`self.encoder`**: Represents the **Hybrid Encoder** with **AIFI** and **CCFM** modules (layers 10 to 27). Fuses backbone features and passes the refined multi-scale feature maps.
* **`self.decoder`**: Represents the **RTDETRTransformer** decoder (layer 28). Takes the outputs of layers 21, 24, and 27 to generate final predictions.
