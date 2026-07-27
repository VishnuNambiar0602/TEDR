# AIR-DETR Model Evaluation & Performance Report: rtdetr-l.pt
- **Current Local Time**: 2026-07-27 13:55:19
- **Model**: `rtdetr-l.pt`
- **Evaluation Dataset**: `DriveIndia Public Dataset` (500 validation images)
- **Inference Device**: NVIDIA GeForce RTX 4050 Laptop GPU

## 📊 Performance Comparison Table
| Mode | mAP50 | mAP50-95 | Throughput (FPS) | Latency (ms) | Peak VRAM (MB) | VRAM Reduction | Accuracy Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | 51.30% | 30.58% | 3.07 | 325.5 | 1609.6 | 0.0% | 0.00% |
| **STREAMING FP32** | 51.30% | 30.58% | 3.56 | 281.0 | 727.6 | 54.8% | 0.00% |

## 🎯 Analysis of Objectives & Success Criteria
1. **VRAM Reduction >= 50%**:
   - FP32 Streaming VRAM Reduction: **54.8%**
   - *Status*: **SUCCESS** (VRAM requirements decreased from 1609.6 MB to 727.6 MB)

2. **Streaming Overhead <= 25%**:
   - Throughput drop (Baseline vs Streaming FP32): **-15.8%**
   - *Status*: **SUCCESS**

## 💻 Software & Hardware Environment
- **Python Version**: `3.14.0`
- **PyTorch Version**: `2.11.0.dev20260119+cu126`
- **CUDA Available**: `True`
- **CUDA Version**: `12.6`
- **GPU Model**: `NVIDIA GeForce RTX 4050 Laptop GPU`