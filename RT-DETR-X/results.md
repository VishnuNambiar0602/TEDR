# AIR-DETR Model Evaluation & Performance Report: rtdetr-x.pt
- **Current Local Time**: 2026-06-16 21:49:36
- **Model**: `rtdetr-x.pt`
- **Evaluation Dataset**: `DriveIndia Public Dataset` (500 validation images)
- **Inference Device**: NVIDIA GeForce RTX 4050 Laptop GPU

## 📊 Performance Comparison Table
| Mode | mAP50 | mAP50-95 | Throughput (FPS) | Latency (ms) | Peak VRAM (MB) | VRAM Reduction | Accuracy Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | 13.79% | 12.41% | 3.47 | 287.9 | 2338.2 | 0.0% | 0.00% |
| **STREAMING FP32** | 13.79% | 12.41% | 4.07 | 245.6 | 826.4 | 64.7% | 0.00% |

## 🎯 Analysis of Objectives & Success Criteria
1. **VRAM Reduction >= 50%**:
   - FP32 Streaming VRAM Reduction: **64.7%**
   - *Status*: **SUCCESS** (VRAM requirements decreased from 2338.2 MB to 826.4 MB)

2. **Streaming Overhead <= 25%**:
   - Throughput drop (Baseline vs Streaming FP32): **-17.2%**
   - *Status*: **SUCCESS**

## 💻 Software & Hardware Environment
- **Python Version**: `3.14.0`
- **PyTorch Version**: `2.11.0.dev20260119+cu126`
- **CUDA Available**: `True`
- **CUDA Version**: `12.6`
- **GPU Model**: `NVIDIA GeForce RTX 4050 Laptop GPU`