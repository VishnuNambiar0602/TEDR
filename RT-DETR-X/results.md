# AIR-DETR Model Evaluation & Performance Report: rtdetr-x.pt
- **Current Local Time**: 2026-06-16 21:49:36
- **Model**: `rtdetr-x.pt`
- **Evaluation Dataset**: `COCO Auto Rickshaw` (40 validation images)
- **Inference Device**: NVIDIA GeForce RTX 4050 Laptop GPU

## 📊 Performance Comparison Table
| Mode | mAP50 | mAP50-95 | Throughput (FPS) | Latency (ms) | Peak VRAM (MB) | VRAM Reduction | Accuracy Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | 13.79% | 12.41% | 3.47 | 287.9 | 2338.2 | 0.0% | 0.00% |
| **STREAMING FP32** | 13.79% | 12.41% | 4.07 | 245.6 | 826.4 | 64.7% | 0.00% |
| **STREAMING INT8** | 21.69% | 11.23% | 3.45 | 290.0 | 826.6 | 64.6% | -7.90% |
| **STREAMING INT4** | 36.07% | 25.07% | 3.42 | 292.3 | 826.6 | 64.6% | -22.28% |
| **CALIBRATED INT8** | 21.69% | 11.23% | 3.74 | 267.5 | 826.6 | 64.6% | -7.90% |
| **CALIBRATED INT4** | 17.88% | 1.79% | 3.81 | 262.6 | 826.6 | 64.6% | -4.10% |
| **AWQ** | 13.79% | 2.76% | 3.27 | 305.4 | 826.6 | 64.6% | 0.00% |
| **GPTQ** | 36.58% | 22.60% | 4.14 | 241.4 | 826.6 | 64.6% | -22.79% |

## 🎯 Analysis of Objectives & Success Criteria
1. **VRAM Reduction >= 70%**:
   - FP32 Streaming VRAM Reduction: **64.7%**
   - INT8 Streaming VRAM Reduction: **64.6%**
   - INT4 Streaming VRAM Reduction: **64.6%**
   - *Status*: **FAILED** (VRAM requirements decreased from 2338.2 MB to 826.6 MB)

2. **Streaming Overhead <= 25%**:
   - Throughput drop (Baseline vs Streaming FP32): **-17.2%**
   - *Status*: **SUCCESS**

3. **INT8 Accuracy Drop < 1%**:
   - Accuracy drop: **-7.90%** (Baseline 13.79% vs INT8 21.69%)
   - *Status*: **SUCCESS**

4. **INT4 Accuracy Drop < 3%**:
   - Accuracy drop: **-22.28%** (Baseline 13.79% vs INT4 36.07%)
   - *Status*: **SUCCESS**

## 💻 Software & Hardware Environment
- **Python Version**: `3.14.0`
- **PyTorch Version**: `2.11.0.dev20260119+cu126`
- **CUDA Available**: `True`
- **CUDA Version**: `12.6`
- **GPU Model**: `NVIDIA GeForce RTX 4050 Laptop GPU`