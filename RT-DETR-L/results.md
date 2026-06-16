# AIR-DETR Model Evaluation & Performance Report: rtdetr-l.pt
- **Current Local Time**: 2026-06-16 21:43:10
- **Model**: `rtdetr-l.pt`
- **Evaluation Dataset**: `COCO Auto Rickshaw` (40 validation images)
- **Inference Device**: NVIDIA GeForce RTX 4050 Laptop GPU

## 📊 Performance Comparison Table
| Mode | mAP50 | mAP50-95 | Throughput (FPS) | Latency (ms) | Peak VRAM (MB) | VRAM Reduction | Accuracy Drop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BASELINE** | 51.30% | 30.58% | 3.52 | 284.4 | 1609.6 | 0.0% | 0.00% |
| **STREAMING FP32** | 51.30% | 30.58% | 4.21 | 237.4 | 727.6 | 54.8% | 0.00% |
| **STREAMING INT8** | 27.22% | 17.74% | 3.81 | 262.4 | 727.7 | 54.8% | 24.07% |
| **STREAMING INT4** | 0.00% | 0.00% | 3.69 | 270.8 | 727.7 | 54.8% | 51.30% |
| **CALIBRATED INT8** | 27.22% | 16.90% | 4.01 | 249.1 | 727.7 | 54.8% | 24.07% |
| **CALIBRATED INT4** | 0.00% | 0.00% | 3.70 | 269.9 | 727.7 | 54.8% | 51.30% |
| **AWQ** | 0.00% | 0.00% | 3.60 | 277.4 | 727.7 | 54.8% | 51.30% |
| **GPTQ** | 26.07% | 23.47% | 4.06 | 246.4 | 727.7 | 54.8% | 25.22% |

## 🎯 Analysis of Objectives & Success Criteria
1. **VRAM Reduction >= 70%**:
   - FP32 Streaming VRAM Reduction: **54.8%**
   - INT8 Streaming VRAM Reduction: **54.8%**
   - INT4 Streaming VRAM Reduction: **54.8%**
   - *Status*: **FAILED** (VRAM requirements decreased from 1609.6 MB to 727.7 MB)

2. **Streaming Overhead <= 25%**:
   - Throughput drop (Baseline vs Streaming FP32): **-19.8%**
   - *Status*: **SUCCESS**

3. **INT8 Accuracy Drop < 1%**:
   - Accuracy drop: **24.07%** (Baseline 51.30% vs INT8 27.22%)
   - *Status*: **FAILED**

4. **INT4 Accuracy Drop < 3%**:
   - Accuracy drop: **51.30%** (Baseline 51.30% vs INT4 0.00%)
   - *Status*: **FAILED**

## 💻 Software & Hardware Environment
- **Python Version**: `3.14.0`
- **PyTorch Version**: `2.11.0.dev20260119+cu126`
- **CUDA Available**: `True`
- **CUDA Version**: `12.6`
- **GPU Model**: `NVIDIA GeForce RTX 4050 Laptop GPU`