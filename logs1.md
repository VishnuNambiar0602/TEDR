# Model Evaluation & Hardware/Software Performance Report

## 📊 Key Evaluation Parameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model** | `RT-DETR-l` | Real-Time DEtection TRansformer (Large) |
| **mAP50** | `78.2%` | Mean Average Precision on India Driving Dataset (IDD) |
| **FPS** | `71.24` | Frames Per Second processed (system-wide throughput) |
| **VRAM** | `1050.0 MiB` | Peak Video RAM allocated during inference |

## 📝 Execution Details
- **Current Local Time**: 2026-07-27 13:46:41 India Standard Time
- **Test Video File**: `D:\Projects\TEDR\test_video.mp4`
- **Video Details**: 359 total frames, 24.0 FPS, 720x1280 resolution
- **Inference Device**: NVIDIA GeForce RTX 4050 Laptop GPU
- **Execution Time**: 5.039 seconds

## 💻 Software Environment
- **Python Version**: `3.14.0`
- **PyTorch Version**: `2.11.0.dev20260119+cu126`
- **CUDA Available**: `True` (CUDA version: `12.6`)

### Git Repository Status
```
On branch main
Your branch is ahead of 'origin/main' by 1 commit.
  (use "git push" to publish your local commits)

nothing to commit, working tree clean
```

### Recent Git Commits
```
ff14b8b More updates
ac7cf9e Update UI and app configurations
f2e6893 Implement weight streaming, quantization modes, and run benchmarks
2d2f56c Merge pull request #2 from VishnuNambiar0602/copilot/implement-detr-object-detection
241960e Merge branch 'main' into copilot/implement-detr-object-detection
```

## 🔋 Hardware & Resource Usage Statistics
| Metric | Min | Average | Max | Unit |
| :--- | :---: | :---: | :---: | :---: |
| **System CPU Usage** | 0.0% | 20.5% | 39.4% | % |
| **System RAM Usage** | 8.12 | 8.88 | 9.28 | GB |
| **Python Process CPU Usage** | 0.0% | 134.1% | 306.3% | % |
| **Python Process RAM (RSS)** | 664.3 | 1416.2 | 1817.8 | MB |
| **GPU Core Utilization** | 1.0% | 4.8% | 14.0% | % |
| **VRAM Memory Usage** | 631.0 | 945.2 | 1050.0 | MiB |
| **GPU Core Temperature** | 42.0°C | 44.2°C | 46.0°C | °C |
| **CPU Temperature** | N/A (Access Denied / Not Supported on Windows without Admin) | - | - | - |

## 📊 Model Inference Results
- **Success Status**: `True`
- **Analyzed Frames**: 359 (Frame Skip: 1)
- **Average Vehicle Count per Frame**: `89.192` vehicles
- **Average Occupancy Ratio**: `8.90%` of frame area
- **Congestion Frame Classification Distribution**:
  - **LOW**: 359 frames
  - **MEDIUM**: 0 frames
  - **HIGH**: 0 frames
- **Processed Video Output Path**: `D:\Projects\TEDR\temp\processed\processed_eval_video.mp4`