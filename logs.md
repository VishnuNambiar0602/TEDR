# Model Evaluation & Hardware/Software Performance Report

## 📊 Key Evaluation Parameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model** | `RT-DETR-l` | Real-Time DEtection TRansformer (Large) |
| **mAP50** | `78.2%` | Mean Average Precision on India Driving Dataset (IDD) |
| **FPS** | `6.00` | Frames Per Second processed (system-wide throughput) |
| **VRAM** | `1406.0 MiB` | Peak Video RAM allocated during inference |

## 📝 Execution Details
- **Current Local Time**: 2026-06-12 15:44:59 India Standard Time
- **Test Video File**: `C:\Users\vishn\Downloads\videoplayback (1).mp4`
- **Video Details**: 359 total frames, 24.0 FPS, 720x1280 resolution
- **Inference Device**: NVIDIA GeForce RTX 4050 Laptop GPU
- **Execution Time**: 59.841 seconds

## 💻 Software Environment
- **Python Version**: `3.14.0`
- **PyTorch Version**: `2.11.0.dev20260119+cu126`
- **CUDA Available**: `True` (CUDA version: `12.6`)

### Git Repository Status
```
On branch main
Your branch is ahead of 'origin/main' by 1 commit.
  (use "git push" to publish your local commits)

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   test_video.mp4

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	logs.md
	scripts/evaluate_custom_video.py
	scripts/monitor_performance.py

no changes added to commit (use "git add" and/or "git commit -a")
```

### Recent Git Commits
```
7677b06 Migrate transformer-based object detection system with video processing capabilities
2d2f56c Merge pull request #2 from VishnuNambiar0602/copilot/implement-detr-object-detection
241960e Merge branch 'main' into copilot/implement-detr-object-detection
02d8f66 Merge pull request #1 from VishnuNambiar0602/copilot/create-detr-object-detection
2049820 Add final validation checklist - all requirements complete
```

## 🔋 Hardware & Resource Usage Statistics
| Metric | Min | Average | Max | Unit |
| :--- | :---: | :---: | :---: | :---: |
| **System CPU Usage** | 0.0% | 80.3% | 95.2% | % |
| **System RAM Usage** | 11.82 | 12.50 | 12.67 | GB |
| **Python Process CPU Usage** | 0.0% | 825.8% | 1002.2% | % |
| **Python Process RAM (RSS)** | 782.5 | 1566.5 | 1601.7 | MB |
| **GPU Core Utilization** | 0.0% | 11.4% | 40.0% | % |
| **VRAM Memory Usage** | 1220.0 | 1384.1 | 1406.0 | MiB |
| **GPU Core Temperature** | 46.0°C | 48.4°C | 50.0°C | °C |
| **CPU Temperature** | N/A (Access Denied / Not Supported on Windows without Admin) | - | - | - |

## 📊 Model Inference Results
- **Success Status**: `True`
- **Analyzed Frames**: 359 (Frame Skip: 1)
- **Average Vehicle Count per Frame**: `22.253` vehicles
- **Average Occupancy Ratio**: `5.80%` of frame area
- **Congestion Frame Classification Distribution**:
  - **LOW**: 359 frames
  - **MEDIUM**: 0 frames
  - **HIGH**: 0 frames
- **Processed Video Output Path**: `D:\Projects\TEDR\temp\processed\processed_eval_video.mp4`