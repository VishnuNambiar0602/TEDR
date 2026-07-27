# Model Evaluation & Hardware/Software Performance Report

## 📊 Key Evaluation Parameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model** | `RT-DETR-l` | Real-Time DEtection TRansformer (Large) |
| **mAP50** | `78.2%` | Mean Average Precision on India Driving Dataset (IDD) |
| **FPS** | `98.23` | Frames Per Second processed (system-wide throughput) |
| **VRAM** | `1034.0 MiB` | Peak Video RAM allocated during inference |

## 📝 Execution Details
- **Current Local Time**: 2026-06-18 22:23:25 India Standard Time
- **Test Video File**: `D:\Projects\TEDR\test_video.mp4`
- **Video Details**: 359 total frames, 24.0 FPS, 720x1280 resolution
- **Inference Device**: NVIDIA GeForce RTX 4050 Laptop GPU
- **Execution Time**: 3.655 seconds

## 💻 Software Environment
- **Python Version**: `3.14.0`
- **PyTorch Version**: `2.11.0.dev20260119+cu126`
- **CUDA Available**: `True` (CUDA version: `12.6`)

### Git Repository Status
```
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   analyzer.py
	modified:   app.py
	modified:   scripts/evaluate_custom_video.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	New Text Document.txt
	logs3.md
	pure_RT-DETR/
	rtdetr-l.engine
	rtdetr-l.onnx
	temp_test.avi
	temp_test.mp4
	visualise.html

no changes added to commit (use "git add" and/or "git commit -a")
```

### Recent Git Commits
```
f2e6893 Implement weight streaming, quantization modes, and run benchmarks
2d2f56c Merge pull request #2 from VishnuNambiar0602/copilot/implement-detr-object-detection
241960e Merge branch 'main' into copilot/implement-detr-object-detection
02d8f66 Merge pull request #1 from VishnuNambiar0602/copilot/create-detr-object-detection
2049820 Add final validation checklist - all requirements complete
```

## 🔋 Hardware & Resource Usage Statistics
| Metric | Min | Average | Max | Unit |
| :--- | :---: | :---: | :---: | :---: |
| **System CPU Usage** | 0.0% | 21.3% | 33.9% | % |
| **System RAM Usage** | 8.56 | 9.28 | 9.68 | GB |
| **Python Process CPU Usage** | 0.0% | 157.1% | 266.1% | % |
| **Python Process RAM (RSS)** | 650.1 | 1387.9 | 1811.8 | MB |
| **GPU Core Utilization** | 0.0% | 4.0% | 15.0% | % |
| **VRAM Memory Usage** | 618.0 | 930.0 | 1034.0 | MiB |
| **GPU Core Temperature** | 42.0°C | 44.8°C | 48.0°C | °C |
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