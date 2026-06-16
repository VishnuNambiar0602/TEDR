"""
AIR-DETR Long Video Evaluation & System Monitoring Tool.

Features:
- Processes the 34-minute video frame-by-frame with batch size 32
- Uses StreamingRTDETR FP32 for weight streaming
- Monitors system CPU, RAM, GPU, VRAM, and GPU Temperature in a background thread
- Writes real-time progress and performance logs to logs2.md
"""

import os
import sys
import time
import json
import subprocess
import threading
import psutil
import torch
import cv2
from typing import List, Dict, Any
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from air_detr.model import StreamingRTDETR
from air_detr.vram_manager import VRAMManager

# Configurations
VIDEO_PATH = r"C:\Users\vishn\Downloads\vidssave.com Road traffic video for object recognition 360P.mp4"
MODEL_PATH = "rtdetr-l.pt"
SHARD_DIR = "temp/video_eval_shards"
LOG_FILE = "logs2.md"
BATCH_SIZE = 32
MONITOR_INTERVAL = 5.0 # seconds

# Thread-safe collections for monitoring logs
cpu_usage_logs = []
ram_usage_logs = []
gpu_usage_logs = []
vram_usage_logs = []
gpu_temp_logs = []
monitoring_active = True

# Target classes mapping in COCO
TARGET_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    17: 'cat',
    18: 'dog',
    19: 'cow'
}

def get_system_metrics():
    """Query system performance metrics."""
    metrics = {
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().used / (1024 ** 3), # GB
        "gpu": 0.0,
        "vram": 0.0,
        "gpu_temp": 0.0
    }
    
    try:
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits"
        res = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        gpu_util, vram_used, temp = [float(x.strip()) for x in res.split(',')]
        metrics["gpu"] = gpu_util
        metrics["vram"] = vram_used
        metrics["gpu_temp"] = temp
    except Exception:
        # Fallback to PyTorch VRAM if nvidia-smi fails
        if torch.cuda.is_available():
            metrics["vram"] = torch.cuda.memory_allocated() / (1024 ** 2) # MB
            
    return metrics

def monitor_loop():
    """Background monitoring thread loop."""
    global monitoring_active
    # Warm up CPU call
    psutil.cpu_percent()
    
    while monitoring_active:
        metrics = get_system_metrics()
        cpu_usage_logs.append(metrics["cpu"])
        ram_usage_logs.append(metrics["ram"])
        gpu_usage_logs.append(metrics["gpu"])
        vram_usage_logs.append(metrics["vram"])
        gpu_temp_logs.append(metrics["gpu_temp"])
        time.sleep(MONITOR_INTERVAL)

def format_stats(logs: List[float]) -> Dict[str, float]:
    """Calculates min, average, and max stats from a log list."""
    if not logs:
        return {"min": 0.0, "avg": 0.0, "max": 0.0}
    return {
        "min": min(logs),
        "avg": sum(logs) / len(logs),
        "max": max(logs)
    }

def write_log_report(
    current_frame: int,
    total_frames: int,
    start_time: float,
    class_counts: Dict[str, int],
    status: str = "RUNNING"
):
    """Generates the logs2.md report."""
    elapsed = time.time() - start_time
    fps = current_frame / elapsed if elapsed > 0 else 0.0
    progress_pct = (current_frame / total_frames) * 100 if total_frames > 0 else 0.0
    
    # Calculate stats
    cpu_stats = format_stats(cpu_usage_logs)
    ram_stats = format_stats(ram_usage_logs)
    gpu_stats = format_stats(gpu_usage_logs)
    vram_stats = format_stats(vram_usage_logs)
    temp_stats = format_stats(gpu_temp_logs)
    
    report = []
    report.append("# AIR-DETR Video Analysis & Resource Monitoring Report")
    report.append(f"- **Execution Status**: `{status}`")
    report.append(f"- **Current Local Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"- **Video File**: `{VIDEO_PATH}`")
    report.append(f"- **Resolution**: 640x360 @ 25 FPS")
    report.append(f"- **Processed Frames**: {current_frame} / {total_frames} ({progress_pct:.2f}% progress)")
    report.append(f"- **Elapsed Execution Time**: {elapsed / 60:.2f} minutes ({elapsed:.1f} seconds)")
    report.append(f"- **Current Throughput**: {fps:.2f} FPS")
    report.append(f"- **Model**: `RT-DETR-L` (AirLLM-style FP32 Streaming, batch size {BATCH_SIZE})")
    report.append("")
    
    report.append("## 🔋 Hardware & Resource Usage Statistics")
    report.append("| Metric | Min | Average | Max | Unit |")
    report.append("| :--- | :---: | :---: | :---: | :---: |")
    report.append(f"| **System CPU Usage** | {cpu_stats['min']:.1f}% | {cpu_stats['avg']:.1f}% | {cpu_stats['max']:.1f}% | % |")
    report.append(f"| **System RAM Usage** | {ram_stats['min']:.2f} | {ram_stats['avg']:.2f} | {ram_stats['max']:.2f} | GB |")
    report.append(f"| **GPU Core Utilization** | {gpu_stats['min']:.1f}% | {gpu_stats['avg']:.1f}% | {gpu_stats['max']:.1f}% | % |")
    report.append(f"| **VRAM Memory Usage** | {vram_stats['min']:.1f} | {vram_stats['avg']:.1f} | {vram_stats['max']:.1f} | MiB |")
    report.append(f"| **GPU Core Temperature** | {temp_stats['min']:.1f}°C | {temp_stats['avg']:.1f}°C | {temp_stats['max']:.1f}°C | °C |")
    report.append("")
    
    report.append("## 📊 Object Detection Summary")
    report.append("Total occurrences of detected objects in the video:")
    report.append("| Object Class | Total Count |")
    report.append("| :--- | :---: |")
    for cls_name in sorted(class_counts.keys()):
        report.append(f"| **{cls_name.upper()}** | {class_counts[cls_name]} |")
        
    report.append("")
    report.append("---")
    report.append(f"*Note: Report updated automatically. Last update at frame {current_frame}.*")
    
    # Write to file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

def main():
    global monitoring_active
    print("Initializing Video Processor...")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        sys.exit(1)
        
    # Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Opened video: {total_frames} total frames.")
    
    # Load and wrap model
    yolo_model = YOLO(MODEL_PATH)
    stream_model = StreamingRTDETR(
        yolo_model=yolo_model,
        shard_dir=SHARD_DIR,
        quantization=None,
        format="pt"
    )
    
    # Initialize counts
    class_counts = {name: 0 for name in TARGET_CLASSES.values()}
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    
    start_time = time.time()
    frames_processed = 0
    
    print("Starting video processing loop...")
    try:
        while cap.isOpened():
            frames = []
            for _ in range(BATCH_SIZE):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                
            if not frames:
                break
                
            # Perform inference on batch
            # Disable grad and speed optimization
            with torch.no_grad():
                results = yolo_model(frames, verbose=False, device='cuda')
                
            # Parse results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    if cls_id in TARGET_CLASSES:
                        cls_name = TARGET_CLASSES[cls_id]
                        class_counts[cls_name] += 1
                        
            frames_processed += len(frames)
            
            # Periodically update report file
            if (frames_processed // BATCH_SIZE) % 10 == 0:
                print(f"Processed {frames_processed}/{total_frames} frames. FPS: {frames_processed/(time.time()-start_time):.2f}")
                write_log_report(frames_processed, total_frames, start_time, class_counts, status="RUNNING")
                
        # Processing complete
        print("Video processing complete!")
        monitoring_active = False
        monitor_thread.join(timeout=5)
        write_log_report(frames_processed, total_frames, start_time, class_counts, status="COMPLETED")
        
    except KeyboardInterrupt:
        print("Interrupted by user! Saving partial logs...")
        monitoring_active = False
        monitor_thread.join(timeout=5)
        write_log_report(frames_processed, total_frames, start_time, class_counts, status="INTERRUPTED")
        
    except Exception as e:
        print(f"Error during video processing: {e}")
        monitoring_active = False
        monitor_thread.join(timeout=5)
        write_log_report(frames_processed, total_frames, start_time, class_counts, status=f"FAILED ({str(e)})")
        
    finally:
        cap.release()
        stream_model.restore()
        print("Resources released.")

if __name__ == "__main__":
    main()
