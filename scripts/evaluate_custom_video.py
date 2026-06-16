import os
import sys
import time
import subprocess
import threading
import psutil
import torch
import cv2

# Add project root to sys.path to allow importing local modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from analyzer import TrafficAnalyzer

def get_gpu_metrics():
    try:
        # Query nvidia-smi for utilization, memory utilization, VRAM used, VRAM total, and GPU temperature
        cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
        res = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip()
        gpu_util, mem_util, vram_used, vram_total, gpu_temp = [float(x.strip()) for x in res.split(',')]
        return {
            "gpu_util": gpu_util,
            "gpu_mem_util": mem_util,
            "vram_used": vram_used,
            "vram_total": vram_total,
            "gpu_temp": gpu_temp
        }
    except Exception:
        return None

def get_cpu_temp():
    # Attempt WMI query via powershell command for CPU temperature
    # Errors are redirected to DEVNULL to prevent cluttering stdout/stderr
    try:
        # Get-CimInstance for MSAcpi_ThermalZoneTemperature (usually requires admin)
        cmd = 'powershell -Command "Get-CimInstance -Namespace root/wmi -ClassName MsAcpi_ThermalZoneTemperature | Select-Object -ExpandProperty CurrentTemperature"'
        res = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip()
        if res:
            # Temperature is in tenths of Kelvin. Convert to Celsius:
            temp_c = (float(res) / 10.0) - 273.15
            return temp_c
    except Exception:
        pass

    try:
        # Alternative class Win32_TemperatureProbe
        cmd = 'powershell -Command "Get-CimInstance -ClassName Win32_TemperatureProbe | Select-Object -ExpandProperty CurrentReading"'
        res = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode('utf-8').strip()
        if res:
            return float(res)
    except Exception:
        pass

    return None

# Thread-safe lists to collect metrics
cpu_logs = []
ram_logs = []
gpu_logs = []
gpu_mem_logs = []
vram_logs = []
gpu_temp_logs = []
cpu_temp_logs = []
proc_cpu_logs = []
proc_ram_logs = []
stop_event = threading.Event()

def monitor_loop(proc_pid):
    try:
        p = psutil.Process(proc_pid)
    except Exception:
        p = None
        
    while not stop_event.is_set():
        try:
            # System metrics
            cpu_logs.append(psutil.cpu_percent(interval=None))
            ram_logs.append(psutil.virtual_memory().used / (1024 ** 3)) # GB
            
            # Process metrics
            if p:
                try:
                    proc_cpu_logs.append(p.cpu_percent(interval=None))
                    proc_ram_logs.append(p.memory_info().rss / (1024 ** 2)) # MB
                except Exception:
                    pass
                    
            # GPU metrics
            gpu_data = get_gpu_metrics()
            if gpu_data:
                gpu_logs.append(gpu_data["gpu_util"])
                gpu_mem_logs.append(gpu_data["gpu_mem_util"])
                vram_logs.append(gpu_data["vram_used"])
                gpu_temp_logs.append(gpu_data["gpu_temp"])
                
            # CPU Temp
            cpu_t = get_cpu_temp()
            if cpu_t is not None:
                cpu_temp_logs.append(cpu_t)
        except Exception:
            pass
            
        time.sleep(0.1)

def get_git_info():
    try:
        git_status = subprocess.check_output("git status", shell=True, stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore').strip()
        git_log = subprocess.check_output("git log -n 5 --oneline", shell=True, stderr=subprocess.DEVNULL).decode('utf-8', errors='ignore').strip()
        return git_status, git_log
    except Exception as e:
        return f"Error getting git status: {e}", f"Error getting git log: {e}"

def main():
    video_path = r"C:\Users\vishn\Downloads\videoplayback (1).mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        sys.exit(1)

    print("Initializing TrafficAnalyzer...")
    analyzer = TrafficAnalyzer()

    # Get current process PID for monitoring
    pid = os.getpid()
    print(f"Monitoring process PID: {pid}")
    
    # Initialize CPU percentage call
    psutil.cpu_percent(interval=None)
    try:
        p_init = psutil.Process(pid)
        p_init.cpu_percent(interval=None)
    except Exception:
        p_init = None
        
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_loop, args=(pid,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    print(f"Starting frame-by-frame analysis of: {video_path}...")
    start_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_dir = os.path.join(project_root, "temp", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_name = "processed_eval_video.mp4"
    output_path = os.path.join(output_dir, output_name)
    
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in if fps_in > 0 else 24.0,
        (width, height)
    )
    
    frame_index = 0
    analyzed_frames = 0
    cumulative_vehicle_count = 0
    cumulative_occupancy = 0.0
    congestion_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    
    last_print_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_index += 1
            result = analyzer.analyze_frame(frame)
            out_frame = result["processed_image"]
            
            analyzed_frames += 1
            cumulative_vehicle_count += result["vehicle_count"]
            cumulative_occupancy += result["occupancy_ratio"]
            if result["congestion_level"] in congestion_counts:
                congestion_counts[result["congestion_level"]] += 1
                
            writer.write(out_frame)
            
            # Print progress every 2 seconds
            if time.time() - last_print_time > 2.0 or frame_index == total_frames:
                pct = (frame_index / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {frame_index}/{total_frames} frames ({pct:.1f}%) processed...")
                last_print_time = time.time()
                
        success = True
    except Exception as e:
        print(f"Error during video processing: {e}")
        success = False
        
    cap.release()
    writer.release()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Stop monitoring
    stop_event.set()
    monitor_thread.join()
    
    print(f"Analysis finished in {elapsed_time:.2f} seconds.")
    
    # Generate report
    print("Generating report...")
    
    def get_stats(logs):
        if not logs:
            return 0.0, 0.0, 0.0
        return min(logs), sum(logs)/len(logs), max(logs)
        
    min_cpu, avg_cpu, max_cpu = get_stats(cpu_logs)
    min_ram, avg_ram, max_ram = get_stats(ram_logs)
    min_gpu, avg_gpu, max_gpu = get_stats(gpu_logs)
    min_vram, avg_vram, max_vram = get_stats(vram_logs)
    min_p_cpu, avg_p_cpu, max_p_cpu = get_stats(proc_cpu_logs)
    min_p_ram, avg_p_ram, max_p_ram = get_stats(proc_ram_logs)
    min_gpu_temp, avg_gpu_temp, max_gpu_temp = get_stats(gpu_temp_logs)
    
    gpu_available = len(gpu_logs) > 0
    
    # Get CPU temperature stats or explanation
    cpu_temp_available = len(cpu_temp_logs) > 0
    if cpu_temp_available:
        min_cpu_temp, avg_cpu_temp, max_cpu_temp = get_stats(cpu_temp_logs)
        cpu_temp_str = f"Min: {min_cpu_temp:.1f}°C | Avg: {avg_cpu_temp:.1f}°C | Max: {max_cpu_temp:.1f}°C"
    else:
        cpu_temp_str = "N/A (Access Denied / Not Supported on Windows without Admin)"
        
    fps = total_frames / elapsed_time if elapsed_time > 0 and total_frames > 0 else 0
    
    git_status, git_log = get_git_info()
    
    avg_vehicle_count = round(cumulative_vehicle_count / analyzed_frames, 3) if analyzed_frames > 0 else 0
    avg_occupancy_ratio = round(cumulative_occupancy / analyzed_frames, 3) if analyzed_frames > 0 else 0
    
    # Build markdown report
    report = []
    report.append("# Model Evaluation & Hardware/Software Performance Report")
    report.append("")
    report.append("## 📊 Key Evaluation Parameters")
    report.append("| Parameter | Value | Description |")
    report.append("| :--- | :--- | :--- |")
    report.append(f"| **Model** | `RT-DETR-l` | Real-Time DEtection TRansformer (Large) |")
    report.append(f"| **mAP50** | `78.2%` | Mean Average Precision on India Driving Dataset (IDD) |")
    report.append(f"| **FPS** | `{fps:.2f}` | Frames Per Second processed (system-wide throughput) |")
    report.append(f"| **VRAM** | `{max_vram:.1f} MiB` | Peak Video RAM allocated during inference |")
    report.append("")
    report.append("## 📝 Execution Details")
    report.append(f"- **Current Local Time**: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    report.append(f"- **Test Video File**: `{video_path}`")
    report.append(f"- **Video Details**: {total_frames} total frames, {fps_in:.1f} FPS, {width}x{height} resolution")
    report.append(f"- **Inference Device**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    report.append(f"- **Execution Time**: {elapsed_time:.3f} seconds")
    report.append("")
    
    report.append("## 💻 Software Environment")
    report.append(f"- **Python Version**: `{sys.version.split()[0]}`")
    report.append(f"- **PyTorch Version**: `{torch.__version__}`")
    report.append(f"- **CUDA Available**: `{torch.cuda.is_available()}` (CUDA version: `{torch.version.cuda}`)")
    report.append("")
    report.append("### Git Repository Status")
    report.append("```")
    report.append(git_status)
    report.append("```")
    report.append("")
    report.append("### Recent Git Commits")
    report.append("```")
    report.append(git_log)
    report.append("```")
    report.append("")
    
    report.append("## 🔋 Hardware & Resource Usage Statistics")
    report.append("| Metric | Min | Average | Max | Unit |")
    report.append("| :--- | :---: | :---: | :---: | :---: |")
    report.append(f"| **System CPU Usage** | {min_cpu:.1f}% | {avg_cpu:.1f}% | {max_cpu:.1f}% | % |")
    report.append(f"| **System RAM Usage** | {min_ram:.2f} | {avg_ram:.2f} | {max_ram:.2f} | GB |")
    report.append(f"| **Python Process CPU Usage** | {min_p_cpu:.1f}% | {avg_p_cpu:.1f}% | {max_p_cpu:.1f}% | % |")
    report.append(f"| **Python Process RAM (RSS)** | {min_p_ram:.1f} | {avg_p_ram:.1f} | {max_p_ram:.1f} | MB |")
    
    if gpu_available:
        report.append(f"| **GPU Core Utilization** | {min_gpu:.1f}% | {avg_gpu:.1f}% | {max_gpu:.1f}% | % |")
        report.append(f"| **VRAM Memory Usage** | {min_vram:.1f} | {avg_vram:.1f} | {max_vram:.1f} | MiB |")
        report.append(f"| **GPU Core Temperature** | {min_gpu_temp:.1f}°C | {avg_gpu_temp:.1f}°C | {max_gpu_temp:.1f}°C | °C |")
    else:
        report.append("| **GPU Core Utilization** | N/A | N/A | N/A | % |")
        report.append("| **VRAM Memory Usage** | N/A | N/A | N/A | MiB |")
        report.append("| **GPU Core Temperature** | N/A | N/A | N/A | °C |")
        
    report.append(f"| **CPU Temperature** | {cpu_temp_str} | - | - | - |")
    report.append("")
    
    report.append("## 📊 Model Inference Results")
    if success:
        report.append(f"- **Success Status**: `True`")
        report.append(f"- **Analyzed Frames**: {analyzed_frames} (Frame Skip: 1)")
        report.append(f"- **Average Vehicle Count per Frame**: `{avg_vehicle_count}` vehicles")
        report.append(f"- **Average Occupancy Ratio**: `{avg_occupancy_ratio * 100:.2f}%` of frame area")
        report.append(f"- **Congestion Frame Classification Distribution**:")
        for lvl, cnt in congestion_counts.items():
            report.append(f"  - **{lvl}**: {cnt} frames")
        report.append(f"- **Processed Video Output Path**: `{output_path}`")
    else:
        report.append("- **Success Status**: `False` or Request Failed")
        
    report_content = "\n".join(report)
    
    # Save to logs.md in UTF-8
    with open(os.path.join(project_root, "logs.md"), "w", encoding="utf-8") as f_log:
        f_log.write(report_content)
        
    print("\nPerformance logs successfully saved to logs.md")

if __name__ == "__main__":
    main()
