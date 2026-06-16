import os
import sys
import time
import subprocess
import threading
import requests
import psutil

def get_gpu_metrics():
    try:
        # Query nvidia-smi for utilization and memory
        cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits"
        res = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        gpu_util, mem_util, vram_used, vram_total = [float(x.strip()) for x in res.split(',')]
        return {
            "gpu_util": gpu_util,
            "gpu_mem_util": mem_util,
            "vram_used": vram_used,
            "vram_total": vram_total
        }
    except Exception:
        return None

# Thread-safe lists to collect metrics
cpu_logs = []
ram_logs = []
gpu_logs = []
vram_logs = []
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
                    # Measure process cpu percent
                    proc_cpu_logs.append(p.cpu_percent(interval=None))
                    proc_ram_logs.append(p.memory_info().rss / (1024 ** 2)) # MB
                except Exception:
                    pass
                    
            # GPU metrics
            gpu_data = get_gpu_metrics()
            if gpu_data:
                gpu_logs.append(gpu_data["gpu_util"])
                vram_logs.append(gpu_data["vram_used"])
        except Exception as e:
            print(f"Error in monitor loop: {e}")
            
        time.sleep(0.2)

def main():
    print("Starting TrafficAI server subprocess...")
    # Start the server using python -m uvicorn app:app --port 8000
    server_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "app:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    # Wait for the server to be ready
    print("Waiting for server to load models and start...")
    ready = False
    for _ in range(45): # Give up to 45 seconds for PyTorch and RT-DETR to load
        try:
            r = requests.get("http://127.0.0.1:8000/")
            if r.status_code == 200:
                ready = True
                print("Server is ready!")
                break
        except Exception:
            pass
        # Check if the process exited unexpectedly
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            print("Server process terminated early!")
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            sys.exit(1)
        time.sleep(1)
        
    if not ready:
        print("Server failed to start in time. Terminating...")
        server_process.terminate()
        sys.exit(1)
        
    pid = server_process.pid
    print(f"Server PID: {pid}")
    
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
    
    # Path to test video
    video_path = "test_video.mp4"
    if not os.path.exists(video_path):
        print("Error: test_video.mp4 not found!")
        stop_event.set()
        server_process.terminate()
        sys.exit(1)
        
    print(f"Sending video processing request to /analyze_video with {video_path}...")
    start_time = time.time()
    
    response_data = None
    success = False
    try:
        with open(video_path, 'rb') as f:
            files = {'file': f}
            data = {'frame_skip': '1'}
            response = requests.post("http://127.0.0.1:8000/analyze_video", files=files, data=data, timeout=120)
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Request finished in {elapsed_time:.2f} seconds.")
        
        if response.status_code == 200:
            response_data = response.json()
            success = response_data.get("success", False)
            print("Response:", response_data)
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(response.text)
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Exception during request: {e}")
        success = False
        
    # Stop monitoring
    stop_event.set()
    monitor_thread.join()
    
    # Terminate server
    print("Terminating server...")
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()
        
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
    
    gpu_available = len(gpu_logs) > 0
    
    total_frames = response_data.get("total_frames", 0) if response_data else 0
    analyzed_frames = response_data.get("analyzed_frames", 0) if response_data else 0
    fps = total_frames / elapsed_time if elapsed_time > 0 and total_frames > 0 else 0
    
    # Build markdown report
    report = []
    report.append("# Performance & Resource Usage Logs")
    report.append(f"- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"- **Test Video**: `{video_path}`")
    report.append(f"- **Total Video Frames**: {total_frames}")
    report.append(f"- **Analyzed Frames**: {analyzed_frames} (Frame Skip: 1)")
    report.append(f"- **Execution Time**: {elapsed_time:.3f} seconds")
    report.append(f"- **Throughput**: {fps:.2f} FPS")
    report.append(f"- **Inference Device**: {'GPU (CUDA)' if gpu_available else 'CPU'}")
    report.append("")
    report.append("## Resource Usage Statistics")
    report.append("| Metric | Min | Average | Max | Unit |")
    report.append("| :--- | :---: | :---: | :---: | :---: |")
    report.append(f"| **System CPU Usage** | {min_cpu:.1f}% | {avg_cpu:.1f}% | {max_cpu:.1f}% | % |")
    report.append(f"| **System RAM Usage** | {min_ram:.2f} | {avg_ram:.2f} | {max_ram:.2f} | GB |")
    report.append(f"| **Uvicorn CPU Usage** | {min_p_cpu:.1f}% | {avg_p_cpu:.1f}% | {max_p_cpu:.1f}% | % |")
    report.append(f"| **Uvicorn RAM (RSS)** | {min_p_ram:.1f} | {avg_p_ram:.1f} | {max_p_ram:.1f} | MB |")
    
    if gpu_available:
        report.append(f"| **GPU Core Utilization** | {min_gpu:.1f}% | {avg_gpu:.1f}% | {max_gpu:.1f}% | % |")
        report.append(f"| **VRAM Memory Usage** | {min_vram:.1f} | {avg_vram:.1f} | {max_vram:.1f} | MiB |")
    else:
        report.append("| **GPU Core Utilization** | N/A | N/A | N/A | % |")
        report.append("| **VRAM Memory Usage** | N/A | N/A | N/A | MiB |")
        
    report.append("")
    report.append("## API Response Details")
    if success and response_data:
        report.append(f"- **Success**: `True`")
        report.append(f"- **Average Vehicle Count**: {response_data.get('average_vehicle_count', 0)}")
        report.append(f"- **Average Occupancy Ratio**: {response_data.get('average_occupancy_ratio', 0)}")
        report.append(f"- **Congestion Level Distribution**: {response_data.get('congestion_frame_counts', {})}")
        report.append(f"- **Processed Video Download Link**: `{response_data.get('download_url', '')}`")
    else:
        report.append("- **Success**: `False` or Request Failed")
        
    report_content = "\n".join(report)
    
    # Save to logs.md
    with open("logs.md", "w") as f_log:
        f_log.write(report_content)
        
    print("\nPerformance logs successfully saved to logs.md")

if __name__ == "__main__":
    main()
