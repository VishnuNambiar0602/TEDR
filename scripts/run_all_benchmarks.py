import os
import shutil
import subprocess
import sys

def main():
    print("Starting AIR-DETR benchmarking pipeline...")
    
    # 1. Setup directories
    os.makedirs("RT-DETR-L", exist_ok=True)
    os.makedirs("RT-DETR-X", exist_ok=True)
    
    # 2. Copy models to respective directories
    print("Copying RT-DETR-L model to RT-DETR-L/rtdetr-l.pt...")
    if os.path.exists("rtdetr-l.pt"):
        shutil.copy("rtdetr-l.pt", "RT-DETR-L/rtdetr-l.pt")
    else:
        print("Warning: rtdetr-l.pt not found in root. It will be downloaded dynamically.")
        
    print("Copying RT-DETR-X model to RT-DETR-X/rtdetr-x.pt...")
    if os.path.exists("rtdetr-x.pt"):
        shutil.copy("rtdetr-x.pt", "RT-DETR-X/rtdetr-x.pt")
    else:
        print("Warning: rtdetr-x.pt not found in root. It will be downloaded dynamically.")
        
    # 3. Clean up any leftover benchmark shards to make sure it runs clean
    shutil.rmtree("RT-DETR-L/shards", ignore_errors=True)
    shutil.rmtree("RT-DETR-X/shards", ignore_errors=True)
    
    # 4. Run RT-DETR-L benchmark
    print("\n-------------------------------------------")
    print("Running Benchmark Suite for RT-DETR-L...")
    print("-------------------------------------------")
    cmd_l = [
        sys.executable,
        "benchmarks/benchmark_streaming.py",
        "--model", "RT-DETR-L/rtdetr-l.pt",
        "--out-dir", "RT-DETR-L",
        "--shard-dir", "RT-DETR-L/shards"
    ]
    subprocess.run(cmd_l, check=True)
    print("RT-DETR-L benchmark completed successfully.")
    
    # 5. Run RT-DETR-X benchmark
    print("\n-------------------------------------------")
    print("Running Benchmark Suite for RT-DETR-X...")
    print("-------------------------------------------")
    cmd_x = [
        sys.executable,
        "benchmarks/benchmark_streaming.py",
        "--model", "RT-DETR-X/rtdetr-x.pt",
        "--out-dir", "RT-DETR-X",
        "--shard-dir", "RT-DETR-X/shards"
    ]
    subprocess.run(cmd_x, check=True)
    print("RT-DETR-X benchmark completed successfully.")
    
    print("\n[SUCCESS] Benchmarking pipeline completed. Results saved in RT-DETR-L/results.md and RT-DETR-X/results.md.")

if __name__ == "__main__":
    main()
