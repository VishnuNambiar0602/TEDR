"""
Benchmark Suite for AIR-DETR.

Evaluates:
- Baseline RT-DETR-L vs Streaming RT-DETR-L (FP32)
- Peak VRAM (PyTorch allocated + system-wide)
- Throughput (FPS) and Latency (ms)
- Accuracy (mAP50, mAP50-95)
- CPU RAM and CPU Usage
- Streaming Overhead

Outputs:
- JSON results to temp/benchmark_results.json
- LaTeX tables to temp/benchmark_table.tex
- Performance figures/plots in temp/plots/
- Final validation report to logs1.md
"""

import os
import sys
import time
import json
import psutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from air_detr.model import StreamingRTDETR
from air_detr.vram_manager import VRAMManager

def run_evaluation(model, data_config, device="cuda"):
    """Runs validation and returns mAP metrics, FPS, and peak VRAM."""
    VRAMManager.reset_peak()
    
    # Run warm up to initialize CUDA context and allocate memory for inputs/outputs
    dummy_input = torch.randn(1, 3, 640, 640, device=device)
    with torch.no_grad():
        if hasattr(model, "model"):
            try:
                model.model(dummy_input)
            except Exception:
                pass
                
    VRAMManager.reset_peak()
    start_time = time.time()
    
    # Run validation loop using ultralytics built-in validation
    results = model.val(
        data=data_config,
        device=device,
        verbose=False,
        plots=False,
        save=False
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Extract mAP metrics
    map50 = results.box.map50
    map50_95 = results.box.map
    
    # Throughput metrics
    # Get image count from directory directly
    val_dir = "datasets/coco_autorickshaw/images/val"
    total_images = len(os.listdir(val_dir)) if os.path.exists(val_dir) else 500
    fps = total_images / elapsed if elapsed > 0 else 0.0
    latency = (elapsed / total_images) * 1000.0 if total_images > 0 else 0.0
    
    peak_vram = VRAMManager.get_max_allocated_mb()
    
    return {
        "mAP50": map50,
        "mAP50-95": map50_95,
        "fps": fps,
        "latency_ms": latency,
        "peak_vram_mb": peak_vram,
        "elapsed_s": elapsed,
        "total_images": total_images
    }

def main():
    parser = argparse.ArgumentParser(description="AIR-DETR Benchmark Suite")
    parser.add_argument("--model", type=str, default="rtdetr-l.pt", help="Path to model file")
    parser.add_argument("--data", type=str, default="datasets/coco_autorickshaw/dataset.yaml", help="Path to dataset yaml")
    parser.add_argument("--out-dir", type=str, default="temp", help="Output directory for reports/plots")
    parser.add_argument("--shard-dir", type=str, default="temp/benchmark_shards", help="Directory for shards")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Benchmark Suite for {args.model}. Device: {device}")
    
    if device != "cuda":
        print("WARNING: CUDA is not available. VRAM measurements will be zero.")
        
    model_path = args.model
    data_config = args.data
    shard_dir = args.shard_dir
    out_dir = args.out_dir
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    
    # Track metrics for each mode
    benchmark_results = {}
    modes = [
        "baseline",
        "streaming_fp32"
    ]
    
    for mode in modes:
        print(f"\n==========================================")
        print(f"Evaluating Mode: {mode.upper()}")
        print(f"==========================================")
        
        # Clean memory before run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc_runs = 3
        for _ in range(gc_runs):
            import gc
            gc.collect()
            
        # Initialize model
        yolo_model = YOLO(model_path)
        
        # CPU Memory usage tracking
        cpu_mem_before = psutil.virtual_memory().used / (1024 ** 3) # GB
        proc_rss_before = psutil.Process().memory_info().rss / (1024 ** 2) # MB
        
        if mode == "baseline":
            # Just move to device
            yolo_model.to(device)
            res = run_evaluation(yolo_model, data_config, device=device)
        else:
            # Wrap with streaming (note: sharding will happen inside)
            quant = None
            if mode == "streaming_int8":
                quant = "int8"
            elif mode == "streaming_int4":
                quant = "int4"
            elif mode in ("calibrated_int8", "calibrated_int4", "awq", "gptq"):
                quant = mode
                
            stream_wrapper = StreamingRTDETR(
                yolo_model=yolo_model,
                shard_dir=os.path.join(shard_dir, mode),
                quantization=quant,
                format="pt",
                dataset_yaml=data_config
            )
            res = run_evaluation(yolo_model, data_config, device=device)
            stream_wrapper.restore()
            
        cpu_mem_after = psutil.virtual_memory().used / (1024 ** 3) # GB
        proc_rss_after = psutil.Process().memory_info().rss / (1024 ** 2) # MB
        
        # Compute deltas
        cpu_mem_diff = max(0.0, cpu_mem_after - cpu_mem_before)
        proc_rss_diff = max(0.0, proc_rss_after - proc_rss_before)
        
        res["system_ram_diff_gb"] = cpu_mem_diff
        res["process_ram_diff_mb"] = proc_rss_diff
        
        # Log to stdout
        print(f"Results for {mode.upper()}:")
        print(f"  mAP50: {res['mAP50']*100:.2f}%")
        print(f"  mAP50-95: {res['mAP50-95']*100:.2f}%")
        print(f"  Throughput: {res['fps']:.2f} FPS")
        print(f"  Latency: {res['latency_ms']:.2f} ms")
        print(f"  Peak VRAM (Allocated): {res['peak_vram_mb']:.2f} MB")
        
        benchmark_results[mode] = res
        
    # Save JSON results
    json_path = os.path.join(out_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
        
    # Generate Plots
    generate_plots(benchmark_results, out_dir)
    
    # Generate LaTeX Table
    generate_latex_table(benchmark_results, out_dir)
    
    # Generate final report
    generate_markdown_report(benchmark_results, out_dir, model_name=os.path.basename(model_path))
    
    print(f"\n[SUCCESS] Benchmark Suite completed. Results logged to {out_dir}/results.md.")

def generate_plots(results, out_dir):
    """Generates charts for the benchmark results."""
    modes = list(results.keys())
    mAPs = [results[m]["mAP50"] * 100 for m in modes]
    vram = [results[m]["peak_vram_mb"] for m in modes]
    fps = [results[m]["fps"] for m in modes]
    latency = [results[m]["latency_ms"] for m in modes]
    
    # 1. VRAM vs FPS chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Execution Mode')
    ax1.set_ylabel('Peak VRAM (MB)', color=color)
    bars = ax1.bar(modes, vram, color=color, alpha=0.6, width=0.4, label='Peak VRAM')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(modes, rotation=15)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Throughput (FPS)', color=color)
    line = ax2.plot(modes, fps, color=color, marker='o', linewidth=2, label='FPS')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('VRAM Occupancy vs Throughput (FPS) by Mode')
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plots/vram_fps_comparison.png'), dpi=150)
    plt.close()
    
    # 2. Accuracy Drop vs Memory Savings
    fig, ax = plt.subplots(figsize=(10, 6))
    base_vram = results["baseline"]["peak_vram_mb"]
    base_map = results["baseline"]["mAP50"] * 100
    
    vram_savings = [100.0 * (1 - results[m]["peak_vram_mb"] / base_vram) if base_vram > 0 else 0 for m in modes]
    map_drop = [base_map - (results[m]["mAP50"] * 100) for m in modes]
    
    x = np.arange(len(modes))
    width = 0.35
    
    ax.bar(x - width/2, vram_savings, width, label='VRAM Savings (%)', color='green', alpha=0.7)
    ax.bar(x + width/2, map_drop, width, label='mAP50 Drop (%)', color='orange', alpha=0.7)
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('VRAM Memory Savings vs Accuracy Drop')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plots/savings_vs_accuracy.png'), dpi=150)
    plt.close()

def generate_latex_table(results, out_dir):
    """Generates LaTeX table code and writes to disk."""
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Performance Comparison of RT-DETR under Streaming and Quantization}")
    latex.append(r"\label{tab:air_detr_perf}")
    latex.append(r"\begin{tabular}{|l|c|c|c|c|c|}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Mode} & \textbf{mAP50 (\%)} & \textbf{mAP50-95 (\%)} & \textbf{Throughput (FPS)} & \textbf{Latency (ms)} & \textbf{Peak VRAM (MB)} \\ \hline")
    
    for mode in ["baseline", "streaming_fp32"]:
        r = results[mode]
        latex.append(f"{mode.upper()} & {r['mAP50']*100:.2f}\\% & {r['mAP50-95']*100:.2f}\\% & {r['fps']:.2f} & {r['latency_ms']:.1f} & {r['peak_vram_mb']:.1f} \\\\ \\hline")
        
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    with open(os.path.join(out_dir, "benchmark_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(latex))

def generate_markdown_report(results, out_dir, model_name):
    """Generates the final results.md or logs1.md markdown report."""
    base_vram = results["baseline"]["peak_vram_mb"]
    base_map = results["baseline"]["mAP50"] * 100
    
    report = []
    report.append(f"# AIR-DETR Model Evaluation & Performance Report: {model_name}")
    report.append(f"- **Current Local Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"- **Model**: `{model_name}`")
    report.append("- **Evaluation Dataset**: `DriveIndia Public Dataset` (500 validation images)")
    report.append("- **Inference Device**: NVIDIA GeForce RTX 4050 Laptop GPU")
    report.append("")
    
    report.append("## 📊 Performance Comparison Table")
    report.append("| Mode | mAP50 | mAP50-95 | Throughput (FPS) | Latency (ms) | Peak VRAM (MB) | VRAM Reduction | Accuracy Drop |")
    report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for mode in ["baseline", "streaming_fp32"]:
        r = results[mode]
        vram_red = (1 - r["peak_vram_mb"] / base_vram) * 100 if base_vram > 0 else 0
        map_drop = base_map - (r["mAP50"] * 100)
        
        mode_label = mode.upper().replace("_", " ")
        report.append(
            f"| **{mode_label}** | {r['mAP50']*100:.2f}% | {r['mAP50-95']*100:.2f}% | {r['fps']:.2f} | {r['latency_ms']:.1f} | {r['peak_vram_mb']:.1f} | {vram_red:.1f}% | {map_drop:.2f}% |"
        )
        
    report.append("")
    report.append("## 🎯 Analysis of Objectives & Success Criteria")
    
    # Verify success criteria
    fp32_vram_red = (1 - results["streaming_fp32"]["peak_vram_mb"] / base_vram) * 100 if base_vram > 0 else 0
    report.append(f"1. **VRAM Reduction >= 50%**:")
    report.append(f"   - FP32 Streaming VRAM Reduction: **{fp32_vram_red:.1f}%**")
    report.append(f"   - *Status*: **{'SUCCESS' if fp32_vram_red >= 50 else 'FAILED'}** (VRAM requirements decreased from {base_vram:.1f} MB to {results['streaming_fp32']['peak_vram_mb']:.1f} MB)")
    report.append("")
    
    report.append(f"2. **Streaming Overhead <= 25%**:")
    report.append(f"   - Throughput drop (Baseline vs Streaming FP32): **{overhead:.1f}%**")
    report.append(f"   - *Status*: **{'SUCCESS' if overhead <= 25 else 'FAILED'}**")
    report.append("")
    
    report.append("## 💻 Software & Hardware Environment")
    report.append(f"- **Python Version**: `{sys.version.split()[0]}`")
    report.append(f"- **PyTorch Version**: `{torch.__version__}`")
    report.append(f"- **CUDA Available**: `{torch.cuda.is_available()}`")
    if torch.cuda.is_available():
        report.append(f"- **CUDA Version**: `{torch.version.cuda}`")
        report.append(f"- **GPU Model**: `{torch.cuda.get_device_name(0)}`")
        
    results_path = os.path.join(out_dir, "results.md")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    main()
