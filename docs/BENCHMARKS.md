# AIR-DETR Benchmark Suite

This document describes the benchmark setup, methodology, and measurement criteria for evaluating **AIR-DETR (v1)** performance.

---

## 📊 1. Evaluated Configurations

The benchmark compares four distinct system configurations:

1. **BASELINE**: The standard RT-DETR-L model loaded fully into GPU VRAM (FP32).
2. **STREAMING_FP32**: AirLLM-style layer streaming from CPU RAM to GPU, without weight quantization.
3. **STREAMING_INT8**: Streaming with Post-Training Quantization (PTQ) to INT8, utilizing channel-wise scaling.
4. **STREAMING_INT4**: Streaming with 4-bit packed weights unpacked on the fly.

---

## 📐 2. Key Performance Indicators (KPIs)

For each configuration, the suite measures:

### Memory Metrics
* **Peak VRAM (Allocated)**: The maximum GPU memory allocated by PyTorch for weights and activations (measured via `torch.cuda.max_memory_allocated()`).
* **System RAM Usage**: Additional CPU RAM footprint caused by storing model layer shards in host memory.

### Speed Metrics
* **Latency**: The average execution time per frame in milliseconds.
* **Throughput (FPS)**: The system-wide throughput (Frames Per Second) processed during the validation pass.
* **Streaming Overhead**: The percentage reduction in throughput compared to the baseline, defined as:
  $$\text{Overhead} = \left( 1 - \frac{\text{FPS}_{\text{streaming}}}{\text{FPS}_{\text{baseline}}} \right) \times 100\%$$

### Accuracy Metrics
* **mAP50**: Mean Average Precision at an Intersection over Union (IoU) threshold of 0.5.
* **mAP50-95**: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95 (COCO standard).
* **Accuracy Drop**: The absolute percentage drop in mAP50 compared to the baseline.

---

## 🏃 3. Re-running the Benchmarks

You can run the benchmark suite with a single command from the project root:

```bash
python benchmarks/benchmark_streaming.py
```

Or using the automated `Makefile` target:

```bash
make benchmark
```

### Outputs Generated:
* **`logs1.md`**: The formatted performance report containing markdown comparison tables and success criteria validation.
* **`temp/benchmark_results.json`**: Raw metric data.
* **`temp/benchmark_table.tex`**: Automatically formatted LaTeX table code for research reports.
* **`temp/plots/vram_fps_comparison.png`**: Matplotlib plot comparing peak VRAM and FPS across modes.
* **`temp/plots/savings_vs_accuracy.png`**: Chart showing VRAM memory savings against accuracy drop.
