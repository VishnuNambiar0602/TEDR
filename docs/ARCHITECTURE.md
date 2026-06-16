# AIR-DETR Architecture (v1)

This document provides a technical deep dive into the design and implementation of **AIR-DETR (Adaptive Inference RT-DETR)**, a systems engineering framework designed to enable the execution of RT-DETR object detection models with significantly reduced VRAM footprint.

---

## 🛠️ System Overview

AIR-DETR implements **AirLLM-style layer-by-layer weight streaming** combined with **Post-Training Quantization (PTQ)**. The core philosophy is that only the weights for the *currently active layer* during the forward pass should reside in GPU VRAM. The remaining layers are stored on CPU RAM (L2 cache) or NVMe SSD (L3 cache) and loaded on demand.

```
       +---------------------------------------------+
       |                  NVMe SSD                   |
       |  (Sharded layer files: layer_000.safetensors)|
       +----------------------|----------------------+
                              | (Background Thread)
                              v
       +---------------------------------------------+
       |                  CPU RAM                    |
       |   (L2 Cache: Pinned Host Memory state_dicts) |
       +----------------------|----------------------+
                              | (Non-blocking PCIe Transfer)
                              v
       +---------------------------------------------+
       |                  GPU VRAM                   |
       |  (L1 Cache: Active Layer Weights + Dequant) |
       +----------------------|----------------------+
                              v
       +---------------------------------------------+
       |               Execution Stream              |
       |            (Layer Forward Pass)             |
       +---------------------------------------------+
```

---

## 📦 Core Architectural Components

### 1. Quantization Engine (`QuantizationManager`)
AIR-DETR supports channel-wise symmetric Post-Training Quantization (PTQ) to INT8 and INT4.
* **INT8 PTQ**: Weights are scaled per-channel (dimension 0) such that the maximum absolute value maps to $127$. Scale factors are stored in FP32/FP16.
* **INT4 PTQ**: Weights are scaled such that the maximum absolute value maps to $7$. To save space, two 4-bit values are packed into a single `uint8` byte using bitwise operations:
  $$\text{packed} = (Q_1 + 8) \mid ((Q_2 + 8) \ll 4)$$
* **On-the-fly Dequantization**: Quantized weights are copied to the GPU and dequantized into temporary floating-point parameters immediately before executing the layer's forward pass, maintaining full compatibility with PyTorch/CUDA operators without requiring custom kernels.

### 2. Shard Manager (`LayerShardManager`)
The shard manager splits the sequential submodules of the RT-DETR model (indices 0 to 28) into individual layer files on disk.
* **Format**: Supports standard PyTorch pickling (`.pt`) and Hugging Face `.safetensors`.
* **State Preservation**: Saves both parameter state dicts and buffer states (BatchNorm stats, anchor grids). For layers without parameters (e.g., Concat, Upsample), it writes empty dictionaries to ensure sequence continuity.

### 3. Streaming Scheduler (`WeightStreamingScheduler`)
The scheduler coordinates the data-loading pipeline:
* **SSD -> RAM (L3 to L2)**: A background daemon thread reads shard files from storage into CPU RAM ahead of time.
* **RAM -> GPU (L2 to L1)**: Host CPU memory is *pinned* (`pin_memory()`), enabling asynchronous, non-blocking PCIe transfers to the GPU via PyTorch CUDA Streams.
* **Prefetching**: While layer $N$ is executing on the GPU, the scheduler asynchronously transfers layer $N+1$'s weights to the GPU and loads layer $N+2$ from SSD to CPU RAM.

### 4. VRAM Manager (`VRAMManager`)
Monitors GPU memory usage using PyTorch memory stats (`torch.cuda.memory_allocated()`, `torch.cuda.max_memory_allocated()`) and can check memory limits.

### 5. Model Wrapper & Monkey-patching (`StreamingRTDETR`)
Integrates the system with the existing training and validation pipelines by replacing the `_predict_once` method of the underlying `DetectionModel` instance.
* When a forward pass begins, it unloads all layer parameters/buffers from GPU VRAM, replacing them with empty CPU placeholders.
* During the sequential execution loop, it requests weights from the scheduler, dequantizes them on GPU, executes the layer's native forward pass, and immediately evicts the weights (restoring empty CPU placeholders) before executing the next layer.

---

## 🛰️ RT-DETR Layer Pipeline

The RT-DETR-L architecture consists of 29 submodules in the sequential list:

| Indices | Component | Layer Types | Memory Footprint (FP16) |
| :--- | :--- | :--- | :--- |
| **0 - 9** | Backbone (HGNetv2) | HGStem, HGBlock, DWConv | ~30.8 MB |
| **10** | Transformer Projector | Conv | ~1.0 MB |
| **11** | Transformer Encoder | AIFI | ~1.6 MB |
| **12 - 27** | Hybrid Encoder (FPN/PAN) | Conv, Upsample, Concat, RepC3 | ~18.3 MB |
| **28** | Transformer Decoder / Head | RTDETRDecoder | ~14.9 MB |

### Skip Connections and Caching
Because the hybrid encoder relies on skip connections (Concat layers pulling outputs from earlier layers, such as backbone layers 3 and 7), AIR-DETR maintains an activation cache list (`y`). These intermediate activations remain on the GPU, but the *weights* of the layers that produced them are safely evicted from VRAM immediately after execution.
