# AIR-DETR Mathematical Foundations

This document provides the mathematical derivations, formulas, and performance models used in **AIR-DETR (Adaptive Inference RT-DETR)**.

---

## 🔢 1. Weight Quantization (Symmetric Channel-wise)

For a weight tensor $W \in \mathbb{R}^{C_{out} \times C_{in} \times K_1 \times K_2}$ representing a convolutional or linear layer:

### Quantization Scale Factor
Quantization is performed channel-wise across the output channels (dimension 0) to preserve fine-grained activation resolution.
For each channel $c \in \{0, \dots, C_{out}-1\}$, the scale factor $s_c$ is:

$$s_c = \frac{\max_{i,j,k} |W_{c,i,j,k}|}{2^{B-1} - 1}$$

Where:
* $B$ is the bitwidth (e.g., $B = 8$ for INT8, $B = 4$ for INT4).
* The denominator is $127$ for INT8 and $7$ for INT4.
* Scale factors are clamped to a minimum value of $10^{-8}$ to prevent division by zero:
  $$s_c = \max\left(s_c, 10^{-8}\right)$$

### Forward Quantization
The floating-point weight $w$ is mapped to the quantized integer representation $q$:

$$q = \text{round}\left(\frac{w}{s_c}\right)$$

Values are clamped to the representable range of the target bitwidth:
* **INT8**: $q \in [-128, 127]$
* **INT4**: $q \in [-8, 7]$

### Reconstruction (Dequantization)
During inference, weights are reconstructed on the GPU:

$$\tilde{w} = q \cdot s_c$$

Where $\tilde{w} \approx w$ is the reconstructed floating-point parameter.

---

## 🗜️ 2. Compression Ratio (CR)

The compression ratio measures the reduction in weight storage size from the original 16-bit floating point format (FP16) or 32-bit (FP32) to the quantized integer format:

$$\text{CR} = \frac{\text{Original Size (Bytes)}}{\text{Quantized Size (Bytes)}}$$

For a model with $N$ parameters:
* **FP32 size**: $4N$ bytes.
* **FP16 size**: $2N$ bytes.
* **INT8 size**: $1N$ bytes (quantized weights) + $4 \times C_{out}$ bytes (FP32 scales) $\approx 1N$ bytes.
* **INT4 size**: $0.5N$ bytes (packed 4-bit weights) + $4 \times C_{out}$ bytes (FP32 scales) $\approx 0.5N$ bytes.

### Theoretical Compression Limits (vs FP16):
* **INT8**: $\text{CR} \approx 2.0\times$
* **INT4**: $\text{CR} \approx 4.0\times$

---

## ⏱️ 3. Execution & Prefetching Performance Models

### Baseline (Non-streaming) Execution Time
The total time to run a forward pass without streaming is simply the sum of compute times for each layer:

$$T_{\text{baseline}} = \sum_{i=1}^{L} T_{\text{compute}}(i)$$

### Sequential (No Prefetching) Streaming Time
If weight sharding is used without prefetching, each layer must be loaded from SSD, transferred to the GPU over PCIe, and computed sequentially:

$$T_{\text{seq}} = \sum_{i=1}^{L} \left( T_{\text{IO}}(i) + T_{\text{transfer}}(i) + T_{\text{compute}}(i) \right)$$

Where:
* $T_{\text{IO}}(i)$: Time to read layer $i$ shard from SSD to CPU RAM.
* $T_{\text{transfer}}(i)$: Time to copy layer $i$ parameters from CPU RAM to GPU over PCIe.
* $T_{\text{compute}}(i)$: GPU execution time of layer $i$'s forward pass.

### Asynchronous Prefetching (Overlapped) Streaming Time
AIR-DETR implements overlapped I/O and computation. While layer $i$ is executing on the GPU, layer $i+1$ is copied to the GPU (PCIe), and layer $i+2$ is read from SSD to RAM.

The total execution time for a prefetched pipeline is modeled as:

$$T_{\text{prefetch}} = T_{\text{IO}}(1) + T_{\text{transfer}}(1) + \sum_{i=1}^{L} \max\left( T_{\text{IO}}(i+1) + T_{\text{transfer}}(i+1), \, T_{\text{compute}}(i) \right)$$

When the system is I/O-bound (i.e., loading weights takes longer than compute):

$$T_{\text{prefetch}} \approx \sum_{i=1}^{L} \left( T_{\text{IO}}(i) + T_{\text{transfer}}(i) \right)$$

When the system is compute-bound (i.e., execution is slower than loading):

$$T_{\text{prefetch}} \approx \sum_{i=1}^{L} T_{\text{compute}}(i)$$

---

## 💾 4. VRAM Memory Model

The peak VRAM usage ($M_{\text{peak}}$) of the model during a forward pass is composed of three components:

$$M_{\text{peak}} = M_{\text{CUDA}} + M_{\text{activations}} + M_{\text{active\_weights}} + M_{\text{cache}}$$

Where:
1. $M_{\text{CUDA}}$: Static baseline memory allocated by PyTorch for CUDA context initialization (~800–1000 MB).
2. $M_{\text{activations}}$: Memory required to store intermediate layer activations during the forward pass.
3. $M_{\text{active\_weights}}$: The size of the weights of the *single largest layer* active at any moment:
   $$M_{\text{active\_weights}} = \max_{i} \left( \text{Size}(\text{Layer}_i) \right)$$
   For RT-DETR-L, the largest layer is Layer 28 (decoder/head, 7.47M parameters $\approx$ 14.9 MB in FP16).
4. $M_{\text{cache}}$: Memory for cached skip-connection activations ($y[3]$, $y[7]$, etc.).
