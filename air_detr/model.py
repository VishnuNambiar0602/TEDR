"""
AIR-DETR Streaming Model Wrapper.

Wraps standard RT-DETR model and replaces the forward pass with
an adaptive, layer-wise weight streaming and quantized execution pipeline.
"""

import os
import types
import gc
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from air_detr.quantization import QuantizationManager
from air_detr.shard_manager import LayerShardManager
from air_detr.scheduler import WeightStreamingScheduler
from air_detr.vram_manager import VRAMManager

def apply_weights_to_layer(layer: nn.Module, state_dict: Dict[str, Any], quantization_type: Optional[str] = None, device: str = "cuda"):
    """
    Applies weights from a state dict to the parameters and buffers of a layer.
    Dequantizes parameters on the fly if they were quantized.
    
    Args:
        layer: The nn.Module layer to load weights into.
        state_dict: The state dict containing loaded weights (and shapes).
        quantization_type: Quantization mode ('int8', 'int4', or None).
        device: Target execution device.
    """
    tensors = state_dict["tensors"]
    shapes = state_dict["shapes"]
    
    # 1. Apply parameters
    for name, param in layer.named_parameters():
        qweight_key = f"{name}_qweight"
        scale_key = f"{name}_scale"
        
        if qweight_key in tensors and scale_key in tensors:
            # Param was quantized, dequantize on the fly
            qweight = tensors[qweight_key]
            scale = tensors[scale_key]
            orig_shape = shapes[name]
            
            # Dequantize back to param dtype on GPU
            dequantized = QuantizationManager.dequantize_tensor(
                q_tensor=qweight,
                scale=scale,
                mode=quantization_type,
                original_shape=orig_shape
            )
            # Make sure it's on the correct device and matches param type
            param.data = dequantized.to(device).to(param.dtype)
        elif name in tensors:
            # Parameter was not quantized (bias, bn weight, etc.)
            param.data = tensors[name].to(device).to(param.dtype)
            
    # 2. Apply buffers (BatchNorm running stats, anchors, etc.)
    for name, buf in layer.named_buffers():
        if name in tensors:
            buf.data = tensors[name].to(device).to(buf.dtype)

def evict_weights_from_layer(layer: nn.Module):
    """
    Clears parameter and buffer storage on the GPU, replacing them
    with empty CPU tensors to immediately reclaim VRAM.
    
    Args:
        layer: The nn.Module layer to evict.
    """
    for param in layer.parameters():
        param.data = torch.empty(0, device="cpu")
    for buf in layer.buffers():
        buf.data = torch.empty(0, device="cpu")
    for sub in layer.modules():
        if hasattr(sub, "shapes"):
            sub.shapes = None

class StreamingRTDETR:
    """
    Wrapper for RT-DETR model to enable AirLLM-style layer weight streaming
    and multi-level quantization during inference.
    """
    
    def __init__(
        self,
        yolo_model,
        shard_dir: str = "temp/air_detr_shards",
        quantization: Optional[str] = None,
        format: str = "safetensors",
        vram_limit_mb: Optional[float] = None,
        dataset_yaml: Optional[str] = None
    ):
        """
        Initialize the streaming wrapper.
        
        Args:
            yolo_model: Ultralytics YOLO/RTDETR instance.
            shard_dir: Directory to save/load layer shards.
            quantization: Quantization mode ('int8', 'int4', or None).
            format: Storage format ('safetensors' or 'pt').
            vram_limit_mb: Optional hard VRAM limit.
            dataset_yaml: Optional path to dataset configuration file.
        """
        self.yolo_model = yolo_model
        # Fuse Conv and BatchNorm layers in-place immediately before sharding or offloading
        if hasattr(self.yolo_model, "fuse"):
            print("Fusing Conv and BatchNorm layers for optimized streaming...")
            try:
                self.yolo_model.fuse()
            except Exception as e:
                print(f"Model fusion skipped: {e}")
        # The underlying PyTorch DetectionModel
        self.model = yolo_model.model
        self.device = next(self.model.parameters()).device
        
        self.quantization = quantization
        self.shard_dir = shard_dir
        self.format = format
        
        self.vram_manager = VRAMManager(limit_mb=vram_limit_mb)
        self.shard_manager = LayerShardManager(shard_dir=shard_dir, format=format)
        
        # Check if shards already exist, if not shard the model
        metadata_path = os.path.join(self.shard_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            print("Shards not found. Preparing layer shards...")
            # Run sharding. Note: sharded weights are quantized if quantization is specified
            self.shard_manager.shard_model(
                self.yolo_model,
                quantization=quantization,
                dataset_yaml=dataset_yaml,
                device=str(self.device)
            )
            
        # Initialize scheduler
        self.scheduler = WeightStreamingScheduler(
            shard_manager=self.shard_manager,
            device=str(self.device),
            max_ram_slots=5,
            max_gpu_slots=2
        )
        
        # Save original forward method so we can unpatch if needed
        self.original_predict_once = self.model._predict_once
        
        # Save original fuse method to prevent fusion crashes on empty parameters
        self.original_fuse = getattr(self.model, "fuse", None)
        if self.original_fuse:
            self.model.fuse = types.MethodType(lambda self_instance, *args, **kwargs: self_instance, self.model)
            
        # Patch the model forward pass
        self._patch_forward()
        
        # Evict all weights from the base model layers to free GPU memory
        print("Offloading initial model parameters to CPU RAM/SSD...")
        for layer in self.model.model:
            evict_weights_from_layer(layer)
            
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            VRAMManager.reset_peak()
            
        print(f"Streaming RT-DETR initialized. Active quantization: {self.quantization}")
        print(f"Current VRAM allocated: {VRAMManager.get_allocated_mb():.2f} MB")
        
    def _patch_forward(self):
        """Monkeys patches the _predict_once method of the model."""
        scheduler = self.scheduler
        quantization_type = self.quantization
        original_predict_once = self.original_predict_once
        vram_manager = self.vram_manager
        
        def streaming_predict_once(model_instance, x, profile=False, visualize=False, embed=None):
            # Track outputs for skip connections (Concat layers)
            y, dt, embeddings = [], [], []
            embed = frozenset(embed) if embed is not None else {-1}
            max_idx = max(embed)
            
            # Determine execution device dynamically from inputs
            device = x.device if isinstance(x, torch.Tensor) else (x[0].device if isinstance(x, list) and len(x) > 0 else "cuda")
            
            # Reset peak VRAM tracker for this forward pass to measure clean metrics
            VRAMManager.reset_peak()
            
            for m in model_instance.model:
                # 1. Prefetch next layer to overlap IO and Compute
                next_idx = m.i + 1
                if next_idx < len(model_instance.model):
                    scheduler.prefetch_ssd_to_ram(next_idx)
                    scheduler.prefetch_ram_to_gpu(next_idx)
                    
                # 2. Get weights for the current layer (will load/wait if not prefetched)
                # Note: if a layer has no parameters (like Concat), get_layer_weights returns empty structures
                state_dict = scheduler.get_layer_weights(m.i)
                
                # 3. Apply weights to the module (dequantizing on-the-fly)
                apply_weights_to_layer(m, state_dict, quantization_type, device=device)
                
                # 4. Input preparation for the layer
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                    
                # 5. Run forward pass of the layer
                if profile:
                    model_instance._profile_one_layer(m, x, dt)
                x = m(x)
                
                # 6. Save layer output for future skip connections if it's in the save list
                y.append(x if m.i in model_instance.save else None)
                
                if visualize:
                    feature_visualization(x, m.type, m.i, save_dir=visualize)
                    
                if m.i in embed:
                    embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                    if m.i == max_idx:
                        # Evict current layer weights before return
                        evict_weights_from_layer(m)
                        scheduler.evict_layer(m.i)
                        return torch.unbind(torch.cat(embeddings, 1), dim=0)
                        
                # 7. Evict current layer weights immediately after execution
                evict_weights_from_layer(m)
                scheduler.evict_layer(m.i)
                
                # Check VRAM limit
                vram_manager.check_limit()
                
            return x
            
        # Bind the method to the model instance
        self.model._predict_once = types.MethodType(streaming_predict_once, self.model)
        
    def restore(self):
        """Restore the original, non-streaming forward pass."""
        self.model._predict_once = self.original_predict_once
        if getattr(self, "original_fuse", None):
            self.model.fuse = self.original_fuse
        self.scheduler.shutdown()
        print("Restored original model forward pass.")
