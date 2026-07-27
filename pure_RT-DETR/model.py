"""
pure_RT-DETR Model Definition.

Implements the official PekingU/lyuwenyu RT-DETR three-component PyTorch architecture 
(Backbone, Hybrid Encoder, and Transformer Decoder) and runs them with layer weight 
streaming and on-the-fly dequantization.
"""

import os
import gc
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from .quantization import QuantizationManager
from .shard_manager import LayerShardManager
from .scheduler import WeightStreamingScheduler
from .vram_manager import VRAMManager

def apply_weights_to_layer(layer: nn.Module, state_dict: Dict[str, Any], quantization_type: Optional[str] = None, device: str = "cuda"):
    """
    Applies weights from a state dict to the parameters and buffers of a layer.
    Dequantizes parameters on the fly if they were quantized.
    """
    tensors = state_dict.get("tensors", {})
    shapes = state_dict.get("shapes", {})
    
    # Apply parameters
    for name, param in layer.named_parameters():
        qweight_key = f"{name}_qweight"
        scale_key = f"{name}_scale"
        
        if qweight_key in tensors and scale_key in tensors:
            qweight = tensors[qweight_key]
            scale = tensors[scale_key]
            orig_shape = shapes[name]
            
            dequantized = QuantizationManager.dequantize_tensor(
                q_tensor=qweight,
                scale=scale,
                mode=quantization_type,
                original_shape=orig_shape
            )
            param.data = dequantized.to(device).to(param.dtype)
        elif name in tensors:
            param.data = tensors[name].to(device).to(param.dtype)
            
    # Apply buffers
    for name, buf in layer.named_buffers():
        if name in tensors:
            buf.data = tensors[name].to(device).to(buf.dtype)

def evict_weights_from_layer(layer: nn.Module):
    """
    Clears parameter and buffer storage on the GPU, replacing them
    with empty CPU tensors to immediately reclaim VRAM.
    """
    for param in layer.parameters():
        param.data = torch.empty(0, dtype=param.dtype, device="cpu")
    for buf in layer.buffers():
        buf.data = torch.empty(0, dtype=buf.dtype, device="cpu")
    for sub in layer.modules():
        if hasattr(sub, "shapes"):
            sub.shapes = None

class BackboneComponent(nn.Module):
    """
    Groups layers 0 to 9 representing the RT-DETR backbone (HGNetv2).
    """
    def __init__(self, modules: nn.ModuleList, scheduler: WeightStreamingScheduler, quantization: Optional[str], vram_manager: VRAMManager):
        super().__init__()
        self.modules_list = modules
        self.scheduler = scheduler
        self.quantization = quantization
        self.vram_manager = vram_manager
        
    def forward(self, x: torch.Tensor) -> List[Optional[torch.Tensor]]:
        device = x.device
        y = []
        for m in self.modules_list:
            next_idx = m.i + 1
            if next_idx < 29:
                self.scheduler.prefetch_ssd_to_ram(next_idx)
                self.scheduler.prefetch_ram_to_gpu(next_idx)
                
            state_dict = self.scheduler.get_layer_weights(m.i)
            apply_weights_to_layer(m, state_dict, self.quantization, device=device)
            
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                
            x = m(x)
            y.append(x)
            
            evict_weights_from_layer(m)
            self.scheduler.evict_layer(m.i)
            self.vram_manager.check_limit()
            
        return y

class EncoderComponent(nn.Module):
    """
    Groups layers 10 to 27 representing the RT-DETR Hybrid Encoder (AIFI and CCFM).
    """
    def __init__(self, modules: nn.ModuleList, scheduler: WeightStreamingScheduler, quantization: Optional[str], vram_manager: VRAMManager):
        super().__init__()
        self.modules_list = modules
        self.scheduler = scheduler
        self.quantization = quantization
        self.vram_manager = vram_manager
        
    def forward(self, y_backbone: List[Optional[torch.Tensor]]) -> List[torch.Tensor]:
        y = list(y_backbone)
        x = y[-1]
        device = x.device
        
        for m in self.modules_list:
            next_idx = m.i + 1
            if next_idx < 29:
                self.scheduler.prefetch_ssd_to_ram(next_idx)
                self.scheduler.prefetch_ram_to_gpu(next_idx)
                
            state_dict = self.scheduler.get_layer_weights(m.i)
            apply_weights_to_layer(m, state_dict, self.quantization, device=device)
            
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                
            x = m(x)
            y.append(x)
            
            evict_weights_from_layer(m)
            self.scheduler.evict_layer(m.i)
            self.vram_manager.check_limit()
            
        # The decoder takes multiscale inputs from layers 21, 24, and 27
        return [y[21], y[24], y[27]]

class DecoderComponent(nn.Module):
    """
    Groups layer 28 representing the Transformer Decoder.
    """
    def __init__(self, decoder_module: nn.Module, scheduler: WeightStreamingScheduler, quantization: Optional[str], vram_manager: VRAMManager):
        super().__init__()
        self.decoder_module = decoder_module
        self.scheduler = scheduler
        self.quantization = quantization
        self.vram_manager = vram_manager
        
    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        device = encoder_features[0].device
        m = self.decoder_module
        
        state_dict = self.scheduler.get_layer_weights(m.i)
        apply_weights_to_layer(m, state_dict, self.quantization, device=device)
        
        x = m(encoder_features)
        
        evict_weights_from_layer(m)
        self.scheduler.evict_layer(m.i)
        self.vram_manager.check_limit()
        
        return x

class RTDETR(nn.Module):
    """
    Original PekingU/lyuwenyu style RT-DETR architecture wrapping Backbone, Encoder, and Decoder components.
    """
    def __init__(
        self,
        model_path: str = "rtdetr-l.pt",
        quantization: Optional[str] = None,
        shard_dir: str = "temp/pure_rtdetr_shards",
        format: str = "safetensors",
        vram_limit_mb: Optional[float] = None,
        dataset_yaml: Optional[str] = None
    ):
        super().__init__()
        self.model_path = model_path
        self.quantization = quantization
        self.shard_dir = shard_dir
        self.format = format
        
        # 1. Load the original PyTorch DetectionModel from the checkpoint
        print(f"Loading native RT-DETR model from {model_path}...")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.original_model = ckpt["model"]
        self.names = getattr(self.original_model, "names", None)
        
        # Save original forward passes
        self._original_forward = self.original_model.forward
        self._original_predict_once = getattr(self.original_model, "_predict_once", None)
        
        # 2. Setup shard manager, scheduler, and vram manager
        self.vram_manager = VRAMManager(limit_mb=vram_limit_mb)
        self.shard_manager = LayerShardManager(shard_dir=shard_dir, format=format)
        
        # Shard the model if metadata doesn't exist
        metadata_path = os.path.join(self.shard_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            print("Layer shards not found. Preparing shards on disk...")
            self.shard_manager.shard_model(
                self.original_model,
                quantization=quantization,
                dataset_yaml=dataset_yaml,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
        # 3. Initialize scheduler
        self.scheduler = WeightStreamingScheduler(
            shard_manager=self.shard_manager,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_ram_slots=5,
            max_gpu_slots=2
        )
        
        # 4. Partition the flat 29 layers into official three components
        self.backbone = BackboneComponent(self.original_model.model[0:10], self.scheduler, self.quantization, self.vram_manager)
        self.encoder = EncoderComponent(self.original_model.model[10:28], self.scheduler, self.quantization, self.vram_manager)
        self.decoder = DecoderComponent(self.original_model.model[28], self.scheduler, self.quantization, self.vram_manager)
        
        # 5. Evict all weights from the base model layers to free GPU memory
        print("Offloading initial model parameters to CPU RAM/SSD...")
        for m in self.original_model.model:
            evict_weights_from_layer(m)
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            VRAMManager.reset_peak()
            
        # 6. Monkey-patch the original model's forward to redirect to this three-component streaming wrapper
        self.original_model.forward = lambda x, *args, **kwargs: self.forward(x)
        if hasattr(self.original_model, "_predict_once"):
            self.original_model._predict_once = lambda x, *args, **kwargs: self.forward(x)
            
        print(f"pure_RT-DETR initialized with active quantization: {self.quantization}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Original RT-DETR architecture sequential forward execution."""
        # 1. Forward through backbone (HGNetv2)
        y_backbone = self.backbone(x)
        # 2. Forward through hybrid encoder (AIFI and CCFM)
        encoder_features = self.encoder(y_backbone)
        # 3. Forward through transformer decoder (RTDETRTransformer)
        out = self.decoder(encoder_features)
        return out
        
    def restore(self):
        """Restore original parameters and forward pass of the model."""
        self.scheduler.shutdown()
        
        print(f"Reloading original weights and model from {self.model_path}...")
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        original_model = ckpt["model"]
        
        # Restore parameters and buffer shapes/data in-place
        for name, param in self.original_model.named_parameters():
            orig_param = dict(original_model.named_parameters())[name]
            param.data = orig_param.data.to(param.device)
            
        for name, buf in self.original_model.named_buffers():
            orig_buf = dict(original_model.named_buffers())[name]
            buf.data = orig_buf.data.to(buf.device)
            
        # Restore original forward methods
        self.original_model.forward = self._original_forward
        if self._original_predict_once:
            self.original_model._predict_once = self._original_predict_once
            
        print("Restored original model forward pass.")
