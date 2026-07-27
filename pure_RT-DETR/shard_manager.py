"""
Shard Manager for pure_RT-DETR.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .quantization import QuantizationManager

try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

class LayerShardManager:
    """
    Manages sharding, calibration, and loading of model layer weights.
    """
    
    def __init__(self, shard_dir: str, format: str = "safetensors"):
        self.shard_dir = shard_dir
        self.format = format if (format == "safetensors" and SAFETENSORS_AVAILABLE) else "pt"
        os.makedirs(self.shard_dir, exist_ok=True)
        
    def get_shard_path(self, layer_idx: int) -> str:
        ext = "safetensors" if self.format == "safetensors" else "pt"
        return os.path.join(self.shard_dir, f"layer_{layer_idx:03d}.{ext}")
        
    def shard_model(
        self,
        model: torch.nn.Module,
        quantization: Optional[str] = None,
        dataset_yaml: Optional[str] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        print(f"Sharding model layers to {self.shard_dir} (format: {self.format}, quantization: {quantization})...")
        
        stats = None
        if quantization in ("calibrated_int8", "calibrated_int4", "awq", "gptq"):
            if not dataset_yaml:
                raise ValueError(f"dataset_yaml is required for calibration mode: {quantization}")
            if not os.path.exists(dataset_yaml):
                raise FileNotFoundError(f"dataset_yaml file not found: {dataset_yaml}")
                
            from .calibration import CalibrationManager
            collect_hess = (quantization == "gptq")
            calibrator = CalibrationManager(model, collect_hessian=collect_hess)
            calibrator.run_calibration(dataset_yaml, num_images=8, device=device)
            stats = calibrator.get_stats()
            
        shards_metadata = {
            "quantization": quantization,
            "format": self.format,
            "layers": {}
        }
        
        # Support various model structures (flat sequential list or container object)
        pytorch_model = model.model if hasattr(model, "model") else model
        layers = pytorch_model.model if hasattr(pytorch_model, "model") else pytorch_model
        
        for i, layer in enumerate(layers):
            state_dict = layer.state_dict()
            shard_path = self.get_shard_path(i)
            save_dict = {}
            shapes_meta = {}
            quantized_params = set()
            
            if quantization in ("calibrated_int8", "calibrated_int4", "awq", "gptq") and stats is not None:
                for sub_name, sub_mod in layer.named_modules():
                    if isinstance(sub_mod, (nn.Conv2d, nn.Linear)):
                        abs_name = f"model.{i}.{sub_name}" if sub_name else f"model.{i}"
                        param_name = f"{sub_name}.weight" if sub_name else "weight"
                        
                        if param_name in state_dict:
                            W = state_dict[param_name]
                            if W.ndim >= 2 and W.dtype in (torch.float32, torch.float16, torch.bfloat16):
                                q_mode = "int8" if "int8" in quantization else "int4"
                                
                                if "calibrated" in quantization:
                                    qweight, scale = QuantizationManager.quantize_mse_grid_search(W, q_mode)
                                elif quantization == "awq":
                                    act_scale = stats.get(abs_name, {}).get("scale")
                                    if act_scale is None:
                                        act_scale = stats.get(abs_name.replace("model.", "", 1), {}).get("scale")
                                    if act_scale is not None:
                                        qweight, scale = QuantizationManager.apply_awq(W, act_scale, mode="int4")
                                    else:
                                        qweight, scale = QuantizationManager.quantize_mse_grid_search(W, "int4")
                                elif quantization == "gptq":
                                    hessian = stats.get(abs_name, {}).get("hessian")
                                    if hessian is None:
                                        hessian = stats.get(abs_name.replace("model.", "", 1), {}).get("hessian")
                                    if hessian is not None:
                                        qweight, scale = QuantizationManager.apply_gptq(W, hessian, mode="int4")
                                    else:
                                        qweight, scale = QuantizationManager.quantize_mse_grid_search(W, "int4")
                                        
                                save_dict[f"{param_name}_qweight"] = qweight
                                save_dict[f"{param_name}_scale"] = scale
                                shapes_meta[param_name] = list(W.shape)
                                quantized_params.add(param_name)
                                
            for name, param in state_dict.items():
                if name in quantized_params:
                    continue
                    
                if (quantization in ("int8", "int4") and 
                    "weight" in name and 
                    param.ndim >= 2 and 
                    param.dtype in (torch.float32, torch.float16, torch.bfloat16)):
                    
                    qweight, scale = QuantizationManager.quantize_tensor(param, quantization)
                    save_dict[f"{name}_qweight"] = qweight
                    save_dict[f"{name}_scale"] = scale
                    shapes_meta[name] = list(param.shape)
                else:
                    save_dict[name] = param
            
            if self.format == "safetensors":
                metadata = {"shapes": json.dumps(shapes_meta)}
                save_dict_contiguous = {k: v.contiguous() for k, v in save_dict.items()}
                safetensors.torch.save_file(save_dict_contiguous, shard_path, metadata=metadata)
            else:
                save_data = {
                    "tensors": save_dict,
                    "shapes": shapes_meta
                }
                torch.save(save_data, shard_path)
                
            param_count = sum(p.numel() for p in layer.parameters())
            shards_metadata["layers"][str(i)] = {
                "type": type(layer).__name__,
                "param_count": param_count,
                "shard_file": os.path.basename(shard_path)
            }
            
        with open(os.path.join(self.shard_dir, "metadata.json"), "w") as f:
            json.dump(shards_metadata, f, indent=2)
            
        print("[SUCCESS] Model sharded successfully!")
        return shards_metadata

    def load_shard_cpu(self, layer_idx: int) -> Dict[str, Any]:
        shard_path = self.get_shard_path(layer_idx)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
            
        if self.format == "safetensors":
            tensors = safetensors.torch.load_file(shard_path, device="cpu")
            with open(shard_path, "rb") as f:
                header_size = int.from_bytes(f.read(8), "little")
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes.decode("utf-8"))
                metadata = header.get("__metadata__", {})
                shapes_str = metadata.get("shapes", "{}")
                shapes = json.loads(shapes_str)
            return {
                "tensors": tensors,
                "shapes": {k: torch.Size(v) for k, v in shapes.items()}
            }
        else:
            save_data = torch.load(shard_path, map_location="cpu")
            return {
                "tensors": save_data["tensors"],
                "shapes": {k: torch.Size(v) for k, v in save_data["shapes"].items()}
            }
