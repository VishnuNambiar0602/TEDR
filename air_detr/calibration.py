"""
Calibration Engine for AIR-DETR.

Handles:
- Capturing activations using PyTorch forward hooks
- Accumulating activation scales for AWQ
- Accumulating input covariance matrices (Hessians) for GPTQ
- Running calibration passes on representative image samples
"""

import os
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional

class ActivationHook:
    """
    Hook to capture activation stats during calibration.
    """
    def __init__(self, collect_hessian: bool = False):
        self.collect_hessian = collect_hessian
        self.scale_sum = None
        self.count = 0
        
        # For Hessian accumulation (GPTQ)
        self.raw_hessian = None
        self.nsamples = 0
        
    def hook_fn(self, module: nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor):
        """Forward hook function to record input activations."""
        if not inp or not isinstance(inp[0], torch.Tensor):
            return
            
        x = inp[0].detach()
        
        # 1. Collect channel-wise scale stats (AWQ)
        if x.ndim == 4: # Conv2d: (B, C, H, W)
            # Average absolute value per channel
            channel_scales = x.abs().mean(dim=(0, 2, 3))
        else: # Linear: (B, ..., C)
            flat_x = x.view(-1, x.shape[-1])
            channel_scales = flat_x.abs().mean(dim=0)
            
        if self.scale_sum is None:
            self.scale_sum = channel_scales
        else:
            self.scale_sum += channel_scales
        self.count += 1
        
        # 2. Collect covariance / Hessian stats (GPTQ)
        if self.collect_hessian:
            if x.ndim == 4: # Conv2d: (B, C, H, W)
                # Reshape to (C, B*H*W)
                flat = x.permute(1, 0, 2, 3).flatten(1)
            else: # Linear: (B, ..., C)
                flat = x.view(-1, x.shape[-1]).t()
                
            # Accumulate H = X * X^T
            h_add = torch.matmul(flat, flat.t())
            if self.raw_hessian is None:
                self.raw_hessian = h_add.to(torch.float64)
            else:
                self.raw_hessian += h_add.to(torch.float64)
            self.nsamples += flat.shape[1]

class CalibrationManager:
    """
    Manages registering hooks, running calibration passes, and retrieving stats.
    """
    def __init__(self, model: nn.Module, collect_hessian: bool = False):
        self.model = model
        self.collect_hessian = collect_hessian
        self.hooks: Dict[str, ActivationHook] = {}
        self.handles = []
        
    def register_hooks(self):
        """Registers hooks on all Conv2d and Linear layers recursively."""
        pytorch_model = self.model.model if hasattr(self.model, "model") else self.model
        for name, module in pytorch_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = ActivationHook(collect_hessian=self.collect_hessian)
                self.hooks[name] = hook
                handle = module.register_forward_hook(hook.hook_fn)
                self.handles.append(handle)
                
    def remove_hooks(self):
        """Removes all active hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        
    def run_calibration(self, dataset_yaml: str, num_images: int = 8, device: str = "cuda"):
        """
        Runs forward passes on a small set of images to collect activation stats.
        
        Args:
            dataset_yaml: Path to the dataset configuration file.
            num_images: Number of images to process for calibration.
            device: Execution device.
        """
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.utils import DEFAULT_CFG
        import yaml
        
        # Load dataset config
        with open(dataset_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
            
        val_path = os.path.join(data_cfg['path'], data_cfg['val'])
        
        # Register hooks
        self.register_hooks()
        
        # Retrieve validation images directly
        images = []
        for root, _, files in os.walk(val_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(os.path.join(root, file))
                    if len(images) >= num_images:
                        break
            if len(images) >= num_images:
                break
                
        print(f"Running calibration pass on {len(images)} images...")
        pytorch_model = self.model.model if hasattr(self.model, "model") else self.model
        pytorch_model.eval()
        
        # We temporarily restore the original forward pass if monkey-patched
        is_patched = False
        original_forward = None
        if hasattr(pytorch_model, "_predict_once") and hasattr(pytorch_model, "original_predict_once"):
            original_forward = pytorch_model._predict_once
            pytorch_model._predict_once = pytorch_model.original_predict_once
            is_patched = True
            
        try:
            with torch.no_grad():
                for img_path in images:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    # Run preprocessor and model forward
                    if hasattr(self.model, "predict"):
                        # It is a YOLO/RTDETR wrapper
                        self.model.predict(img, verbose=False, device=device)
                    elif hasattr(self.model, "__call__"):
                        try:
                            # Try calling with YOLO style arguments
                            self.model(img, verbose=False, device=device)
                        except TypeError:
                            # Raw PyTorch module
                            h, w, c = img.shape
                            tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
                            tensor = tensor / 255.0
                            try:
                                self.model(tensor)
                            except Exception:
                                # Fallback if model has fixed input channel/shape
                                self.model(torch.randn(1, c, h, w, device=device))
        finally:
            # Restore patch if it was patched
            if is_patched:
                pytorch_model._predict_once = original_forward
            self.remove_hooks()
            
        print("✓ Calibration run complete.")
        
    def get_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Returns the collected statistics for each module."""
        stats = {}
        for name, hook in self.hooks.items():
            stats[name] = {}
            if hook.scale_sum is not None and hook.count > 0:
                stats[name]["scale"] = hook.scale_sum / hook.count
            if hook.raw_hessian is not None:
                stats[name]["hessian"] = hook.raw_hessian
                stats[name]["nsamples"] = torch.tensor(hook.nsamples)
        return stats
