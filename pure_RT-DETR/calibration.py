"""
Calibration Engine for pure_RT-DETR.
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
        
        self.raw_hessian = None
        self.nsamples = 0
        
    def hook_fn(self, module: nn.Module, inp: Tuple[torch.Tensor], out: torch.Tensor):
        if not inp or not isinstance(inp[0], torch.Tensor):
            return
            
        x = inp[0].detach()
        
        if x.ndim == 4:
            channel_scales = x.abs().mean(dim=(0, 2, 3))
        else:
            flat_x = x.view(-1, x.shape[-1])
            channel_scales = flat_x.abs().mean(dim=0)
            
        if self.scale_sum is None:
            self.scale_sum = channel_scales
        else:
            self.scale_sum += channel_scales
        self.count += 1
        
        if self.collect_hessian:
            if x.ndim == 4:
                flat = x.permute(1, 0, 2, 3).flatten(1)
            else:
                flat = x.view(-1, x.shape[-1]).t()
                
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
        # Support both original model container and components
        pytorch_model = self.model.model if hasattr(self.model, "model") else self.model
        for name, module in pytorch_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hook = ActivationHook(collect_hessian=self.collect_hessian)
                self.hooks[name] = hook
                handle = module.register_forward_hook(hook.hook_fn)
                self.handles.append(handle)
                
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        
    def run_calibration(self, dataset_yaml: str, num_images: int = 8, device: str = "cuda"):
        import yaml
        
        with open(dataset_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
            
        val_path = os.path.join(data_cfg['path'], data_cfg['val'])
        self.register_hooks()
        
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
        
        try:
            with torch.no_grad():
                for img_path in images:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img_resized = cv2.resize(img, (640, 640))
                    img_rgb = img_resized[:, :, ::-1].copy()
                    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                    tensor = tensor / 255.0
                    self.model(tensor)
        finally:
            self.remove_hooks()
            
        print("✓ Calibration run complete.")
        
    def get_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        stats = {}
        for name, hook in self.hooks.items():
            stats[name] = {}
            if hook.scale_sum is not None and hook.count > 0:
                stats[name]["scale"] = hook.scale_sum / hook.count
            if hook.raw_hessian is not None:
                stats[name]["hessian"] = hook.raw_hessian
                stats[name]["nsamples"] = torch.tensor(hook.nsamples)
        return stats
