"""
VRAM Manager for pure_RT-DETR.
"""

import subprocess
import torch
from typing import Dict, Any, Optional

class VRAMManager:
    """
    Monitors and manages GPU memory allocations during inference.
    """
    
    def __init__(self, limit_mb: Optional[float] = None):
        """
        Initialize the VRAM manager.
        """
        self.limit_bytes = limit_mb * 1024 * 1024 if limit_mb else None
        
    @staticmethod
    def get_allocated_mb() -> float:
        """Get currently allocated VRAM by PyTorch in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 * 1024)
        
    @staticmethod
    def get_max_allocated_mb() -> float:
        """Get peak VRAM allocated by PyTorch since last reset in MB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
        
    @staticmethod
    def reset_peak():
        """Reset the peak memory tracking in PyTorch."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    @staticmethod
    def get_system_gpu_info() -> Optional[Dict[str, float]]:
        """Queries nvidia-smi for total and used system VRAM in MB."""
        try:
            cmd = "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits"
            res = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
            vram_used, vram_total = [float(x.strip()) for x in res.split(',')]
            return {
                "system_vram_used_mb": vram_used,
                "system_vram_total_mb": vram_total
            }
        except Exception:
            return None
            
    def check_limit(self) -> bool:
        """
        Checks if current allocation exceeds the specified limit.
        """
        if self.limit_bytes is None:
            return True
        current = torch.cuda.memory_allocated()
        if current > self.limit_bytes:
            print(f"[VRAMManager] WARNING: VRAM allocation ({current / (1024*1024):.2f} MB) "
                  f"exceeded limit ({self.limit_bytes / (1024*1024):.2f} MB)")
            return False
        return True
