"""
Streaming Scheduler for pure_RT-DETR.
"""

import threading
import queue
import torch
from typing import Dict, Any, Optional
from .shard_manager import LayerShardManager

class WeightStreamingScheduler:
    """
    Schedules and coordinates the loading of layer weights from SSD to RAM, 
    and RAM to GPU, with prefetching and eviction.
    """
    
    def __init__(
        self,
        shard_manager: LayerShardManager,
        device: str = "cuda",
        max_ram_slots: int = 5,
        max_gpu_slots: int = 2
    ):
        self.shard_manager = shard_manager
        self.device = device
        self.max_ram_slots = max_ram_slots
        self.max_gpu_slots = max_gpu_slots
        
        self.ram_cache: Dict[int, Dict[str, Any]] = {}
        self.gpu_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        
        self.lock = threading.Lock()
        
        self.prefetch_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
    def _worker_loop(self):
        while not self.stop_event.is_set():
            try:
                layer_idx = self.prefetch_queue.get(timeout=0.5)
                if layer_idx is None:
                    break
            except queue.Empty:
                continue
                
            with self.lock:
                in_cache = layer_idx in self.ram_cache
                
            if not in_cache:
                try:
                    shard_data = self.shard_manager.load_shard_cpu(layer_idx)
                    
                    pinned_tensors = {}
                    for k, v in shard_data["tensors"].items():
                        if isinstance(v, torch.Tensor):
                            pinned_tensors[k] = v.pin_memory()
                        else:
                            pinned_tensors[k] = v
                    shard_data["tensors"] = pinned_tensors
                    
                    with self.lock:
                        if len(self.ram_cache) >= self.max_ram_slots:
                            evict_idx = next(iter(self.ram_cache.keys()))
                            del self.ram_cache[evict_idx]
                        self.ram_cache[layer_idx] = shard_data
                except Exception as e:
                    print(f"[Scheduler] Error prefetching layer {layer_idx} from SSD: {e}")
                    
            self.prefetch_queue.task_done()
            
    def shutdown(self):
        self.stop_event.set()
        self.prefetch_queue.put(None)
        self.worker_thread.join()
        
    def prefetch_ssd_to_ram(self, layer_idx: int):
        with self.lock:
            in_ram = layer_idx in self.ram_cache
            in_gpu = layer_idx in self.gpu_cache
            
        if not in_ram and not in_gpu:
            self.prefetch_queue.put(layer_idx)
            
    def prefetch_ram_to_gpu(self, layer_idx: int):
        with self.lock:
            if layer_idx in self.gpu_cache:
                return
                
        shard_data = None
        with self.lock:
            if layer_idx in self.ram_cache:
                shard_data = self.ram_cache[layer_idx]
                
        if shard_data is None:
            return
            
        gpu_tensors = {}
        tensors = shard_data["tensors"]
        
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                gpu_tensors[k] = v.to(self.device, non_blocking=True)
            else:
                gpu_tensors[k] = v
                
        with self.lock:
            if len(self.gpu_cache) >= self.max_gpu_slots:
                evict_idx = next(iter(self.gpu_cache.keys()))
                del self.gpu_cache[evict_idx]
            self.gpu_cache[layer_idx] = {
                "tensors": gpu_tensors,
                "shapes": shard_data["shapes"]
            }
            
    def get_layer_weights(self, layer_idx: int) -> Dict[str, Any]:
        with self.lock:
            if layer_idx in self.gpu_cache:
                return self.gpu_cache[layer_idx]
                
        shard_data = None
        with self.lock:
            if layer_idx in self.ram_cache:
                shard_data = self.ram_cache[layer_idx]
                
        if shard_data is None:
            try:
                shard_data = self.shard_manager.load_shard_cpu(layer_idx)
                with self.lock:
                    if len(self.ram_cache) >= self.max_ram_slots:
                        evict_idx = next(iter(self.ram_cache.keys()))
                        del self.ram_cache[evict_idx]
                    self.ram_cache[layer_idx] = shard_data
            except Exception:
                return {"tensors": {}, "shapes": {}}
                
        gpu_tensors = {}
        for k, v in shard_data["tensors"].items():
            if isinstance(v, torch.Tensor):
                gpu_tensors[k] = v.to(self.device)
            else:
                gpu_tensors[k] = v
                
        state_dict = {
            "tensors": gpu_tensors,
            "shapes": shard_data["shapes"]
        }
        
        with self.lock:
            if len(self.gpu_cache) >= self.max_gpu_slots:
                evict_idx = next(iter(self.gpu_cache.keys()))
                del self.gpu_cache[evict_idx]
            self.gpu_cache[layer_idx] = state_dict
            
        return state_dict
        
    def evict_layer(self, layer_idx: int):
        with self.lock:
            if layer_idx in self.gpu_cache:
                del self.gpu_cache[layer_idx]
