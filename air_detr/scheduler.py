"""
Streaming Scheduler for AIR-DETR.

Implements:
- Asynchronous prefetching from SSD to RAM using a background worker thread
- Asynchronous transfer from RAM to GPU (using pinned memory and non-blocking copies)
- Eviction of layer weights from GPU VRAM after execution
"""

import threading
import queue
import torch
from typing import Dict, Any, Optional
from air_detr.shard_manager import LayerShardManager

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
        """
        Initialize the scheduler.
        
        Args:
            shard_manager: The shard manager instance.
            device: Target GPU device ('cuda' or 'cuda:0').
            max_ram_slots: Max number of layers to keep in RAM cache.
            max_gpu_slots: Max number of layers to keep in GPU cache.
        """
        self.shard_manager = shard_manager
        self.device = device
        self.max_ram_slots = max_ram_slots
        self.max_gpu_slots = max_gpu_slots
        
        # Caches
        self.ram_cache: Dict[int, Dict[str, Any]] = {}  # layer_idx -> CPU shard data
        self.gpu_cache: Dict[int, Dict[str, torch.Tensor]] = {}  # layer_idx -> GPU state_dict
        
        self.lock = threading.Lock()
        
        # Threading queue for SSD -> RAM prefetching
        self.prefetch_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
    def _worker_loop(self):
        """Background thread loop to load shards from SSD to RAM."""
        while not self.stop_event.is_set():
            try:
                # Wait for a layer prefetch request with a timeout to allow stop checks
                layer_idx = self.prefetch_queue.get(timeout=0.5)
                if layer_idx is None:
                    break
            except queue.Empty:
                continue
                
            # Check if layer is already in RAM cache
            with self.lock:
                in_cache = layer_idx in self.ram_cache
                
            if not in_cache:
                try:
                    shard_data = self.shard_manager.load_shard_cpu(layer_idx)
                    
                    # Pin the tensors in CPU RAM to speed up CPU -> GPU PCIe transfer
                    pinned_tensors = {}
                    for k, v in shard_data["tensors"].items():
                        if isinstance(v, torch.Tensor):
                            pinned_tensors[k] = v.pin_memory()
                        else:
                            pinned_tensors[k] = v
                    shard_data["tensors"] = pinned_tensors
                    
                    with self.lock:
                        # Manage RAM cache capacity (FIFO eviction)
                        if len(self.ram_cache) >= self.max_ram_slots:
                            # Find a slot that is not the immediately needed ones
                            evict_idx = next(iter(self.ram_cache.keys()))
                            del self.ram_cache[evict_idx]
                        self.ram_cache[layer_idx] = shard_data
                except Exception as e:
                    print(f"[Scheduler] Error prefetching layer {layer_idx} from SSD: {e}")
                    
            self.prefetch_queue.task_done()
            
    def shutdown(self):
        """Shut down the background prefetching thread."""
        self.stop_event.set()
        self.prefetch_queue.put(None)
        self.worker_thread.join()
        
    def prefetch_ssd_to_ram(self, layer_idx: int):
        """Queue a layer to be loaded from SSD to RAM."""
        with self.lock:
            if layer_idx in self.ram_cache:
                return
        self.prefetch_queue.put(layer_idx)
        
    def prefetch_ram_to_gpu(self, layer_idx: int):
        """Copies layer tensors from CPU RAM to GPU memory asynchronously."""
        with self.lock:
            # Already in GPU cache?
            if layer_idx in self.gpu_cache:
                return
                
            # Is it in RAM cache? If not, load synchronously
            if layer_idx not in self.ram_cache:
                try:
                    shard_data = self.shard_manager.load_shard_cpu(layer_idx)
                    self.ram_cache[layer_idx] = shard_data
                except Exception as e:
                    print(f"[Scheduler] Sync load failed for layer {layer_idx}: {e}")
                    return
                    
            shard_data = self.ram_cache[layer_idx]
            
            # Copy to GPU using non_blocking=True (which is active since CPU memory is pinned)
            gpu_tensors = {}
            for k, v in shard_data["tensors"].items():
                if isinstance(v, torch.Tensor):
                    gpu_tensors[k] = v.to(self.device, non_blocking=True)
                else:
                    gpu_tensors[k] = v
                    
            # Manage GPU cache capacity
            if len(self.gpu_cache) >= self.max_gpu_slots:
                # Evict oldest GPU layer
                evict_idx = next(iter(self.gpu_cache.keys()))
                del self.gpu_cache[evict_idx]
                
            self.gpu_cache[layer_idx] = gpu_tensors
            
    def get_layer_weights(self, layer_idx: int) -> Dict[str, Any]:
        """
        Retrieves the GPU-loaded weights for a layer.
        
        Args:
            layer_idx: Index of the layer.
            
        Returns:
            A dictionary containing tensors (and shapes metadata).
        """
        # Ensure it is moved to GPU
        self.prefetch_ram_to_gpu(layer_idx)
        
        with self.lock:
            gpu_tensors = self.gpu_cache.get(layer_idx)
            
            # Fallback if asynchronous transfer failed or was not finished
            if gpu_tensors is None:
                # Synchronous transfer fallback
                if layer_idx not in self.ram_cache:
                    shard_data = self.shard_manager.load_shard_cpu(layer_idx)
                    self.ram_cache[layer_idx] = shard_data
                shard_data = self.ram_cache[layer_idx]
                gpu_tensors = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in shard_data["tensors"].items()
                }
                self.gpu_cache[layer_idx] = gpu_tensors
                
            # Include shapes metadata from RAM cache
            shapes = self.ram_cache[layer_idx]["shapes"]
            
        return {
            "tensors": gpu_tensors,
            "shapes": shapes
        }
        
    def evict_layer(self, layer_idx: int):
        """Evict weights of a layer from GPU and RAM caches to free up memory."""
        with self.lock:
            if layer_idx in self.gpu_cache:
                del self.gpu_cache[layer_idx]
            if layer_idx in self.ram_cache:
                del self.ram_cache[layer_idx]
