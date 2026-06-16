"""
AIR-DETR: Adaptive Inference RT-DETR using AirLLM-style Weight Streaming and Multi-Level Quantization.

v1 Core Modules:
- StreamingRTDETR: Model wrapper patching forward pass
- QuantizationManager: INT8/INT4 PTQ engine
- LayerShardManager: Layer-wise model splitter and saver
- WeightStreamingScheduler: SSD -> RAM -> GPU prefetching queue
- VRAMManager: Memory resource manager and limit checker
"""

from air_detr.quantization import QuantizationManager, pack_int4, unpack_int4
from air_detr.shard_manager import LayerShardManager
from air_detr.scheduler import WeightStreamingScheduler
from air_detr.vram_manager import VRAMManager
from air_detr.model import StreamingRTDETR, apply_weights_to_layer, evict_weights_from_layer

__version__ = "1.0.0"
__all__ = [
    "StreamingRTDETR",
    "QuantizationManager",
    "LayerShardManager",
    "WeightStreamingScheduler",
    "VRAMManager",
    "apply_weights_to_layer",
    "evict_weights_from_layer",
    "pack_int4",
    "unpack_int4",
]
