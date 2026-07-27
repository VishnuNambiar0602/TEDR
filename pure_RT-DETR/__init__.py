"""
pure_RT-DETR: Native PyTorch PekingU/lyuwenyu style RT-DETR supporting layer-by-layer weight streaming (AIRLLM-style) and multi-level quantization.
"""

from .quantization import QuantizationManager, pack_int4, unpack_int4
from .shard_manager import LayerShardManager
from .scheduler import WeightStreamingScheduler
from .vram_manager import VRAMManager
from .calibration import CalibrationManager
from .model import RTDETR, apply_weights_to_layer, evict_weights_from_layer

__version__ = "1.0.0"
__all__ = [
    "RTDETR",
    "QuantizationManager",
    "LayerShardManager",
    "WeightStreamingScheduler",
    "VRAMManager",
    "CalibrationManager",
    "apply_weights_to_layer",
    "evict_weights_from_layer",
    "pack_int4",
    "unpack_int4",
]
