"""
TEDR Model Package
Transformer-based object detection for Indian roads
"""

from .detr_detector import DETRDetector
from .config import Config

__all__ = ['DETRDetector', 'Config']
