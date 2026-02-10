"""Configuration management for TEDR backend."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Application configuration class."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
                
        return value
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.get('model.name', 'facebook/detr-resnet-50')
    
    @property
    def confidence_threshold(self) -> float:
        """Get confidence threshold."""
        return self.get('model.confidence_threshold', 0.7)
    
    @property
    def image_size(self) -> int:
        """Get image size for model input."""
        return self.get('model.image_size', 800)
    
    @property
    def device(self) -> str:
        """Get device for model inference."""
        return self.get('model.device', 'cuda')
    
    @property
    def coco_classes(self) -> Dict[int, str]:
        """Get COCO class mapping."""
        return self.get('classes.coco_classes', {})
    
    @property
    def api_host(self) -> str:
        """Get API host."""
        return self.get('api.host', '0.0.0.0')
    
    @property
    def api_port(self) -> int:
        """Get API port."""
        return self.get('api.port', 8000)
    
    @property
    def cors_origins(self) -> list:
        """Get CORS origins."""
        return self.get('api.cors_origins', ['*'])


# Global configuration instance
config = Config()
