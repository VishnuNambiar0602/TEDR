"""
Unit tests for the AIR-DETR shard manager.
"""

import sys
import os
import shutil
import torch
import torch.nn as nn
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from air_detr.shard_manager import LayerShardManager

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple model with submodules matching a sequence
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )

class TestShardManager(unittest.TestCase):
    
    def setUp(self):
        self.test_dir = "temp/test_shards"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_sharding_pt(self):
        """Test model sharding in PyTorch (.pt) format without quantization."""
        model = ToyModel()
        manager = LayerShardManager(shard_dir=self.test_dir, format="pt")
        
        # Shard the model
        meta = manager.shard_model(model)
        
        # Verify metadata
        self.assertEqual(meta["format"], "pt")
        self.assertIsNone(meta["quantization"])
        self.assertIn("0", meta["layers"])
        self.assertIn("1", meta["layers"])
        self.assertIn("3", meta["layers"])
        self.assertIn("2", meta["layers"])
        self.assertEqual(meta["layers"]["2"]["param_count"], 0) # ReLU has no parameters
        
        # Verify files on disk
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "metadata.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "layer_000.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "layer_001.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "layer_002.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "layer_003.pt")))
        
        # Load and verify content
        shard_data = manager.load_shard_cpu(0)
        self.assertIn("weight", shard_data["tensors"])
        self.assertIn("bias", shard_data["tensors"])
        self.assertEqual(shard_data["tensors"]["weight"].shape, torch.Size([16, 3, 3, 3]))
        
    def test_sharding_safetensors(self):
        """Test model sharding in safetensors format."""
        model = ToyModel()
        manager = LayerShardManager(shard_dir=self.test_dir, format="safetensors")
        
        # If safetensors is not installed, it falls back to pt, which we should handle
        meta = manager.shard_model(model)
        fmt = meta["format"]
        
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, f"layer_000.{fmt}")))
        shard_data = manager.load_shard_cpu(0)
        self.assertIn("weight", shard_data["tensors"])
        self.assertEqual(shard_data["tensors"]["weight"].shape, torch.Size([16, 3, 3, 3]))

if __name__ == "__main__":
    unittest.main()
