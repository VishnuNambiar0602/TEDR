"""
Unit tests for the AIR-DETR calibration engine and advanced quantization modes.
"""

import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import unittest
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from air_detr.calibration import CalibrationManager, ActivationHook
from air_detr.quantization import QuantizationManager
from air_detr.shard_manager import LayerShardManager

class MockYOLO(nn.Module):
    def __init__(self):
        super().__init__()
        # To match named_modules and model.model structure in RT-DETR
        self.model = nn.Sequential()
        layer = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 4, kernel_size=1)
        )
        self.model.add_module("0", layer)
        
    def __call__(self, img, verbose=False, device="cpu"):
        # Mock forward pass
        x = torch.randn(1, 3, 32, 32)
        for layer in self.model:
            x = layer(x)
        return x

class TestCalibration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = "temp/test_calib_data"
        os.makedirs(os.path.join(self.temp_dir, "images/val"), exist_ok=True)
        
        # Create a dummy image
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(self.temp_dir, "images/val/dummy1.jpg"), img)
        cv2.imwrite(os.path.join(self.temp_dir, "images/val/dummy2.jpg"), img)
        
        # Create dataset.yaml
        self.yaml_path = os.path.join(self.temp_dir, "dataset.yaml")
        data = {
            "path": os.path.abspath(self.temp_dir).replace("\\", "/"),
            "val": "images/val"
        }
        with open(self.yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)
            
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def test_activation_hook_basic(self):
        """Test ActivationHook registers and processes activation tensors."""
        hook = ActivationHook(collect_hessian=True)
        x = torch.randn(2, 4, 8, 8)
        module = nn.Conv2d(4, 8, kernel_size=3)
        
        hook.hook_fn(module, (x,), x)
        
        self.assertEqual(hook.count, 1)
        self.assertEqual(hook.scale_sum.shape, (4,))
        self.assertEqual(hook.raw_hessian.shape, (4, 4))
        self.assertEqual(hook.nsamples, 2 * 8 * 8)
        
    def test_calibration_manager_flow(self):
        """Test CalibrationManager runs calibration on dummy dataset and gets stats."""
        model = MockYOLO()
        calibrator = CalibrationManager(model, collect_hessian=True)
        
        calibrator.run_calibration(self.yaml_path, num_images=2, device="cpu")
        stats = calibrator.get_stats()
        
        self.assertIn("0.0", stats)
        self.assertIn("scale", stats["0.0"])
        self.assertIn("hessian", stats["0.0"])
        
    def test_awq_gptq_quantization_with_stats(self):
        """Test applying AWQ and GPTQ quantization using captured stats."""
        model = MockYOLO()
        calibrator = CalibrationManager(model, collect_hessian=True)
        calibrator.run_calibration(self.yaml_path, num_images=2, device="cpu")
        stats = calibrator.get_stats()
        
        # Conv2d weights shape: (8, 3, 3, 3) -> out_channels=8, in_channels=3
        W = model.model[0][0].weight.data
        act_scale = stats["0.0"]["scale"]
        hessian = stats["0.0"]["hessian"]
        
        # 1. AWQ
        qweight_awq, scale_awq = QuantizationManager.apply_awq(W, act_scale, mode="int4")
        self.assertEqual(qweight_awq.dtype, torch.uint8)
        self.assertEqual(scale_awq.shape, (8, 3, 1, 1))
        
        dequant_awq = QuantizationManager.dequantize_tensor(qweight_awq, scale_awq, mode="int4", original_shape=W.shape)
        self.assertEqual(dequant_awq.shape, W.shape)
        
        # 2. GPTQ
        qweight_gptq, scale_gptq = QuantizationManager.apply_gptq(W, hessian, mode="int4")
        self.assertEqual(qweight_gptq.dtype, torch.uint8)
        self.assertEqual(scale_gptq.shape, (8, 1, 1, 1))
        
        dequant_gptq = QuantizationManager.dequantize_tensor(qweight_gptq, scale_gptq, mode="gptq", original_shape=W.shape)
        self.assertEqual(dequant_gptq.shape, W.shape)

    def test_sharding_advanced_modes(self):
        """Test sharding with all advanced quantization modes."""
        for mode in ["calibrated_int8", "calibrated_int4", "awq", "gptq"]:
            model = MockYOLO()
            mode_dir = os.path.join(self.temp_dir, f"shards_{mode}")
            manager = LayerShardManager(shard_dir=mode_dir, format="pt")
            
            meta = manager.shard_model(model.model, quantization=mode, dataset_yaml=self.yaml_path, device="cpu")
            self.assertEqual(meta["quantization"], mode)
            
            shard_data = manager.load_shard_cpu(0)
            self.assertIn("0.weight_qweight", shard_data["tensors"])
            self.assertIn("0.weight_scale", shard_data["tensors"])

if __name__ == "__main__":
    unittest.main()
