"""
Unit tests for the AIR-DETR quantization engine.
"""

import sys
import os
import torch
import unittest

# Add workspace root to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from air_detr.quantization import QuantizationManager, pack_int4, unpack_int4

class TestQuantization(unittest.TestCase):
    
    def test_pack_unpack_int4(self):
        """Test that packing and unpacking of INT4 weights is lossless and correct."""
        # Create a toy tensor with values in the INT4 range [-8, 7]
        original = torch.tensor([-8, -4, 0, 3, 7, -7, 5, 2, -1, 0], dtype=torch.int8)
        
        # Pack
        packed = pack_int4(original)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.numel(), original.numel() // 2)
        
        # Unpack
        unpacked = unpack_int4(packed, original.shape)
        self.assertEqual(unpacked.dtype, torch.int8)
        self.assertTrue(torch.equal(original, unpacked))
        
    def test_pack_unpack_odd_elements(self):
        """Test packing/unpacking on odd-sized tensors."""
        original = torch.tensor([-8, 0, 7], dtype=torch.int8)
        packed = pack_int4(original)
        unpacked = unpack_int4(packed, original.shape)
        self.assertTrue(torch.equal(original, unpacked))
        
    def test_int8_quantization(self):
        """Test INT8 Post-Training Quantization (PTQ) scaling and reconstruction."""
        # Create a mock weight tensor (out_channels=2, in_channels=4)
        weight = torch.tensor([
            [1.5, -2.0, 0.5, 0.1],
            [-0.8, 3.5, -1.2, 0.0]
        ], dtype=torch.float32)
        
        # Quantize
        qweight, scale = QuantizationManager.quantize_tensor(weight, mode="int8")
        self.assertEqual(qweight.dtype, torch.int8)
        self.assertEqual(scale.shape, (2, 1))
        
        # Dequantize
        dequantized = QuantizationManager.dequantize_tensor(qweight, scale, mode="int8", original_shape=weight.shape)
        
        # Assert reconstruction is close to original
        error = torch.abs(weight - dequantized).max().item()
        # Max error should be less than scale / 2
        # For channel 1, max is 2.0. Scale = 2.0 / 127 = 0.0157. Error should be <= 0.0078
        self.assertLess(error, 0.02)
        
    def test_int4_quantization(self):
        """Test INT4 Post-Training Quantization (PTQ) scaling and reconstruction."""
        weight = torch.tensor([
            [1.5, -2.0, 0.5, 0.1],
            [-0.8, 3.5, -1.2, 0.0]
        ], dtype=torch.float32)
        
        # Quantize
        qweight, scale = QuantizationManager.quantize_tensor(weight, mode="int4")
        self.assertEqual(qweight.dtype, torch.uint8) # Packed weights
        self.assertEqual(scale.shape, (2, 1))
        
        # Dequantize
        dequantized = QuantizationManager.dequantize_tensor(qweight, scale, mode="int4", original_shape=weight.shape)
        
        # Check that error is reasonable (max value / 7 is the scale, error should be <= scale / 2)
        # For channel 1, max is 2.0, scale = 2.0 / 7 = 0.285. Error should be <= 0.143
        error = torch.abs(weight - dequantized).max().item()
        self.assertLess(error, 0.3)

if __name__ == "__main__":
    unittest.main()
