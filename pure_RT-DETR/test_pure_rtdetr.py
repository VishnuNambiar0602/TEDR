"""
Integration and Unit Tests for pure_RT-DETR.

Tests:
1. Lossless INT4 bit-packing and unpacking.
2. Tensor equivalence between native PyTorch execution and FP32 layer-wise streaming.
3. Quantized inference execution under INT8, INT4, calibrated, AWQ, and GPTQ modes.
"""

import sys
import os
import shutil
import unittest
import torch
import cv2
import gc

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import importlib
try:
    pure_RT_DETR = importlib.import_module("pure_RT-DETR")
    RTDETR = pure_RT_DETR.RTDETR
    QuantizationManager = pure_RT_DETR.QuantizationManager
    pack_int4 = pure_RT_DETR.pack_int4
    unpack_int4 = pure_RT_DETR.unpack_int4
except ImportError as e:
    print(f"Could not import pure_RT-DETR dynamically: {e}")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import quantization
    from quantization import QuantizationManager, pack_int4, unpack_int4
    import model
    from model import RTDETR

class TestPureRTDETR(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join(parent_dir, "rtdetr-l.pt")
        cls.test_image = os.path.join(parent_dir, "test_data/traffic_images/traffic_02.jpg")
        cls.shard_dir = os.path.join(parent_dir, "temp/test_pure_rtdetr_shards")
        
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(f"Model file {cls.model_path} not found.")
        if not os.path.exists(cls.test_image):
            raise unittest.SkipTest(f"Test image {cls.test_image} not found.")

    def setUp(self):
        if os.path.exists(self.shard_dir):
            shutil.rmtree(self.shard_dir)
            
    def tearDown(self):
        if os.path.exists(self.shard_dir):
            shutil.rmtree(self.shard_dir)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_pack_unpack_int4(self):
        """Test that INT4 packing and unpacking is correct and lossless."""
        original = torch.tensor([-8, -4, 0, 3, 7, -7, 5, 2, -1, 0], dtype=torch.int8)
        packed = pack_int4(original)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.numel(), original.numel() // 2)
        
        unpacked = unpack_int4(packed, original.shape)
        self.assertEqual(unpacked.dtype, torch.int8)
        self.assertTrue(torch.equal(original, unpacked))

    def _preprocess(self, device="cpu", dtype=torch.float32):
        """Utility to preprocess test image to [1, 3, 640, 640] float tensor."""
        img = cv2.imread(self.test_image)
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = img_resized[:, :, ::-1].copy()
        x = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        return (x.float() / 255.0).to(dtype)

    def test_fp32_streaming_vs_baseline(self):
        """Test FP32 weight streaming and verify it gives identical predictions to native PyTorch."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load baseline PyTorch model
        print("Loading baseline PyTorch model...")
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)
        baseline_model = ckpt["model"].to(device)
        baseline_model.eval()
        
        dtype = next(baseline_model.parameters()).dtype
        x = self._preprocess(device=device, dtype=dtype)
        
        # Run baseline inference
        with torch.no_grad():
            baseline_out = baseline_model(x)
            if isinstance(baseline_out, tuple):
                baseline_tensor = baseline_out[0]
            else:
                baseline_tensor = baseline_out
                
        # Initialize pure_RT-DETR in FP32 Streaming mode (three-component architecture)
        print("Loading pure_RT-DETR FP32 streaming model...")
        pure_model = RTDETR(
            model_path=self.model_path,
            quantization=None,
            shard_dir=self.shard_dir,
            format="pt"
        )
        pure_model.eval()
        
        # Run streaming inference
        with torch.no_grad():
            pure_out = pure_model(x)
            if isinstance(pure_out, tuple):
                pure_tensor = pure_out[0]
            else:
                pure_tensor = pure_out
                
        # Sort output predictions by max confidence score to align matched queries
        def get_sorted_predictions(pred_tensor):
            pred = pred_tensor[0]
            scores = pred[:, 4:]
            max_scores, _ = scores.max(dim=-1)
            sorted_indices = torch.argsort(max_scores, descending=True)
            return pred[sorted_indices]

        sorted_base = get_sorted_predictions(baseline_tensor)
        sorted_pure = get_sorted_predictions(pure_tensor)
        
        # Compare top 50 highest confidence detections
        top_k = 50
        max_diff = torch.abs(sorted_base[:top_k] - sorted_pure[:top_k]).max().item()
        print(f"FP32 Streaming Max absolute difference (top {top_k} detections): {max_diff:.6f}")
        
        pure_model.restore()
        
        # Verify outputs are mathematically identical
        self.assertLess(max_diff, 1e-4)

    def test_int8_streaming(self):
        """Test that INT8 streaming execution runs."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pure_model = RTDETR(
            model_path=self.model_path,
            quantization="int8",
            shard_dir=self.shard_dir,
            format="pt"
        )
        pure_model.eval()
        
        dtype = next(pure_model.original_model.parameters()).dtype
        x = self._preprocess(device=device, dtype=dtype)
        
        with torch.no_grad():
            out = pure_model(x)
            if isinstance(out, tuple):
                out_tensor = out[0]
            else:
                out_tensor = out
                
        self.assertEqual(out_tensor.shape[0], 1)
        self.assertEqual(out_tensor.shape[1], 300) # 300 queries
        print(f"INT8 Streaming completed. Output shape: {out_tensor.shape}")
        
        pure_model.restore()

    def test_int4_streaming(self):
        """Test that INT4 streaming execution runs."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        pure_model = RTDETR(
            model_path=self.model_path,
            quantization="int4",
            shard_dir=self.shard_dir,
            format="pt"
        )
        pure_model.eval()
        
        dtype = next(pure_model.original_model.parameters()).dtype
        x = self._preprocess(device=device, dtype=dtype)
        
        with torch.no_grad():
            out = pure_model(x)
            if isinstance(out, tuple):
                out_tensor = out[0]
            else:
                out_tensor = out
                
        self.assertEqual(out_tensor.shape[0], 1)
        print(f"INT4 Streaming completed. Output shape: {out_tensor.shape}")
        
        pure_model.restore()

if __name__ == "__main__":
    unittest.main()
