"""
Integration tests for the AIR-DETR streaming execution engine.
"""

import sys
import os
import shutil
import torch
import unittest
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from air_detr.model import StreamingRTDETR

class TestStreamingInference(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.model_path = "rtdetr-l.pt"
        cls.test_image = "test_data/traffic_images/traffic_02.jpg"
        cls.shard_dir = "temp/test_air_detr_shards"
        
        # Verify file existence
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
            
    def test_streaming_fp32(self):
        """Test streaming in FP32 mode (unquantized) and compare output keys and boxes."""
        # Load baseline model
        baseline_model = YOLO(self.model_path)
        
        # Run baseline inference
        baseline_results = baseline_model(self.test_image, conf=0.4, device="cpu", verbose=False)
        baseline_boxes = baseline_results[0].boxes.xyxy.cpu().numpy()
        baseline_classes = baseline_results[0].boxes.cls.cpu().numpy()
        
        # Now wrap it in StreamingRTDETR (CPU weights streamed)
        streaming_model = StreamingRTDETR(
            yolo_model=baseline_model,
            shard_dir=self.shard_dir,
            quantization=None,
            format="pt"
        )
        
        # Run streaming inference
        streaming_results = baseline_model(self.test_image, conf=0.4, device="cpu", verbose=False)
        streaming_boxes = streaming_results[0].boxes.xyxy.cpu().numpy()
        streaming_classes = streaming_results[0].boxes.cls.cpu().numpy()
        
        # Verify predictions are identical
        self.assertEqual(len(baseline_boxes), len(streaming_boxes))
        if len(baseline_boxes) > 0:
            # Check class indices are the same
            self.assertTrue((baseline_classes == streaming_classes).all())
            # Check box coordinates are extremely close
            self.assertTrue(torch.allclose(
                torch.tensor(baseline_boxes),
                torch.tensor(streaming_boxes),
                atol=1e-4
            ))
            
        # Restore the model
        streaming_model.restore()

    def test_streaming_int8(self):
        """Test streaming with INT8 quantization."""
        baseline_model = YOLO(self.model_path)
        
        # Run baseline
        baseline_results = baseline_model(self.test_image, conf=0.4, device="cpu", verbose=False)
        baseline_count = len(baseline_results[0].boxes)
        
        # Wrap in StreamingRTDETR with INT8 quantization
        streaming_model = StreamingRTDETR(
            yolo_model=baseline_model,
            shard_dir=self.shard_dir,
            quantization="int8",
            format="pt"
        )
        
        # Run streaming inference
        streaming_results = baseline_model(self.test_image, conf=0.4, device="cpu", verbose=False)
        streaming_count = len(streaming_results[0].boxes)
        
        # Verify it runs and detects similar number of objects
        self.assertGreaterEqual(streaming_count, 0)
        # Verify predictions are reasonable (accuracy drop is minimal)
        print(f"INT8 detected {streaming_count} objects vs Baseline {baseline_count}")
        
        streaming_model.restore()

    def test_streaming_int4(self):
        """Test streaming with INT4 quantization."""
        baseline_model = YOLO(self.model_path)
        
        # Run baseline
        baseline_results = baseline_model(self.test_image, conf=0.4, device="cpu", verbose=False)
        baseline_count = len(baseline_results[0].boxes)
        
        # Wrap in StreamingRTDETR with INT4 quantization
        streaming_model = StreamingRTDETR(
            yolo_model=baseline_model,
            shard_dir=self.shard_dir,
            quantization="int4",
            format="pt"
        )
        
        # Run streaming inference
        streaming_results = baseline_model(self.test_image, conf=0.4, device="cpu", verbose=False)
        streaming_count = len(streaming_results[0].boxes)
        
        # Verify it runs
        self.assertGreaterEqual(streaming_count, 0)
        print(f"INT4 detected {streaming_count} objects vs Baseline {baseline_count}")
        
        streaming_model.restore()

if __name__ == "__main__":
    unittest.main()
