"""
API Testing Script for TEDR
Tests all API endpoints to ensure they're working correctly.
"""
import requests
import json
import time
from pathlib import Path


class APITester:
    """Test suite for TEDR API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize tester with base URL."""
        self.base_url = base_url
        self.passed = 0
        self.failed = 0
    
    def print_test(self, name: str, passed: bool, message: str = ""):
        """Print test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if message:
            print(f"  └─ {message}")
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        print("\n--- Testing Root Endpoint ---")
        try:
            response = requests.get(f"{self.base_url}/")
            passed = response.status_code == 200
            data = response.json()
            
            self.print_test(
                "GET /",
                passed,
                f"Status: {response.status_code}, Message: {data.get('message', 'N/A')}"
            )
            return passed
        except Exception as e:
            self.print_test("GET /", False, f"Error: {str(e)}")
            return False
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        print("\n--- Testing Health Endpoint ---")
        try:
            response = requests.get(f"{self.base_url}/health")
            passed = response.status_code == 200
            data = response.json()
            
            self.print_test(
                "GET /health",
                passed,
                f"Status: {data.get('status', 'N/A')}, Model: {data.get('model', 'N/A')}"
            )
            return passed
        except Exception as e:
            self.print_test("GET /health", False, f"Error: {str(e)}")
            return False
    
    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        print("\n--- Testing Model Info Endpoint ---")
        try:
            response = requests.get(f"{self.base_url}/models/info")
            passed = response.status_code == 200
            data = response.json()
            
            self.print_test(
                "GET /models/info",
                passed,
                f"Model: {data.get('model_name', 'N/A')}, Threshold: {data.get('confidence_threshold', 'N/A')}"
            )
            return passed
        except Exception as e:
            self.print_test("GET /models/info", False, f"Error: {str(e)}")
            return False
    
    def test_detect_endpoint(self, image_path: str = None):
        """Test detection endpoint."""
        print("\n--- Testing Detection Endpoint ---")
        
        # Create a simple test image if none provided
        if image_path is None or not Path(image_path).exists():
            print("  └─ No test image provided, skipping detection test")
            self.print_test("POST /detect", None, "Skipped - no test image")
            return None
        
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                start_time = time.time()
                response = requests.post(f"{self.base_url}/detect", files=files)
                elapsed = time.time() - start_time
            
            passed = response.status_code == 200
            
            if passed:
                data = response.json()
                num_detections = data.get('num_detections', 0)
                processing_time = data.get('processing_time', 0)
                
                self.print_test(
                    "POST /detect",
                    passed,
                    f"Detections: {num_detections}, Processing: {processing_time}s, Total: {elapsed:.2f}s"
                )
                
                # Print detected objects
                if num_detections > 0:
                    print("  └─ Detected objects:")
                    for i, det in enumerate(data.get('detections', [])[:5], 1):
                        print(f"      {i}. {det['label']}: {det['confidence']:.2%}")
                    if num_detections > 5:
                        print(f"      ... and {num_detections - 5} more")
            else:
                self.print_test("POST /detect", passed, f"Status: {response.status_code}")
            
            return passed
        except Exception as e:
            self.print_test("POST /detect", False, f"Error: {str(e)}")
            return False
    
    def test_invalid_file_type(self):
        """Test detection with invalid file type."""
        print("\n--- Testing Invalid File Type ---")
        try:
            # Create a dummy text file
            test_file = Path("/tmp/test.txt")
            test_file.write_text("This is not an image")
            
            with open(test_file, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = requests.post(f"{self.base_url}/detect", files=files)
            
            # Should return 400 Bad Request
            passed = response.status_code == 400
            
            self.print_test(
                "POST /detect (invalid file)",
                passed,
                f"Status: {response.status_code} (expected 400)"
            )
            
            test_file.unlink()
            return passed
        except Exception as e:
            self.print_test("POST /detect (invalid file)", False, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self, image_path: str = None):
        """Run all API tests."""
        print("=" * 60)
        print("TEDR API Test Suite")
        print("=" * 60)
        print(f"Testing API at: {self.base_url}")
        
        # Check if API is accessible
        try:
            requests.get(self.base_url, timeout=5)
        except requests.exceptions.ConnectionError:
            print("\n✗ ERROR: Cannot connect to API server")
            print(f"  Make sure the server is running at {self.base_url}")
            print("  Start it with: python backend/main.py")
            return
        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            return
        
        # Run tests
        self.test_root_endpoint()
        self.test_health_endpoint()
        self.test_model_info_endpoint()
        self.test_detect_endpoint(image_path)
        self.test_invalid_file_type()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        
        if self.failed == 0:
            print("\n✓ All tests passed!")
        else:
            print(f"\n✗ {self.failed} test(s) failed")
        
        print("=" * 60)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TEDR API endpoints")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image for detection endpoint"
    )
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    tester.run_all_tests(args.image)


if __name__ == "__main__":
    main()
