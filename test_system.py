"""Simple test to verify TEDR system components."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from backend.config import config
        print("✓ Backend config imported")
        
        from models.detr_model import DETRModel
        print("✓ DETR model imported")
        
        from utils.preprocessing import resize_image
        print("✓ Preprocessing utils imported")
        
        from utils.visualization import draw_bounding_boxes
        print("✓ Visualization utils imported")
        
        print("\n✓ All imports successful!")
        return True
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from backend.config import config
        
        print(f"Model name: {config.model_name}")
        print(f"Confidence threshold: {config.confidence_threshold}")
        print(f"Image size: {config.image_size}")
        print(f"Device: {config.device}")
        print(f"API host: {config.api_host}")
        print(f"API port: {config.api_port}")
        
        print("\n✓ Configuration loaded successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Configuration test failed: {e}")
        return False

def test_model_structure():
    """Test that model structure is correct."""
    print("\nTesting model structure...")
    
    try:
        from models.detr_model import DETRModel
        
        # Check that class exists and has required methods
        assert hasattr(DETRModel, 'detect')
        assert hasattr(DETRModel, 'detect_batch')
        assert hasattr(DETRModel, 'save_model')
        assert hasattr(DETRModel, 'load_model')
        
        print("✓ DETRModel has all required methods")
        print("\n✓ Model structure test passed!")
        return True
    except Exception as e:
        print(f"\n✗ Model structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("TEDR System Component Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_model_structure
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 60)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed! System is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
