#!/usr/bin/env python3
"""
Simple validation script to check if the TEDR (Transformer-based Object Detection 
for Indian Roads) system is properly configured.
Run this after installing dependencies to verify the setup.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")
    
    try:
        print("  ✓ Checking model package...")
        from model.config import Config
        from model.utils import calculate_iou, apply_nms, draw_boxes
        from model.detr_detector import DETRDetector
        print("  ✓ Model package OK")
        
        print("  ✓ Checking Flask app...")
        from app.app import create_app
        print("  ✓ Flask app OK")
        
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def check_structure():
    """Check if all required files and directories exist"""
    print("\nChecking file structure...")
    
    required_files = [
        'model/__init__.py',
        'model/config.py',
        'model/utils.py',
        'model/detr_detector.py',
        'app/__init__.py',
        'app/app.py',
        'app/templates/index.html',
        'app/static/css/style.css',
        'app/static/js/main.js',
        'run.py',
        'requirements.txt',
        'README.md',
        '.gitignore'
    ]
    
    all_exist = True
    for file in required_files:
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\nChecking dependencies...")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('transformers', 'Hugging Face Transformers'),
        ('PIL', 'Pillow'),
        ('flask', 'Flask'),
        ('flask_cors', 'Flask-CORS'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
    ]
    
    all_installed = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (not installed)")
            all_installed = False
    
    return all_installed

def main():
    print("=" * 60)
    print("TEDR System Validation")
    print("=" * 60)
    
    structure_ok = check_structure()
    deps_ok = check_dependencies()
    
    if deps_ok:
        imports_ok = check_imports()
    else:
        print("\nSkipping import checks (dependencies not installed)")
        print("\nTo install dependencies, run:")
        print("  pip install -r requirements.txt")
        imports_ok = False
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"  File structure: {'✓ OK' if structure_ok else '✗ Issues found'}")
    print(f"  Dependencies:   {'✓ OK' if deps_ok else '✗ Not installed'}")
    print(f"  Module imports: {'✓ OK' if imports_ok else '✗ Failed or skipped'}")
    
    if structure_ok and deps_ok and imports_ok:
        print("\n✓ System is ready! Run 'python run.py' to start the server.")
        return 0
    else:
        print("\n✗ System is not ready. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
