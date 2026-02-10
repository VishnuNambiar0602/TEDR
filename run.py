"""
TEDR - Transformer-based Object Detection for Indian Roads
Application Entry Point
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.app import create_app

if __name__ == '__main__':
    app = create_app()
    
    print("=" * 60)
    print("TEDR - Transformer-based Object Detection for Indian Roads")
    print("=" * 60)
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
