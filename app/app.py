"""
Flask Application for DETR Object Detection
"""

import os
import io
import base64
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import traceback

from model.detr_detector import DETRDetector
from model.config import Config


# Global detector instance (lazy loaded)
detector = None


def get_detector():
    """Get or create detector instance"""
    global detector
    if detector is None:
        detector = DETRDetector()
    return detector


def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = Config.MAX_IMAGE_SIZE
    app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
    
    # Enable CORS
    CORS(app)
    
    # Ensure upload folder exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    @app.route('/')
    def index():
        """Serve main UI page"""
        return render_template('index.html')
    
    @app.route('/api/detect', methods=['POST'])
    def detect_objects():
        """
        API endpoint for object detection
        
        Accepts: multipart/form-data with image file
        Returns: JSON with detections, annotated image (base64), and statistics
        """
        try:
            # Check if image file is present
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            
            # Check if file is selected
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Check file extension
            if not Config.allowed_file(file.filename):
                return jsonify({
                    'error': f'Invalid file type. Allowed types: {", ".join(Config.ALLOWED_EXTENSIONS)}'
                }), 400
            
            # Read image
            try:
                image = Image.open(file.stream).convert('RGB')
            except Exception as e:
                return jsonify({'error': f'Invalid image file: {str(e)}'}), 400
            
            # Get detector and perform detection
            det = get_detector()
            result = det.detect(image)
            
            # Convert annotated image to base64
            img_buffer = io.BytesIO()
            result['annotated_image'].save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Prepare response
            response = {
                'success': True,
                'detections': result['detections'],
                'statistics': result['statistics'],
                'annotated_image': f'data:image/png;base64,{img_base64}'
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            # Log error
            print("Error during detection:")
            traceback.print_exc()
            
            return jsonify({
                'error': f'Server error: {str(e)}'
            }), 500
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'model': Config.MODEL_NAME,
            'device': Config.DEVICE
        }), 200
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file too large error"""
        max_size_mb = Config.MAX_IMAGE_SIZE / (1024 * 1024)
        return jsonify({
            'error': f'File too large. Maximum size: {max_size_mb:.1f}MB'
        }), 413
    
    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle internal server errors"""
        return jsonify({
            'error': 'Internal server error'
        }), 500
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
