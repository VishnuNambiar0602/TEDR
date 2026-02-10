"""FastAPI backend for TEDR object detection system."""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import sys

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import config
from backend.inference import get_detector

# Create FastAPI app
app = FastAPI(
    title="TEDR - Object Detection API",
    description="Transformer-based Object Detection for Indian Roads",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/")
async def root():
    """Root endpoint - redirect to frontend."""
    return {
        "message": "TEDR Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint.
    
    Returns:
        Health status of the API
    """
    return {
        "status": "healthy",
        "model": config.model_name,
        "device": config.device,
        "confidence_threshold": config.confidence_threshold
    }


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Detect objects in uploaded image.
    
    Args:
        file: Uploaded image file (JPEG, PNG)
        
    Returns:
        JSON response with detection results
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file."
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Get detector and process image
        detector = get_detector()
        results = detector.process_image(image_bytes)
        
        return JSONResponse(content=results)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/models/info")
async def model_info():
    """Get information about the loaded model.
    
    Returns:
        Model configuration and metadata
    """
    return {
        "model_name": config.model_name,
        "confidence_threshold": config.confidence_threshold,
        "image_size": config.image_size,
        "device": config.device,
        "num_classes": 91,  # COCO dataset classes
        "supported_objects": [
            "person", "bicycle", "car", "motorcycle", "bus", "truck",
            "traffic light", "stop sign", "cow", "elephant", "zebra",
            "and 80+ more COCO classes"
        ],
        "indian_road_focus": [
            "auto_rickshaw (via fine-tuning)",
            "car", "motorcycle", "truck", "bus",
            "person", "bicycle", "cow", "traffic light"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting TEDR Object Detection API...")
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Confidence threshold: {config.confidence_threshold}")
    
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level="info"
    )
