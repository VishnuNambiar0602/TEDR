from fastapi import FastAPI, UploadFile, File, Request, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import base64
import uuid
import asyncio
import numpy as np
from analyzer import TrafficAnalyzer

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize analyzer
analyzer = TrafficAnalyzer()

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)
os.makedirs("temp/processed", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        temp_path = f"temp/{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Analyze
        result = analyzer.analyze(temp_path)
        
        # Encode processed image to base64
        _, buffer = cv2.imencode('.jpg', result["processed_image"])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Clean up
        os.remove(temp_path)

        return JSONResponse({
            "success": True,
            "vehicle_count": result["vehicle_count"],
            "occupancy_ratio": result["occupancy_ratio"],
            "congestion_level": result["congestion_level"],
            "image": f"data:image/jpeg;base64,{img_base64}"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...), frame_skip: int = Form(1)):
    allowed_ext = {".mp4", ".avi", ".mov", ".mkv"}
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in allowed_ext:
        return JSONResponse({
            "success": False,
            "error": "Unsupported video format. Use .mp4, .avi, .mov, or .mkv"
        }, status_code=400)

    if frame_skip < 1:
        return JSONResponse({
            "success": False,
            "error": "frame_skip must be >= 1"
        }, status_code=400)

    input_name = f"{uuid.uuid4().hex}{ext}"
    input_path = os.path.join("temp", input_name)
    output_name = f"processed_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join("temp", "processed", output_name)

    cap = None
    writer = None

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open uploaded video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions")

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_index = 0
        total_frames = 0
        analyzed_frames = 0
        cumulative_vehicle_count = 0
        cumulative_occupancy = 0.0
        congestion_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}

        analyzer.reset()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            # Enable adaptive temporal tracking if frame_skip > 1
            result = analyzer.analyze_frame(frame, frame_skip=frame_skip, adaptive=(frame_skip > 1))
            out_frame = result["processed_image"]
            analyzed_frames += 1
            cumulative_vehicle_count += result["vehicle_count"]
            cumulative_occupancy += result["occupancy_ratio"]
            if result["congestion_level"] in congestion_counts:
                congestion_counts[result["congestion_level"]] += 1

            writer.write(out_frame)
            frame_index += 1

        if total_frames == 0:
            raise ValueError("Uploaded video has no readable frames")

        avg_vehicle_count = round(cumulative_vehicle_count / analyzed_frames, 3) if analyzed_frames > 0 else 0
        avg_occupancy_ratio = round(cumulative_occupancy / analyzed_frames, 3) if analyzed_frames > 0 else 0

        return JSONResponse({
            "success": True,
            "message": "Video processed successfully",
            "total_frames": total_frames,
            "analyzed_frames": analyzed_frames,
            "frame_skip": frame_skip,
            "average_vehicle_count": avg_vehicle_count,
            "average_occupancy_ratio": avg_occupancy_ratio,
            "congestion_frame_counts": congestion_counts,
            "download_url": f"/download_video/{output_name}"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if os.path.exists(input_path):
            os.remove(input_path)


@app.get("/download_video/{video_name}")
async def download_video(video_name: str):
    safe_name = os.path.basename(video_name)
    video_path = os.path.join("temp", "processed", safe_name)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Processed video not found")

    return FileResponse(video_path, media_type="video/mp4", filename=safe_name)

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    try:
        allowed_ext = {".mp4", ".avi", ".mov", ".mkv"}
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_ext:
            return JSONResponse({"success": False, "error": "Unsupported video format. Use .mp4, .avi, .mov, or .mkv"}, status_code=400)
        
        video_id = uuid.uuid4().hex
        input_name = f"{video_id}{ext}"
        input_path = os.path.join("temp", input_name)
        
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open uploaded video")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return JSONResponse({
            "success": True,
            "video_id": video_id,
            "filename": input_name,
            "total_frames": total_frames
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

def process_and_encode_frame(frame, analyzer, frame_skip=1, adaptive=False):
    result = analyzer.analyze_frame(frame, frame_skip=frame_skip, adaptive=adaptive)
    _, buffer = cv2.imencode('.jpg', result["processed_image"])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return {
        "frame": f"data:image/jpeg;base64,{img_base64}",
        "vehicle_count": result["vehicle_count"],
        "occupancy_ratio": result["occupancy_ratio"],
        "congestion_level": result["congestion_level"],
        "processed_image": result["processed_image"]
    }

@app.websocket("/ws/process_video/{video_id}")
async def websocket_process_video(websocket: WebSocket, video_id: str, frame_skip: int = 1):
    await websocket.accept()
    
    video_file = None
    for f in os.listdir("temp"):
        if f.startswith(video_id):
            video_file = os.path.join("temp", f)
            break
            
    if not video_file:
        await websocket.send_json({"type": "error", "message": "Video file not found"})
        await websocket.close()
        return
        
    output_name = f"processed_{video_id}.mp4"
    output_path = os.path.join("temp", "processed", output_name)
    
    cap = None
    writer = None
    
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError("Could not open video")
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        analyzed_frames = 0
        cumulative_vehicle_count = 0
        cumulative_occupancy = 0.0
        congestion_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        
        # Target duration per frame (e.g. 0.0333s for 30 FPS)
        frame_duration = 1.0 / fps
        import time
        
        analyzer.reset()
        
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
                
            # Call process_and_encode_frame on every frame to support smooth tracking visualization
            use_adaptive = (frame_skip > 1)
            res = await asyncio.to_thread(process_and_encode_frame, frame, analyzer, frame_skip, use_adaptive)
            out_frame = res["processed_image"]
            analyzed_frames += 1
            cumulative_vehicle_count += res["vehicle_count"]
            cumulative_occupancy += res["occupancy_ratio"]
            congestion = res["congestion_level"]
            congestion_counts[congestion] += 1
            
            await websocket.send_json({
                "type": "frame",
                "frame": res["frame"],
                "vehicle_count": res["vehicle_count"],
                "occupancy_ratio": res["occupancy_ratio"],
                "congestion_level": congestion,
                "frame_index": frame_index,
                "total_frames": total_frames
            })
            
            # Pacing matches native single frame duration now since we process every frame
            elapsed = time.time() - start_time
            sleep_time = max(0.001, frame_duration - elapsed)
            await asyncio.sleep(sleep_time)
                
            writer.write(out_frame)
            frame_index += 1
            
        cap.release()
        writer.release()
        
        avg_vehicle_count = round(cumulative_vehicle_count / analyzed_frames, 3) if analyzed_frames > 0 else 0
        avg_occupancy_ratio = round(cumulative_occupancy / analyzed_frames, 3) if analyzed_frames > 0 else 0
        dominant_congestion = "LOW"
        max_count = -1
        for level in ["LOW", "MEDIUM", "HIGH"]:
            if congestion_counts[level] > max_count:
                max_count = congestion_counts[level]
                dominant_congestion = level
                
        await websocket.send_json({
            "type": "complete",
            "average_vehicle_count": avg_vehicle_count,
            "average_occupancy_ratio": avg_occupancy_ratio,
            "congestion_level": dominant_congestion,
            "total_frames": total_frames,
            "analyzed_frames": analyzed_frames,
            "download_url": f"/download_video/{output_name}"
        })
        
    except WebSocketDisconnect:
        print(f"WebSocket disconnected during processing for {video_id}")
    except Exception as e:
        print(f"Error in WebSocket processing: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        if os.path.exists(video_file):
            os.remove(video_file)

@app.websocket("/ws/detect_frame")
async def websocket_detect_frame(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Real-time frame detection WebSocket client connected")
    try:
        frame_count = 0
        while True:
            # Receive binary frame (JPEG bytes)
            data = await websocket.receive_bytes()
            frame_count += 1
            
            # Decode JPEG
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"[WS] Error decoding frame {frame_count}")
                await websocket.send_json({"error": "Invalid image data"})
                continue
                
            height, width, _ = img.shape
            image_area = height * width
            
            # Run inference
            results = analyzer.model(img, conf=0.4, verbose=False, device=analyzer.device)
            
            detections = []
            vehicle_area = 0
            
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = analyzer.model.names[cls_id]
                    
                    if class_name not in analyzer.vehicle_classes:
                        continue
                        
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    conf = float(box.conf[0])
                    area = (x2 - x1) * (y2 - y1)
                    
                    vehicle_area += area
                    detections.append({
                        "class": class_name,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf
                    })
                    
            occupancy_ratio = vehicle_area / image_area if image_area > 0 else 0
            
            if occupancy_ratio < analyzer.low_congestion:
                congestion_level = "LOW"
            elif occupancy_ratio < analyzer.high_congestion:
                congestion_level = "MEDIUM"
            else:
                congestion_level = "HIGH"
                
            print(f"[WS] Frame {frame_count}: Detected {len(detections)} vehicles. Occupancy: {occupancy_ratio:.3f}. Congestion: {congestion_level}")
            
            # Send results back
            await websocket.send_json({
                "detections": detections,
                "vehicle_count": len(detections),
                "occupancy_ratio": round(occupancy_ratio, 3),
                "congestion_level": congestion_level,
                "width": width,
                "height": height
            })
            
    except WebSocketDisconnect:
        print("[WS] Real-time frame detection WebSocket client disconnected")
    except Exception as e:
        print(f"[WS] Error in real-time frame detection: {e}")

import subprocess
import json

training_process = None

@app.post("/start_training")
async def start_training():
    global training_process
    if training_process and training_process.poll() is None:
        return JSONResponse({"success": False, "message": "Training already in progress"})
    
    try:
        # Run in separate process
        training_process = subprocess.Popen(["python", "training/train_model.py"])
        return JSONResponse({"success": True, "message": "Training started"})
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})

@app.get("/training_status")
async def get_training_status():
    if not os.path.exists("training_status.json"):
        return JSONResponse({"status": "Idle", "epoch": 0, "metrics": {}})
    
    try:
        with open("training_status.json", "r") as f:
            status = json.load(f)
        return JSONResponse(status)
    except:
        return JSONResponse({"status": "Error reading status"})

@app.post("/stop_training")
async def stop_training():
    global training_process
    if training_process and training_process.poll() is None:
        training_process.terminate()
        training_process = None
        
        # update Status file
        with open("training_status.json", "w") as f:
            json.dump({"status": "Stopped", "epoch": 0, "metrics": {}}, f)
            
        return JSONResponse({"success": True, "message": "Training stopped"})
    return JSONResponse({"success": False, "message": "No training in progress"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
