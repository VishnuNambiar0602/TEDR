@echo off
echo Starting TrafficAI Server...
echo Open http://localhost:8000 in your browser once the server starts.
echo Press Ctrl+C to stop the server.
python -m uvicorn app:app --port 8000 --reload
pause
