.PHONY: setup test benchmark run clean docker-build docker-up docker-down

# Install requirements
setup:
	pip install -r requirements.txt
	pip install ultralytics safetensors psutil matplotlib pandas scikit-learn

# Run all unit and integration tests
test:
	python -m unittest discover -s tests -p "test_*.py"

# Run the benchmark suite
benchmark:
	python benchmarks/benchmark_streaming.py

# Run FastAPI dev server
run:
	python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

# Clean temporary files, caches, and shards
clean:
	rm -rf temp/benchmark_shards temp/test_air_detr_shards temp/test_shards
	rm -rf __pycache__ air_detr/__pycache__ tests/__pycache__
	rm -rf .pytest_cache

# Build Docker image
docker-build:
	docker build -t air-detr:latest .

# Launch container stack
docker-up:
	docker-compose up -d

# Stop container stack
docker-down:
	docker-compose down
