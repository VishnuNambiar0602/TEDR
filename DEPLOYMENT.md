# 🚀 Deployment Guide

This guide covers deploying TEDR in various environments.

## Table of Contents

- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Performance Optimization](#performance-optimization)

## Local Development

### Quick Setup

```bash
# Clone and setup
git clone https://github.com/VishnuNambiar0602/TEDR.git
cd TEDR
./setup.sh  # or setup.bat on Windows

# Run server
source venv/bin/activate  # or venv\Scripts\activate on Windows
python backend/main.py
```

Access at: http://localhost:8000/static/index.html

## Production Deployment

### Prerequisites

- Python 3.8+
- 4GB+ RAM (8GB recommended)
- CUDA-capable GPU (optional but recommended)

### Step 1: Environment Setup

```bash
# Create production user
sudo useradd -m -s /bin/bash tedr

# Setup application
cd /opt
sudo git clone https://github.com/VishnuNambiar0602/TEDR.git
sudo chown -R tedr:tedr TEDR
cd TEDR

# Install dependencies
sudo -u tedr python3 -m venv venv
sudo -u tedr venv/bin/pip install -r requirements.txt
```

### Step 2: Configure for Production

Edit `config.yaml`:
```yaml
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  
model:
  device: "cuda"  # or "cpu"
  confidence_threshold: 0.7
```

### Step 3: System Service

Create `/etc/systemd/system/tedr.service`:

```ini
[Unit]
Description=TEDR Object Detection API
After=network.target

[Service]
Type=simple
User=tedr
WorkingDirectory=/opt/TEDR
Environment="PATH=/opt/TEDR/venv/bin"
ExecStart=/opt/TEDR/venv/bin/python backend/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tedr
sudo systemctl start tedr
sudo systemctl status tedr
```

### Step 4: Nginx Reverse Proxy

Install Nginx:
```bash
sudo apt update
sudo apt install nginx
```

Create `/etc/nginx/sites-available/tedr`:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 10M;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/tedr /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 5: SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Docker Deployment

### Basic Docker

```bash
# Build image
docker build -t tedr:latest .

# Run container
docker run -d \
  --name tedr \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  tedr:latest
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker with GPU

Update `docker-compose.yml`:

```yaml
version: '3.8'

services:
  tedr-api:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - DEVICE=cuda
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
```

Run with:
```bash
docker-compose up -d
```

## Cloud Deployment

### AWS EC2

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance Type: g4dn.xlarge (for GPU) or t3.large (CPU)
   - Storage: 30GB+ EBS
   - Security Group: Allow ports 22, 80, 443

2. **Setup**
   ```bash
   ssh -i your-key.pem ubuntu@ec2-instance-ip
   sudo apt update
   sudo apt install python3-pip python3-venv git
   git clone https://github.com/VishnuNambiar0602/TEDR.git
   cd TEDR
   ./setup.sh
   ```

3. **Run with systemd** (see Production Deployment above)

### Google Cloud Platform

1. **Create Compute Engine Instance**
   - Machine type: n1-standard-4 or GPU instance
   - Boot disk: Ubuntu 22.04 LTS, 30GB
   - Firewall: Allow HTTP/HTTPS

2. **Setup and Deploy**
   Same as AWS EC2 steps

### Azure

1. **Create Virtual Machine**
   - Image: Ubuntu 22.04 LTS
   - Size: Standard_D4s_v3 or GPU instance
   - Networking: Allow ports 80, 443

2. **Setup and Deploy**
   Same as AWS EC2 steps

### Heroku

Create `Procfile`:
```
web: python backend/main.py
```

Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## Performance Optimization

### Model Optimization

1. **Use GPU Acceleration**
   ```yaml
   model:
     device: "cuda"
   ```

2. **Reduce Image Size**
   ```yaml
   model:
     image_size: 640  # Smaller = faster
   ```

3. **Adjust Confidence Threshold**
   ```yaml
   model:
     confidence_threshold: 0.8  # Higher = fewer detections
   ```

### Server Optimization

1. **Use Gunicorn for Production**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker backend.main:app
   ```

2. **Enable Caching**
   - Cache model in memory
   - Use Redis for result caching

3. **Load Balancing**
   - Multiple worker processes
   - Nginx load balancing
   - Kubernetes for scaling

### Monitoring

1. **Application Monitoring**
   ```python
   # Add to backend/main.py
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

2. **System Monitoring**
   ```bash
   # Install monitoring tools
   sudo apt install htop nvidia-smi  # for GPU
   ```

3. **Log Management**
   - Use journalctl for systemd logs
   - Centralized logging with ELK stack
   - CloudWatch/Stackdriver for cloud

## Security

### Best Practices

1. **API Security**
   - Implement rate limiting
   - Add API authentication
   - Use HTTPS only in production

2. **File Upload Security**
   - Validate file types
   - Limit file sizes
   - Scan for malware

3. **Environment Variables**
   ```bash
   # Create .env file
   MODEL_NAME=facebook/detr-resnet-50
   API_KEY=your-secret-key
   ```

4. **Firewall Configuration**
   ```bash
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   ```

## Backup and Recovery

### Database Backups
```bash
# Backup checkpoints
tar -czf checkpoints-backup.tar.gz checkpoints/

# Backup data
tar -czf data-backup.tar.gz data/
```

### Automated Backups
```bash
# Add to crontab
0 2 * * * /opt/TEDR/backup.sh
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process
   sudo lsof -i :8000
   # Kill process
   sudo kill -9 <PID>
   ```

2. **Out of Memory**
   - Reduce batch size
   - Use smaller model
   - Add swap space

3. **Slow Performance**
   - Check GPU usage
   - Optimize image size
   - Enable caching

4. **Connection Refused**
   - Check firewall settings
   - Verify service is running
   - Check nginx configuration

## Scaling

### Horizontal Scaling

1. **Load Balancer**
   ```nginx
   upstream tedr_backend {
       server localhost:8000;
       server localhost:8001;
       server localhost:8002;
   }
   ```

2. **Kubernetes Deployment**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: tedr
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: tedr
     template:
       metadata:
         labels:
           app: tedr
       spec:
         containers:
         - name: tedr
           image: tedr:latest
           ports:
           - containerPort: 8000
   ```

### Vertical Scaling

- Upgrade to larger instance
- Add more GPU memory
- Increase RAM

---

For support, see [README.md](README.md) or open an issue on GitHub.
