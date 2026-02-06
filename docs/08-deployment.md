# Deployment - Hotel Cash Detector

Production deployment guide for AWS EC2, Docker, and GPU instances.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [AWS EC2 Deployment](#aws-ec2-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Systemd Service](#systemd-service)
5. [Nginx Reverse Proxy](#nginx-reverse-proxy)
6. [SSL/HTTPS Configuration](#sslhttps-configuration)
7. [GPU Configuration](#gpu-configuration)
8. [Cost Estimation](#cost-estimation)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)

---

## Deployment Overview

### Recommended Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    AWS EC2 Instance                           │
│                 (GPU-enabled g4dn.xlarge)                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Nginx Reverse Proxy (Port 80/443)             │ │
│  └──────────────────────┬──────────────────────────────────┘ │
│                         │                                    │
│  ┌──────────────────────▼────────────────────────────────┐   │
│  │       Gunicorn/Django Application (Port 8000)         │   │
│  │  - 4 Workers, Background Camera Workers (8+ cameras)  │   │
│  │  - MLDetectorProxy → REST calls to ML Service         │   │
│  └──────────────────────┬────────────────────────────────┘   │
│                         │ HTTP (REST API)                    │
│  ┌──────────────────────▼────────────────────────────────┐   │
│  │       FastAPI ML Service (Port 8001)                  │   │
│  │  - UnifiedDetector, CashDetector, ViolenceDetector    │   │
│  │  - FireDetector, GeminiValidator                      │   │
│  └──────────────────────┬────────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼────────────────────────────────┐   │
│  │          PostgreSQL Database (Port 5432)               │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │     Media Storage (/var/www/.../media/)                │   │
│  │  - clips/ (30s video files)                           │   │
│  │  - thumbnails/ (event thumbnails)                     │   │
│  │  - json/ (event metadata)                             │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │           GPU (NVIDIA Tesla T4)                       │   │
│  │  - YOLOv8 Inference (used by ML Service)              │   │
│  │  - 15GB VRAM                                          │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                        ▲
                        │
          ┌─────────────┼─────────────┐
          │             │             │
    ┌─────▼────┐  ┌─────▼────┐  ┌─────▼────┐
    │ Camera 1 │  │ Camera 2 │  │ Camera N │
    │ (RTSP)   │  │ (RTSP)   │  │ (RTSP)   │
    └──────────┘  └──────────┘  └──────────┘
```

**Deployment Options**:
1. **AWS EC2 with GPU** (Recommended) - g4dn.xlarge or g4dn.2xlarge
2. **Docker Compose** - Django + ML Service + PostgreSQL + Nginx
3. **Dual Systemd Services** - `hotel-cctv` (Django) + `hotel-ml-service` (FastAPI)

---

## AWS EC2 Deployment

### Step 1: Launch EC2 Instance

**Recommended Instance Types**:

| Instance | vCPU | RAM | GPU | Cameras | Monthly Cost |
|----------|------|-----|-----|---------|--------------|
| **g4dn.xlarge** | 4 | 16 GB | Tesla T4 (16GB) | 4-8 | ~$384 |
| **g4dn.2xlarge** | 8 | 32 GB | Tesla T4 (16GB) | 8-16 | ~$549 |
| g4dn.4xlarge | 16 | 64 GB | Tesla T4 (16GB) | 16-24 | ~$879 |

**AMI Selection**:
- **Deep Learning AMI (Ubuntu 20.04)** - Pre-installed CUDA, cuDNN, PyTorch
- Or: **Ubuntu Server 20.04 LTS** - Manual CUDA installation required

**Launch Configuration**:
```bash
# AMI: Deep Learning AMI (Ubuntu 20.04)
# Instance Type: g4dn.xlarge
# Storage: 100 GB GP3 SSD
# Key Pair: Create or select existing
```

---

### Step 2: Security Group Configuration

**Inbound Rules**:

| Type | Protocol | Port | Source | Description |
|------|----------|------|--------|-------------|
| SSH | TCP | 22 | Your IP | SSH access |
| HTTP | TCP | 80 | 0.0.0.0/0 | Web access |
| HTTPS | TCP | 443 | 0.0.0.0/0 | Secure web access |
| Custom TCP | TCP | 8000 | 0.0.0.0/0 | Django dev server (optional) |
| RTSP | TCP | 554 | Camera IPs | RTSP camera streams |

**Outbound Rules**:
- All traffic (default)

---

### Step 3: Connect and Setup

**Connect to Instance**:
```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

**Update System**:
```bash
sudo apt update
sudo apt upgrade -y
```

**Install System Dependencies**:
```bash
# FFmpeg (required for video encoding)
sudo apt install -y ffmpeg

# PostgreSQL (if not using external database)
sudo apt install -y postgresql postgresql-contrib

# Git (if not pre-installed)
sudo apt install -y git
```

---

### Step 4: Clone Repository

```bash
cd ~
git clone https://github.com/Loop-Dimension/Hotel-Cash-Detector.git
cd Hotel-Cash-Detector/django_app
```

---

### Step 5: Setup Python Environment

**Create Virtual Environment**:
```bash
python3 -m venv venv
source venv/bin/activate
```

**Install Dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn psycopg2-binary
```

**Verify GPU**:
```bash
# Check NVIDIA GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected Output**:
```
CUDA available: True
GPU count: 1
GPU name: Tesla T4
```

---

### Step 6: Download AI Models

**Auto-Download Models**:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); YOLO('yolov8s-pose.pt')"
```

**Models will be cached** in `~/.cache/ultralytics/`

**Fire Model**: Obtain `fire_smoke_yolov8.pt` from development team and place in `django_app/models/`.

---

### Step 7: Configure Environment

**Create `.env` file**:
```bash
cd ~/Hotel-Cash-Detector/django_app
nano .env
```

**Production Configuration**:
```env
# Django Settings
SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
DEBUG=False
ALLOWED_HOSTS=your-domain.com,your-ec2-ip

# Database (PostgreSQL)
DB_ENGINE=postgresql
DB_NAME=cctv
DB_USER=orange
DB_PASSWORD=00oo00oo
DB_HOST=localhost
DB_PORT=5432

# GPU Settings
USE_GPU=cuda

# Detection Confidence Thresholds
CASH_DETECTION_CONFIDENCE=0.5
VIOLENCE_DETECTION_CONFIDENCE=0.6
FIRE_DETECTION_CONFIDENCE=0.5

# Model Paths
YOLO_MODEL=models/yolov8s.pt
POSE_MODEL=models/yolov8s-pose.pt
FIRE_MODEL=models/fire_smoke_yolov8.pt

# Optional: Gemini AI Validation
GEMINI_API_KEY=your-gemini-api-key
GEMINI_VALIDATION_ENABLED=True
GEMINI_USE_VIDEO_VALIDATION=False
```

**Generate Secret Key**:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

### Step 8: Setup PostgreSQL Database

See [POSTGRESQL_SETUP.md](../POSTGRESQL_SETUP.md) for complete instructions.

**Quick Setup**:
```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE cctv;
CREATE USER orange WITH PASSWORD '00oo00oo';
GRANT ALL PRIVILEGES ON DATABASE cctv TO orange;

# For PostgreSQL 15+ (Ubuntu 24.04)
\c cctv
GRANT ALL ON SCHEMA public TO orange;
GRANT CREATE ON SCHEMA public TO orange;
ALTER DATABASE cctv OWNER TO orange;
ALTER SCHEMA public OWNER TO orange;
\q
```

---

### Step 9: Run Database Migrations

```bash
cd ~/Hotel-Cash-Detector/django_app
source venv/bin/activate

python manage.py migrate
python manage.py createsuperuser
```

---

### Step 10: Collect Static Files

```bash
python manage.py collectstatic --noinput
```

**Static files** will be copied to `static/` directory for Nginx to serve.

---

## Systemd Services

Production deployments use **two systemd services**: one for Django and one for the ML service.

### Service 1: Django Backend (`hotel-cctv`)

```bash
sudo nano /etc/systemd/system/hotel-cctv.service
```

```ini
[Unit]
Description=Hotel Cash Detector - Django Backend
After=network.target postgresql.service hotel-ml-service.service

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/Hotel-Cash-Detector
Environment="PATH=/var/www/Hotel-Cash-Detector/venv/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="PYTHONWARNINGS=ignore"
ExecStart=/var/www/Hotel-Cash-Detector/venv/bin/gunicorn \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile /var/log/hotel-cctv/access.log \
    --error-logfile /var/log/hotel-cctv/error.log \
    hotel_cctv.wsgi:application
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Important**: `PYTHONUNBUFFERED=1` is required for background thread `print()` statements (like MODE indicators) to appear in real-time in `journalctl`.

### Service 2: ML Service (`hotel-ml-service`)

```bash
sudo nano /etc/systemd/system/hotel-ml-service.service
```

```ini
[Unit]
Description=Hotel Cash Detector - ML Detection Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/var/www/Hotel-Cash-Detector/ml_service
Environment="PATH=/var/www/Hotel-Cash-Detector/ml_service/venv/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/var/www/Hotel-Cash-Detector/ml_service/venv/bin/uvicorn \
    app.main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 1
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Enable and Start Both Services

```bash
# Create log directory
sudo mkdir -p /var/log/hotel-cctv
sudo chown root:root /var/log/hotel-cctv

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable hotel-ml-service hotel-cctv
sudo systemctl start hotel-ml-service
sudo systemctl start hotel-cctv
```

### Managing Services

```bash
# Check status
sudo systemctl status hotel-cctv
sudo systemctl status hotel-ml-service

# View Django logs (includes MODE indicators)
sudo journalctl -u hotel-cctv -f

# View ML service logs
sudo journalctl -u hotel-ml-service -f

# Restart both
sudo systemctl restart hotel-ml-service && sudo systemctl restart hotel-cctv

# Test ML service health
curl http://localhost:8001/health
```

### ML Service Setup

The ML service has its own Python virtual environment:

```bash
cd /var/www/Hotel-Cash-Detector/ml_service

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Nginx Reverse Proxy

### Install Nginx

```bash
sudo apt install -y nginx
```

### Create Nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/hotel-cctv
```

**Configuration**:
```nginx
upstream django_backend {
    server 127.0.0.1:8000;
    keepalive 64;
}

# HTTP → HTTPS redirect
server {
    listen 80;
    listen [::]:80;
    server_name cctv.hio.ai.kr;

    # Redirect all HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name cctv.hio.ai.kr;

    # SSL certificates (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/cctv.hio.ai.kr/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/cctv.hio.ai.kr/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Logs
    access_log /var/log/nginx/cctv_access.log;
    error_log /var/log/nginx/cctv_error.log;

    # Client body size (for file uploads)
    client_max_body_size 100M;

    # Proxy to Django/Gunicorn
    location / {
        proxy_pass http://django_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
    }

    # Static files
    location /static/ {
        alias /home/ubuntu/Hotel-Cash-Detector/django_app/static/;
        expires 30d;
        add_header Cache-Control "public, max-age=2592000, immutable";
    }

    # Media files (video clips, thumbnails)
    location /media/ {
        alias /home/ubuntu/Hotel-Cash-Detector/django_app/media/;
        expires 7d;
        add_header Cache-Control "public, max-age=604800";
    }

    # Video streaming - disable buffering
    location /api/stream/ {
        proxy_pass http://django_backend;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400;
    }
}
```

**Enable Site**:
```bash
sudo ln -s /etc/nginx/sites-available/hotel-cctv /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## SSL/HTTPS Configuration

### Option 1: Let's Encrypt (Free)

**Install Certbot**:
```bash
sudo apt install -y certbot python3-certbot-nginx
```

**Obtain Certificate**:
```bash
sudo certbot --nginx -d cctv.hio.ai.kr
```

**Auto-Renewal**:
```bash
# Test renewal
sudo certbot renew --dry-run

# Cron job is auto-created
sudo systemctl status certbot.timer
```

**Certificate Locations**:
- Certificate: `/etc/letsencrypt/live/cctv.hio.ai.kr/fullchain.pem`
- Private Key: `/etc/letsencrypt/live/cctv.hio.ai.kr/privkey.pem`

---

### Option 2: Self-Signed Certificate (Development/Testing)

```bash
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/cctv-selfsigned.key \
  -out /etc/ssl/certs/cctv-selfsigned.crt
```

**Update Nginx config**:
```nginx
ssl_certificate /etc/ssl/certs/cctv-selfsigned.crt;
ssl_certificate_key /etc/ssl/private/cctv-selfsigned.key;
```

**Note**: Self-signed certificates will show browser warnings.

---

## Docker Deployment

### ML Service Dockerfile

Create `ml_service/Dockerfile`:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
```

**Note**: Use `libgl1` not `libgl1-mesa-glx` (newer Debian/Ubuntu).

### Django Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .
RUN mkdir -p media/clips media/thumbnails media/json

EXPOSE 8000

CMD python manage.py migrate && \
    python manage.py collectstatic --noinput && \
    gunicorn --workers 4 --bind 0.0.0.0:8000 --timeout 120 hotel_cctv.wsgi:application
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
services:
  ml-service:
    build: ./ml_service
    container_name: hotel-ml-service
    ports:
      - "8001:8001"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - USE_GPU=auto
    volumes:
      - ./models:/app/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    restart: unless-stopped

  django-backend:
    build: .
    container_name: hotel-cctv-web
    ports:
      - "8000:8000"
    environment:
      - DEBUG=False
      - ALLOWED_HOSTS=cctv.hio.ai.kr,localhost
      - USE_ML_SERVICE=True
      - ML_SERVICE_URL=http://ml-service:8001
      - ML_SERVICE_TIMEOUT=30
      - DB_ENGINE=postgresql
      - DB_NAME=cctv
      - DB_USER=orange
      - DB_PASSWORD=00oo00oo
      - DB_HOST=postgres
      - DB_PORT=5432
      - PYTHONUNBUFFERED=1
    volumes:
      - ./media:/app/media
      - ./models:/app/models
    depends_on:
      - ml-service
      - postgres
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    container_name: hotel-cctv-db
    environment:
      - POSTGRES_DB=cctv
      - POSTGRES_USER=orange
      - POSTGRES_PASSWORD=00oo00oo
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: hotel-cctv-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf
      - ./static:/static
      - ./media:/media
      - ./certbot/conf:/etc/letsencrypt
      - ./certbot/www:/var/www/certbot
    depends_on:
      - django-backend
    restart: unless-stopped

volumes:
  postgres_data:
```

**Note**: Do not use the `version:` attribute in `docker-compose.yml` (it's obsolete in modern Docker Compose).

**Run with Docker Compose**:
```bash
docker compose up -d
```

**View Logs**:
```bash
# All services
docker compose logs -f

# ML service only
docker compose logs -f ml-service

# Django only
docker compose logs -f django-backend
```

**Verify ML Service**:
```bash
curl http://localhost:8001/health
curl http://localhost:8001/status
```

---

## GPU Configuration

### NVIDIA Driver Installation (If Not Pre-Installed)

**Check GPU**:
```bash
lspci | grep -i nvidia
```

**Install NVIDIA Drivers** (Ubuntu):
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

**Verify Installation**:
```bash
nvidia-smi
```

---

### CUDA Installation

**For Deep Learning AMI**: CUDA is pre-installed.

**For Ubuntu Server** (manual installation):

```bash
# Download CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

---

### PyTorch with CUDA

**Install PyTorch**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Verify**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

---

## Cost Estimation

### AWS EC2 Pricing (On-Demand, US East)

**GPU Instances** (730 hours/month):

| Instance | vCPU | RAM | GPU | Hourly | Monthly (USD) |
|----------|------|-----|-----|--------|---------------|
| **g4dn.xlarge** | 4 | 16 GB | Tesla T4 | $0.526 | **$384** |
| **g4dn.2xlarge** | 8 | 32 GB | Tesla T4 | $0.752 | **$549** |
| g4dn.4xlarge | 16 | 64 GB | Tesla T4 | $1.204 | $879 |
| g5.xlarge | 4 | 16 GB | A10G | $1.006 | $734 |

**CPU Instances** (no GPU, slower):

| Instance | vCPU | RAM | Hourly | Monthly (USD) |
|----------|------|-----|--------|---------------|
| t3.xlarge | 4 | 16 GB | $0.1664 | $122 |
| m5.xlarge | 4 | 16 GB | $0.192 | $140 |

**Savings Plans** (1-Year, No Upfront):
- g4dn.xlarge: ~$276/month (28% savings)
- g4dn.2xlarge: ~$395/month (28% savings)

**Additional Costs**:
- **EBS Storage**: 100 GB GP3 SSD = ~$8/month
- **Data Transfer**: First 100 GB free, then $0.09/GB
- **Total (g4dn.xlarge)**: ~$392/month

---

## Monitoring and Maintenance

### Application Monitoring

**Check Service Status**:
```bash
sudo systemctl status hotel-cctv
```

**View Logs**:
```bash
# Systemd logs
sudo journalctl -u hotel-cctv -f

# Application logs
tail -f /var/log/hotel-cctv/access.log
tail -f /var/log/hotel-cctv/error.log

# Nginx logs
tail -f /var/log/nginx/cctv_access.log
tail -f /var/log/nginx/cctv_error.log
```

---

### GPU Monitoring

**Real-Time GPU Usage**:
```bash
watch -n 1 nvidia-smi
```

**GPU Memory**:
```python
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

---

### Database Backup

**Create Backup Script** (`/home/ubuntu/backup-cctv.sh`):

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/ubuntu/backups"
mkdir -p $BACKUP_DIR

# Dump database
sudo -u postgres pg_dump cctv > $BACKUP_DIR/cctv_backup_$DATE.sql

# Compress
gzip $BACKUP_DIR/cctv_backup_$DATE.sql

# Delete backups older than 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Backup completed: cctv_backup_$DATE.sql.gz"
```

**Make Executable**:
```bash
chmod +x /home/ubuntu/backup-cctv.sh
```

**Add to Crontab** (daily at 2 AM):
```bash
crontab -e

# Add line:
0 2 * * * /home/ubuntu/backup-cctv.sh
```

---

### Updates and Rollback

**Automated Update Script** (`update_server.sh`):

The project includes an `update_server.sh` script that handles updating both Django and ML service:

```bash
./update_server.sh
```

This script:
1. Pulls latest code from `main` branch
2. Installs Django dependencies
3. Runs database migrations
4. Installs ML service dependencies (in `ml_service/venv/`)
5. Restarts both systemd services

**Manual Update**:
```bash
cd /var/www/Hotel-Cash-Detector
git pull origin main

# Update Django
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput

# Update ML Service
cd ml_service
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Restart both services
sudo systemctl restart hotel-ml-service
sudo systemctl restart hotel-cctv
```

**Rollback**:
```bash
git checkout <previous-commit-hash>

# Reinstall deps for both services
source venv/bin/activate && pip install -r requirements.txt
cd ml_service && source venv/bin/activate && pip install -r requirements.txt && cd ..

python manage.py migrate
sudo systemctl restart hotel-ml-service
sudo systemctl restart hotel-cctv
```

---

## Security Checklist

Before going live:

- ✅ Change `SECRET_KEY` to strong random value
- ✅ Set `DEBUG=False` in production
- ✅ Configure `ALLOWED_HOSTS` with actual domain
- ✅ Enable HTTPS with Let's Encrypt
- ✅ Configure firewall (allow 80, 443, 22 only)
- ✅ Restrict PostgreSQL to localhost
- ✅ Use strong database passwords
- ✅ Regular security updates: `sudo apt update && sudo apt upgrade`
- ✅ Backup database regularly
- ✅ Monitor logs for suspicious activity

---

## Related Documentation

- [02 - Setup](02-setup.md) - Development setup
- [03 - Environment Variables](03-env.md) - Configuration reference
- [06 - Integrations](06-integrations.md) - GPU/CUDA integration
- [09 - Troubleshooting](09-troubleshooting.md) - Common deployment issues

---

**Previous**: [← 07 - Flows](07-flows.md) | **Next**: [09 - Troubleshooting →](09-troubleshooting.md)
