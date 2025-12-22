# AWS Deployment Guide - CPU Instance

**Instance:** t3.xlarge or t3.2xlarge (4-8 vCPU, 16-32GB RAM, CPU-only)  
**Domain:** cctv.hio.ai.kr  
**Deployment Path:** /var/www/Hotel-Cash-Detector  
**SSL:** Let's Encrypt (Free)

---

## Step 1: Initial Server Setup

### 1.1 Connect via SSH
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 1.2 Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 1.3 Install System Dependencies
```bash
# Install Python, FFmpeg, and required libraries
sudo apt install -y python3 python3-pip python3-venv ffmpeg git
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# Verify installations
python3 --version
ffmpeg -version
```

---

## Step 2: Clone Repository

```bash
# Create /var/www directory if it doesn't exist
sudo mkdir -p /var/www
sudo chown -R ubuntu:ubuntu /var/www

cd /var/www
git clone https://github.com/Loop-Dimension/Hotel-Cash-Detector.git
cd Hotel-Cash-Detector
```

---

## Step 3: Setup Python Environment

### 3.1 Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3.2 Install Python Dependencies
```bash
pip install --upgrade pip

# Install PyTorch CPU version (no CUDA)
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install -r requirements.txt

# Verify CPU mode
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Using:', 'CPU' if not torch.cuda.is_available() else 'GPU')"
```

**Expected output:**
```
CUDA Available: False
Using: CPU
```

### 3.3 Download YOLO Models
```bash
# Models will auto-download on first run, or manually download:
cd /var/www/Hotel-Cash-Detector/models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt
cd /var/www/Hotel-Cash-Detector
```

---

## Step 4: Configure Django Application

### 4.1 Create Environment File
```bash
nano .env
```

Add the following:
```env
SECRET_KEY=your-super-secret-key-change-this-in-production
DEBUG=False
ALLOWED_HOSTS=cctv.hio.ai.kr,your-ec2-ip,localhost

# Database (SQLite for now)
DATABASE_URL=sqlite:///db.sqlite3

# GPU/CPU Settings
USE_GPU=False

# Detection settings
CASH_DETECTION_CONFIDENCE=0.5
VIOLENCE_DETECTION_CONFIDENCE=0.6
FIRE_DETECTION_CONFIDENCE=0.5
HAND_TOUCH_DISTANCE=100
```

**Generate SECRET_KEY:**
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

### 4.2 Run Database Migrations
```bash
python manage.py migrate
```

### 4.3 Seed Sample Data (Optional)
```bash
# Creates regions, branches, cameras, and test users
python manage.py seed_data

# Login credentials will be displayed:
# Admin: admin / admin123
# Project Manager: pm_seoul / pm123
# Project Manager: pm_gyeonggi / pm123
# Project Manager: pm_busan / pm123
```

### 4.4 Create Superuser (Skip if using seed_data)
```bash
python manage.py createsuperuser
```

### 4.5 Collect Static Files
```bash
python manage.py collectstatic --noinput
```

### 4.6 Test Django Application
```bash
# Test if Django runs
python manage.py runserver 0.0.0.0:8000

# Open another terminal and test
curl http://localhost:8000

# Stop server: Ctrl+C
```

---

## Step 5: Setup Gunicorn Service

### 5.1 Create Gunicorn Configuration
```bash
nano /var/www/Hotel-Cash-Detector/gunicorn_config.py
```

```python
import multiprocessing

bind = "127.0.0.1:8000"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"

# Process naming
proc_name = "hotel-cctv"

# Server mechanics
daemon = False
# Note: pidfile removed - systemd manages the process
user = "ubuntu"
group = "ubuntu"
```

### 5.2 Create Log Directory
```bash
sudo mkdir -p /var/log/gunicorn
sudo chown -R ubuntu:ubuntu /var/log/gunicorn
```

### 5.3 Create Systemd Service
```bash
sudo nano /etc/systemd/system/hotel-cctv.service
```

```ini
[Unit]
Description=Hotel CCTV Detection Service
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/var/www/Hotel-Cash-Detector
Environment="PATH=/var/www/Hotel-Cash-Detector/venv/bin"
EnvironmentFile=/var/www/Hotel-Cash-Detector/.env

ExecStart=/var/www/Hotel-Cash-Detector/venv/bin/gunicorn \
    --config /var/www/Hotel-Cash-Detector/gunicorn_config.py \
    hotel_cctv.wsgi:application

ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 5.4 Enable and Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable hotel-cctv

# IMPORTANT: Before starting, verify these prerequisites
# 1. Virtual environment is activated and dependencies installed
# 2. Database migrations completed: python manage.py migrate
# 3. Static files collected: python manage.py collectstatic --noinput
# 4. .env file exists with required settings

sudo systemctl start hotel-cctv

# Check status
sudo systemctl status hotel-cctv

# If service fails to start, check logs:
sudo journalctl -xeu hotel-cctv.service

# Common fixes:
# - Ensure virtual environment Python path is correct
# - Check .env file exists and has proper format
# - Verify database file permissions
# - Make sure all migrations are applied

# View logs
sudo journalctl -u hotel-cctv -f
```

---

## Step 6: Setup Nginx Reverse Proxy

### 6.1 Install Nginx
```bash
sudo apt install -y nginx
```

### 6.2 Create Nginx Configuration

**Important:** Create a simple HTTP-only config first. Certbot will automatically add HTTPS configuration later.

```bash
sudo nano /etc/nginx/sites-available/hotel-cctv
```

```nginx
# HTTP Server (Certbot will add HTTPS config automatically)
server {
    listen 80;
    server_name cctv.hio.ai.kr;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Video streaming endpoints (no buffering)
    location ~ ^/video-feed {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_buffering off;
        proxy_cache off;
        proxy_http_version 1.1;
        proxy_read_timeout 86400s;
    }

    location /static/ {
        alias /var/www/Hotel-Cash-Detector/staticfiles/;
    }

    location /media/ {
        alias /var/www/Hotel-Cash-Detector/media/;
    }
}
```

### 6.3 Enable Site
```bash
# Test configuration
sudo nginx -t

# Enable site
sudo ln -s /etc/nginx/sites-available/hotel-cctv /etc/nginx/sites-enabled/

# Remove default site
sudo rm /etc/nginx/sites-enabled/default

# Restart Nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
```

---

## Step 7: Setup SSL with Let's Encrypt

### 7.1 Install Certbot
```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 7.2 Verify DNS Configuration
```bash
# Make sure cctv.hio.ai.kr points to your EC2 IP
nslookup cctv.hio.ai.kr

# Or use dig
dig cctv.hio.ai.kr +short
```

**Important:** Before running Certbot, ensure your domain's DNS A record points to your EC2 instance's public IP address.

### 7.3 Obtain SSL Certificate

**Note:** Certbot will automatically modify your Nginx config to add HTTPS and redirect.

```bash
# Get certificate (Certbot will update Nginx config automatically)
sudo certbot --nginx -d cctv.hio.ai.kr

# Follow prompts:
# - Enter email address
# - Agree to Terms of Service (Y)
# - Choose whether to share email (N or Y)
# - Certbot will automatically configure HTTPS redirect

# Verify Nginx is running with HTTPS
sudo systemctl status nginx
curl -I https://cctv.hio.ai.kr
```

**What Certbot does automatically:**
- Creates SSL certificates in `/etc/letsencrypt/`
- Adds `listen 443 ssl` configuration
- Adds SSL certificate paths
- Creates HTTP→HTTPS redirect
- Configures SSL best practices

### 7.4 Test SSL Certificate
```bash
# Check certificate
sudo certbot certificates

# Test renewal (dry run)
sudo certbot renew --dry-run
```

### 7.5 Setup Auto-Renewal
```bash
# Certbot automatically creates a renewal timer
sudo systemctl status certbot.timer

# Enable timer
sudo systemctl enable certbot.timer
```

---

## Step 8: Configure Firewall (AWS Security Group)

### 8.1 Required Inbound Rules

| Type | Protocol | Port | Source | Description |
|------|----------|------|--------|-------------|
| SSH | TCP | 22 | Your IP | SSH access |
| HTTP | TCP | 80 | 0.0.0.0/0 | HTTP (redirects to HTTPS) |
| HTTPS | TCP | 443 | 0.0.0.0/0 | HTTPS access |
| RTSP | TCP | 554 | Camera IPs | Camera streams |

**Steps in AWS Console:**
1. Go to EC2 → Instances → Select your instance
2. Click "Security" tab → Click Security Group
3. Click "Edit inbound rules"
4. Add the rules above
5. Save rules

---

## Step 9: Setup File Permissions

```bash
# Ensure proper ownership
sudo chown -R ubuntu:www-data /var/www/Hotel-Cash-Detector/staticfiles
sudo chown -R ubuntu:www-data /var/www/Hotel-Cash-Detector/static
sudo chown -R ubuntu:www-data /var/www/Hotel-Cash-Detector/media

# Set proper permissions
sudo chmod -R 755 /var/www/Hotel-Cash-Detector/staticfiles
sudo chmod -R 755 /var/www/Hotel-Cash-Detector/static
sudo chmod -R 755 /var/www/Hotel-Cash-Detector/media

# Create media subdirectories if they don't exist
mkdir -p /var/www/Hotel-Cash-Detector/media/{clips,thumbnails,json}
sudo chown -R ubuntu:www-data /var/www/Hotel-Cash-Detector/media
sudo chmod -R 755 /var/www/Hotel-Cash-Detector/media

# Database permissions
chmod 664 /var/www/Hotel-Cash-Detector/db.sqlite3
```

---

## Step 10: Final Verification

### 10.1 Test HTTPS Access
```bash
# From your local machine
curl -I https://cctv.hio.ai.kr

# Should return HTTP/2 200
```

### 10.2 Check All Services
```bash
# Check Django/Gunicorn
sudo systemctl status hotel-cctv

# Check Nginx
sudo systemctl status nginx

# Check Certbot timer
sudo systemctl status certbot.timer

# View Django logs
sudo journalctl -u hotel-cctv -f

# View Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 10.3 Test Web Interface
1. Open browser: `https://cctv.hio.ai.kr`
2. Login with superuser credentials
3. Navigate to Admin panel
4. Add cameras with RTSP URLs
5. Start background workers
6. Verify live streams

---

## Step 11: Maintenance & Monitoring

### 11.1 Update Application
```bash
cd /var/www/Hotel-Cash-Detector
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput
sudo systemctl restart hotel-cctv
```

### 11.2 View Logs
```bash
# Django application logs
sudo journalctl -u hotel-cctv -f

# Nginx access logs
sudo tail -f /var/log/nginx/access.log

# Nginx error logs
sudo tail -f /var/log/nginx/error.log

# Gunicorn logs
sudo tail -f /var/log/gunicorn/access.log
sudo tail -f /var/log/gunicorn/error.log
```

### 11.3 Restart Services
```bash
# Restart Django/Gunicorn
sudo systemctl restart hotel-cctv

# Restart Nginx
sudo systemctl restart nginx

# Restart both
sudo systemctl restart hotel-cctv nginx
```

### 11.4 Monitor CPU & Memory Usage
```bash
# Check CPU and memory usage
top

# Or use htop (install if needed: sudo apt install htop)
htop

# Check system resources
free -h
mpstat 1
```

### 11.5 Monitor Disk Space
```bash
# Check disk usage
df -h

# Check media folder size
du -sh /var/www/Hotel-Cash-Detector/media/

# Clean old clips (older than 30 days)
find /var/www/Hotel-Cash-Detector/media/clips/ -mtime +30 -delete
find /var/www/Hotel-Cash-Detector/media/thumbnails/ -mtime +30 -delete
find /var/www/Hotel-Cash-Detector/media/json/ -mtime +30 -delete
```

### 11.6 Setup Automatic Cleanup (Cron)
```bash
# Edit crontab
crontab -e

# Add this line to clean files older than 30 days (runs daily at 2 AM)
0 2 * * * find /var/www/Hotel-Cash-Detector/media/clips/ -mtime +30 -delete
0 2 * * * find /var/www/Hotel-Cash-Detector/media/thumbnails/ -mtime +30 -delete
0 2 * * * find /var/www/Hotel-Cash-Detector/media/json/ -mtime +30 -delete
```

---

## Step 12: Backup Strategy

### 12.1 Database Backup
```bash
# Create backup directory
mkdir -p ~/backups

# Backup script
nano ~/backup.sh
```

```bash
#!/bin/bash
BACKUP_DIR="/var/www/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_PATH="/var/www/Hotel-Cash-Detector/db.sqlite3"

# Backup database
cp $DB_PATH $BACKUP_DIR/db_backup_$DATE.sqlite3

# Keep only last 7 days
find $BACKUP_DIR -name "db_backup_*.sqlite3" -mtime +7 -delete

echo "Backup completed: db_backup_$DATE.sqlite3"
```

```bash
# Make executable
chmod +x ~/backup.sh

# Add to crontab (daily at 3 AM)
crontab -e
0 3 * * * /home/ubuntu/backup.sh >> /home/ubuntu/backup.log 2>&1
```

### 12.2 S3 Backup (Optional)
```bash
# Install AWS CLI
sudo apt install -y awscli

# Configure AWS credentials
aws configure

# Backup to S3 (add to backup.sh)
aws s3 sync /var/www/Hotel-Cash-Detector/media/ s3://your-bucket/media-backup/
aws s3 cp /var/www/Hotel-Cash-Detector/db.sqlite3 s3://your-bucket/db-backup/db_$(date +%Y%m%d).sqlite3
```

---

## Troubleshooting

### Issue: Service fails to start with "unavailable resources" error
```bash
# Check detailed error logs
sudo journalctl -xeu hotel-cctv.service

# Common causes and fixes:

# 1. Virtual environment not found or incorrect path
ls -la /var/www/Hotel-Cash-Detector/venv/bin/python
# If missing, recreate venv:
cd /var/www/Hotel-Cash-Detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Missing .env file
ls -la /var/www/Hotel-Cash-Detector/.env
# If missing, create it (see Step 4.1)

# 3. Database not initialized
cd /var/www/Hotel-Cash-Detector
source venv/bin/activate
python manage.py migrate

# 4. Missing static files
python manage.py collectstatic --noinput

# 5. Permission issues
sudo chown -R ubuntu:ubuntu /var/www/Hotel-Cash-Detector
chmod 664 /var/www/Hotel-Cash-Detector/db.sqlite3

# After fixing, restart service
sudo systemctl daemon-reload
sudo systemctl restart hotel-cctv
sudo systemctl status hotel-cctv
```

### Issue: Gunicorn won't start
```bash
# Check logs
sudo journalctl -u hotel-cctv -n 50

# Test manually
cd /var/www/Hotel-Cash-Detector
source venv/bin/activate
gunicorn hotel_cctv.wsgi:application --bind 127.0.0.1:8000
```

### Issue: Nginx 502 Bad Gateway
```bash
# Check if Gunicorn is running
sudo systemctl status hotel-cctv

# Check Nginx error logs
sudo tail -f /var/log/nginx/error.log

# Check permissions
ls -la /var/www/Hotel-Cash-Detector/
```

### Issue: SSL certificate fails
```bash
# Verify DNS
nslookup cctv.hio.ai.kr

# Check if port 80 is accessible
curl -I http://cctv.hio.ai.kr

# Check Certbot logs
sudo cat /var/log/letsencrypt/letsencrypt.log
```

### Issue: Static files 403 Forbidden errors
```bash
# Check Nginx error log
sudo tail -f /var/log/nginx/error.log

# If you see "Permission denied" errors, fix permissions:
sudo chown -R ubuntu:www-data /var/www/Hotel-Cash-Detector/staticfiles
sudo chmod -R 755 /var/www/Hotel-Cash-Detector/staticfiles
sudo systemctl restart nginx

# Test static file access
curl -I https://cctv.hio.ai.kr/static/css/style.css
```

### Issue: Video streaming doesn't work
```bash
# Check RTSP connectivity from server
ffmpeg -i "rtsp://camera-ip:554/stream" -frames:v 1 test.jpg

# Check camera RTSP URL in Django admin
# Verify Security Group allows port 554 from camera IPs
```

### Issue: High memory usage
```bash
# Check memory
free -h

# Find memory-heavy processes
ps aux --sort=-%mem | head -n 10

# Restart workers if needed
sudo systemctl restart hotel-cctv
```

---

## Performance Optimization

### CPU Optimization Tips
```bash
# For CPU-only instances, increase worker count
# Edit .env and set:
USE_GPU=False

# Monitor CPU usage
htop
```

### Optimize Worker Count
```bash
# Edit gunicorn config based on camera count
# Rule: 1 worker per 2-3 cameras (CPU uses more workers)
sudo nano /var/www/Hotel-Cash-Detector/gunicorn_config.py

# workers = number_of_cameras / 2 (minimum 2, maximum 8)
```

---

## Security Checklist

- [x] SSH key-based authentication only (disable password auth)
- [x] Firewall configured (AWS Security Group)
- [x] SSL/TLS enabled with Let's Encrypt
- [x] Django DEBUG=False in production
- [x] Strong SECRET_KEY
- [x] Regular system updates
- [x] Database backups enabled
- [x] Log monitoring setup
- [x] File permissions properly set

---

## Quick Reference Commands

```bash
# Restart services
sudo systemctl restart hotel-cctv nginx

# View logs
sudo journalctl -u hotel-cctv -f
sudo tail -f /var/log/nginx/error.log

# Update application
cd /var/www/Hotel-Cash-Detector && git pull && sudo systemctl restart hotel-cctv

# Check system resources
htop

# Check disk space
df -h

# Test site
curl -I https://cctv.hio.ai.kr
```

---

## Support

- GitHub Issues: https://github.com/Loop-Dimension/Hotel-Cash-Detector/issues
- Documentation: [README.md](README.md)

---

**Deployment Date:** December 22, 2025  
**Instance Type:** AWS t3.xlarge/t3.2xlarge (CPU)  
**Deployment Path:** /var/www/Hotel-Cash-Detector  
**Domain:** cctv.hio.ai.kr  
**SSL Provider:** Let's Encrypt  
**GPU:** Disabled (CPU-only)
