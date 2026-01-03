# AWS Deployment Guide - CPU Instance (Root User)

**Instance:** t3.xlarge or t3.2xlarge (4-8 vCPU, 16-32GB RAM, CPU-only)  
**Domain:** cctv.hio.ai.kr  
**Deployment Path:** /var/www/Hotel-Cash-Detector  
**SSL:** Let's Encrypt (Free)  
**User:** root

---

## Step 1: Initial Server Setup

### 1.1 Connect via SSH
```bash
ssh -i your-key.pem root@your-ec2-ip
```

### 1.2 Update System
```bash
apt update && apt upgrade -y
```

### 1.3 Install System Dependencies
```bash
# Install Python, FFmpeg, and required libraries
apt install -y python3 python3-pip python3-venv ffmpeg git
apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# Verify installations
python3 --version
ffmpeg -version
```

---

## Step 2: Clone Repository

```bash
# Create /var/www directory if it doesn't exist
mkdir -p /var/www

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

# IMPORTANT: Use 1 worker + multiple threads for CPU instances
# - workers = 1: Prevents multiprocessing conflicts with background detection
# - threads = 4-8: Uses all CPU cores for web requests
workers = 1
threads = 4
worker_class = "gthread"
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
user = "root"
group = "root"
```

### 5.2 Create Log Directory

**CRITICAL:** This step must be completed before starting the service!

```bash
# Create gunicorn log directory
mkdir -p /var/log/gunicorn

# Verify it was created
ls -la /var/log/ | grep gunicorn
```

### 5.3 Create Systemd Service
```bash
nano /etc/systemd/system/hotel-cctv.service
```

```ini
[Unit]
Description=Hotel CCTV Detection Service
After=network.target

[Service]
Type=simple
User=root
Group=root
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

### 5.4 Create Required Directories

**CRITICAL:** Create Ultralytics config directory before starting service:

```bash
# Create Ultralytics config directory for YOLO models
mkdir -p /root/.config/Ultralytics
chmod 755 /root/.config/Ultralytics

# Verify it was created
ls -la /root/.config/
```

### 5.5 Enable and Start Service
```bash
systemctl daemon-reload
systemctl enable hotel-cctv

# IMPORTANT: Before starting, verify these prerequisites
# 1. Virtual environment is activated and dependencies installed
# 2. Database migrations completed: python manage.py migrate
# 3. Static files collected: python manage.py collectstatic --noinput
# 4. .env file exists with required settings
# 5. Ultralytics config directory created

systemctl start hotel-cctv

# Check status
systemctl status hotel-cctv

# If service fails to start, check logs:
journalctl -xeu hotel-cctv.service

# Common fixes:
# - Ensure virtual environment Python path is correct
# - Check .env file exists and has proper format
# - Verify database file permissions
# - Make sure all migrations are applied

# View logs
journalctl -u hotel-cctv -f
```

---

## Step 6: Setup Nginx Reverse Proxy

### 6.1 Install Nginx
```bash
apt install -y nginx
```

### 6.2 Create Nginx Configuration

**Important:** Create a simple HTTP-only config first. Certbot will automatically add HTTPS configuration later.

```bash
nano /etc/nginx/sites-available/hotel-cctv
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
nginx -t

# Enable site
ln -s /etc/nginx/sites-available/hotel-cctv /etc/nginx/sites-enabled/

# Remove default site
rm /etc/nginx/sites-enabled/default

# Restart Nginx
systemctl restart nginx
systemctl enable nginx
```

---

## Step 7: Setup SSL with Let's Encrypt

### 7.1 Install Certbot
```bash
apt install -y certbot python3-certbot-nginx
```

### 7.2 Verify DNS Configuration
```bash
# Make sure cctv.hio.ai.kr points to your EC2 IP
nslookup cctv.loopdimension.com

# Or use dig
dig cctv.loopdimension.com +short
```

**Important:** Before running Certbot, ensure your domain's DNS A record points to your EC2 instance's public IP address.

### 7.3 Obtain SSL Certificate

**Note:** Certbot will automatically modify your Nginx config to add HTTPS and redirect.

```bash
# Get certificate (Certbot will update Nginx config automatically)
certbot --nginx -d cctv.loopdimension.com

# Follow prompts:
# - Enter email address
# - Agree to Terms of Service (Y)
# - Choose whether to share email (N or Y)
# - Certbot will automatically configure HTTPS redirect

# Verify Nginx is running with HTTPS
systemctl status nginx
curl -I https://cctv.loopdimension.com
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
certbot certificates

# Test renewal (dry run)
certbot renew --dry-run
```

### 7.5 Setup Auto-Renewal
```bash
# Certbot automatically creates a renewal timer
systemctl status certbot.timer

# Enable timer
systemctl enable certbot.timer
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
chown -R root:www-data /var/www/Hotel-Cash-Detector/staticfiles
chown -R root:www-data /var/www/Hotel-Cash-Detector/static
chown -R root:www-data /var/www/Hotel-Cash-Detector/media

# Set proper permissions
chmod -R 755 /var/www/Hotel-Cash-Detector/staticfiles
chmod -R 755 /var/www/Hotel-Cash-Detector/static
chmod -R 755 /var/www/Hotel-Cash-Detector/media

# Create media subdirectories if they don't exist
mkdir -p /var/www/Hotel-Cash-Detector/media/{clips,thumbnails,json}
chown -R root:www-data /var/www/Hotel-Cash-Detector/media
chmod -R 755 /var/www/Hotel-Cash-Detector/media

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
systemctl status hotel-cctv

# Check Nginx
systemctl status nginx

# Check Certbot timer
systemctl status certbot.timer

# View Django logs
journalctl -u hotel-cctv -f

# View Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
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
systemctl restart hotel-cctv
```

### 11.2 View Logs
```bash
# Django application logs
journalctl -u hotel-cctv -f

# Nginx access logs
tail -f /var/log/nginx/access.log

# Nginx error logs
tail -f /var/log/nginx/error.log

# Gunicorn logs
tail -f /var/log/gunicorn/access.log
tail -f /var/log/gunicorn/error.log
```

### 11.3 Restart Services
```bash
# Restart Django/Gunicorn
systemctl restart hotel-cctv

# Restart Nginx
systemctl restart nginx

# Restart both
systemctl restart hotel-cctv nginx
```

### 11.4 Monitor CPU & Memory Usage
```bash
# Check CPU and memory usage
top

# Or use htop (install if needed: apt install htop)
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
mkdir -p /root/backups

# Backup script
nano /root/backup.sh
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
chmod +x /root/backup.sh

# Add to crontab (daily at 3 AM)
crontab -e
0 3 * * * /root/backup.sh >> /root/backup.log 2>&1
```

### 12.2 S3 Backup (Optional)
```bash
# Install AWS CLI
apt install -y awscli

# Configure AWS credentials
aws configure

# Backup to S3 (add to backup.sh)
aws s3 sync /var/www/Hotel-Cash-Detector/media/ s3://your-bucket/media-backup/
aws s3 cp /var/www/Hotel-Cash-Detector/db.sqlite3 s3://your-bucket/db-backup/db_$(date +%Y%m%d).sqlite3
```

---

## Troubleshooting

### Issue: Ultralytics Permission Denied Error
```bash
# Error message:
# [ERROR] Failed to initialize ViolenceDetector: [Errno 13] Permission denied: '/root/.config/Ultralytics'
# ❌ Failed to initialize CashTransactionDetector: [Errno 13] Permission denied: '/root/.config/Ultralytics'

# This means Ultralytics (YOLO) cannot create its config directory

# Solution 1: Create the directory with proper permissions (recommended)
mkdir -p /root/.config/Ultralytics
chmod 755 /root/.config/Ultralytics
systemctl restart hotel-cctv

# Solution 2: Use alternative directory
# Edit service file
nano /etc/systemd/system/hotel-cctv.service

# Add under [Service] section:
# Environment="YOLO_CONFIG_DIR=/var/www/Hotel-Cash-Detector/.config/Ultralytics"

# Create the directory
mkdir -p /var/www/Hotel-Cash-Detector/.config/Ultralytics
chmod 755 /var/www/Hotel-Cash-Detector/.config/Ultralytics

# Reload and restart
systemctl daemon-reload
systemctl restart hotel-cctv

# Verify it's working (errors should be gone)
journalctl -u hotel-cctv -f
```

### Issue: PostgreSQL - remaining connection slots are reserved for roles with the SUPERUSER attribute
```bash
# Error message:
# psycopg2.OperationalError: connection to server at "localhost" (::1), port 5432 failed: 
# FATAL:  remaining connection slots are reserved for roles with the SUPERUSER attribute

# This means PostgreSQL has exhausted all available connections

# Step 1: Check current connections
psql -U postgres -c "SELECT count(*) as total_connections FROM pg_stat_activity;"
psql -U postgres -c "SELECT usename, count(*) FROM pg_stat_activity GROUP BY usename;"

# Step 2: Kill idle connections
psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND pid <> pg_backend_pid();"

# Step 3: Increase max_connections (permanent fix)
nano /etc/postgresql/*/main/postgresql.conf
# Find and change:
# max_connections = 100
# To:
# max_connections = 200  (or higher based on your needs)

# Also consider increasing shared_buffers:
# shared_buffers = 128MB  (increase to 256MB or 512MB)

# Step 4: Restart PostgreSQL
systemctl restart postgresql

# Step 5: Verify new settings
psql -U postgres -c "SHOW max_connections;"
psql -U postgres -c "SHOW shared_buffers;"

# Step 6: Restart your application
systemctl restart hotel-cctv

# Prevention: Add connection pooling to Django settings.py:
# DATABASES = {
#     'default': {
#         ...
#         'CONN_MAX_AGE': 60,  # Keep connections for 60 seconds
#         'OPTIONS': {
#             'connect_timeout': 10,
#         }
#     }
# }
```

### Issue: Gunicorn error - '/var/log/gunicorn/error.log' isn't writable
```bash
# Error message:
# Error: Error: '/var/log/gunicorn/error.log' isn't writable [FileNotFoundError(2, 'No such file or directory')]

# Solution: Create the log directory
mkdir -p /var/log/gunicorn

# Restart the service
systemctl restart hotel-cctv

# Verify it's working
systemctl status hotel-cctv
journalctl -u hotel-cctv -n 20
```

### Issue: Service fails to start with "unavailable resources" error
```bash
# Check detailed error logs
journalctl -xeu hotel-cctv.service

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
chown -R root:root /var/www/Hotel-Cash-Detector
chmod 664 /var/www/Hotel-Cash-Detector/db.sqlite3

# After fixing, restart service
systemctl daemon-reload
systemctl restart hotel-cctv
systemctl status hotel-cctv
```

### Issue: Gunicorn won't start
```bash
# Check logs
journalctl -u hotel-cctv -n 50

# Test manually
cd /var/www/Hotel-Cash-Detector
source venv/bin/activate
gunicorn hotel_cctv.wsgi:application --bind 127.0.0.1:8000
```

### Issue: Nginx 502 Bad Gateway
```bash
# Check if Gunicorn is running
systemctl status hotel-cctv

# Check Nginx error logs
tail -f /var/log/nginx/error.log

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
cat /var/log/letsencrypt/letsencrypt.log
```

### Issue: Static files 403 Forbidden errors
```bash
# Check Nginx error log
tail -f /var/log/nginx/error.log

# If you see "Permission denied" errors, fix permissions:
chown -R root:www-data /var/www/Hotel-Cash-Detector/staticfiles
chmod -R 755 /var/www/Hotel-Cash-Detector/staticfiles
systemctl restart nginx

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
systemctl restart hotel-cctv
```

### Issue: Background Detection Service status not refreshing or shows errors
```bash
# This happens when multiple gunicorn workers conflict with multiprocessing
# Solution: Use 1 worker in gunicorn config

# Edit gunicorn config
nano /var/www/Hotel-Cash-Detector/gunicorn_config.py

# Change this line:
# workers = 4  # WRONG for background detection
# To:
# workers = 1  # CORRECT for background detection

# Restart service
systemctl restart hotel-cctv

# Verify single worker is running
ps aux | grep gunicorn
# You should see only 1 gunicorn worker process (plus 1 master)

# Stop all background workers and restart
# In Django admin: Background Detection Service → Stop All
# Then start cameras individually
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
# IMPORTANT: For background detection, ALWAYS use 1 worker
# Multiple workers cause multiprocessing conflicts

nano /var/www/Hotel-Cash-Detector/gunicorn_config.py

# Set workers = 1 (this is correct for CPU instances with background detection)
# The app uses internal multiprocessing for camera workers
# Each camera runs in its own process, so gunicorn workers must be 1
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
- [ ] Consider creating a non-root service user for improved security

---

## Quick Reference Commands

```bash
# Restart services
systemctl restart hotel-cctv nginx

# View logs
journalctl -u hotel-cctv -f
tail -f /var/log/nginx/error.log

# Update application
cd /var/www/Hotel-Cash-Detector && git pull && systemctl restart hotel-cctv

# Check system resources
htop

# Check disk space
df -h

# Test site
curl -I https://cctv.hio.ai.kr
```

---

## Security Note

**Running as root is not recommended for production environments.** Consider creating a dedicated service user instead:

```bash
# Create service user
useradd -r -s /bin/bash -d /var/www/Hotel-Cash-Detector hotel-cctv

# Change ownership
chown -R hotel-cctv:hotel-cctv /var/www/Hotel-Cash-Detector

# Update systemd service to use hotel-cctv user
# Then update gunicorn_config.py user/group settings
```

---

## Support

- GitHub Issues: https://github.com/Loop-Dimension/Hotel-Cash-Detector/issues
- Documentation: [README.md](README.md)

---

**Deployment Date:** January 3, 2026  
**Instance Type:** AWS t3.xlarge/t3.2xlarge (CPU)  
**Deployment Path:** /var/www/Hotel-Cash-Detector  
**Domain:** cctv.hio.ai.kr  
**SSL Provider:** Let's Encrypt  
**GPU:** Disabled (CPU-only)  
**User:** root


# Stop the service first
systemctl stop hotel-cctv

# Fix ownership of entire project (make root own everything)
chown -R root:root /var/www/Hotel-Cash-Detector

# Create all required directories
mkdir -p /var/www/Hotel-Cash-Detector/media/{clips,thumbnails,json,validation_clips,test_results,uploads}
mkdir -p /var/www/Hotel-Cash-Detector/staticfiles
mkdir -p /root/.config/Ultralytics
mkdir -p /var/log/gunicorn

# Set proper permissions for media directories (read, write, execute)
chmod -R 755 /var/www/Hotel-Cash-Detector/media
chmod -R 755 /var/www/Hotel-Cash-Detector/staticfiles
chmod 755 /root/.config/Ultralytics
chmod 755 /var/log/gunicorn

# Fix database permissions
chmod 664 /var/www/Hotel-Cash-Detector/db.sqlite3

# Verify permissions
ls -la /var/www/Hotel-Cash-Detector/
ls -la /var/www/Hotel-Cash-Detector/media/
ls -la /root/.config/

# Restart service
systemctl daemon-reload
systemctl start hotel-cctv

# Check status
systemctl status hotel-cctv

# Watch logs to verify no more permission errors
journalctl -u hotel-cctv -f