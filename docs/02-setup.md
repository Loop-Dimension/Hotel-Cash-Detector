# Setup - Hotel Cash Detector

Complete guide for local development and production setup.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [ML Service Setup](#ml-service-setup)
4. [AI Model Downloads](#ai-model-downloads)
5. [Database Setup](#database-setup)
6. [Environment Configuration](#environment-configuration)
7. [RTSP Camera Testing](#rtsp-camera-testing)
8. [Running the Application](#running-the-application)
9. [Production Setup](#production-setup)

---

## Prerequisites

### System Requirements

**Minimum (Development)**:
- Python 3.10+
- CPU: Intel i5 or equivalent
- RAM: 8 GB
- Storage: 50 GB
- OS: Windows 10+, Ubuntu 20.04+, or macOS 11+

**Recommended (Production)**:
- Python 3.10+
- CPU: Intel i7+ or AWS g4dn.xlarge
- RAM: 16 GB
- GPU: NVIDIA GTX 1650+ with CUDA support
- Storage: 100+ GB SSD
- OS: Ubuntu 20.04 LTS or Ubuntu 24.04 LTS

### CUDA/GPU Requirements (Optional but Recommended)

**For GPU acceleration**:
- NVIDIA GPU with CUDA support (Compute Capability 3.5+)
- CUDA Toolkit 11.0+ or 12.0+
- cuDNN compatible with CUDA version
- PyTorch with CUDA support

**Check GPU availability**:
```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.x.x       Driver Version: 535.x.x       CUDA Version: 12.2  |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
# | N/A   45C    P0    26W /  70W |      0MiB / 15360MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+
```

If `nvidia-smi` fails, install NVIDIA drivers and CUDA Toolkit from [nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

---

## Local Development Setup

### Step 1: Clone Repository

```bash
# Clone from GitHub
git clone https://github.com/yourorg/Hotel-Cash-Detector.git
cd Hotel-Cash-Detector
```

### Step 2: Create Virtual Environment

**Windows (PowerShell)**:
```powershell
# Navigate to django_app directory
cd django_app

# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1
```

**Linux/macOS**:
```bash
cd django_app

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

**Key Dependencies**:
```
Django==5.2.7
torch==2.0.0+cu118          # PyTorch with CUDA 11.8
torchvision==0.15.0+cu118
ultralytics==8.0.196        # YOLOv8 framework
opencv-python==4.8.1.78     # Computer vision
pillow==10.1.0              # Image processing
numpy==1.24.3               # Array operations
psycopg2-binary==2.9.9      # PostgreSQL adapter (optional)
python-dotenv==1.0.0        # Environment variables
```

**GPU-specific Installation**:

If you have NVIDIA GPU with CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If you have CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only (no GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Configure ML Service Environment Variables

Add to `.env`:
```env
# ML Service
USE_ML_SERVICE=True
ML_SERVICE_URL=http://localhost:8001
ML_SERVICE_TIMEOUT=30
```

Set `USE_ML_SERVICE=False` to use local detectors without the ML service.

### Step 5: Verify PyTorch CUDA

```bash
# Run Python and check CUDA
python
```

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

**Expected Output (with GPU)**:
```
PyTorch version: 2.0.0+cu118
CUDA available: True
CUDA version: 11.8
GPU count: 1
GPU name: Tesla T4
GPU memory: 15.00 GB
```

**Expected Output (CPU-only)**:
```
PyTorch version: 2.0.0+cpu
CUDA available: False
CUDA version: None
```

---

## ML Service Setup

The ML service is a standalone FastAPI application that handles AI detection. It has its own virtual environment and dependencies.

### Step 1: Create ML Service Virtual Environment

```bash
cd ml_service

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\Activate.ps1  # Windows
```

### Step 2: Install ML Service Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Start ML Service

```bash
cd ml_service
source venv/bin/activate

# Start with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Expected Output**:
```
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Started reloader process
```

### Step 4: Verify ML Service

```bash
# Health check
curl http://localhost:8001/health

# Status (shows GPU info)
curl http://localhost:8001/status
```

**Note**: The ML service must be running before starting Django (if `USE_ML_SERVICE=True`). If Django starts and ML service is unavailable, it will automatically fall back to local detection.

---

## AI Model Downloads

### Auto-Download (Recommended)

AI models are **automatically downloaded** on first run when referenced.

**Models Required**:

| Model | File | Size | Purpose |
|-------|------|------|---------|
| YOLOv8 Small | `yolov8s.pt` | ~22 MB | Person detection |
| YOLOv8 Pose Small | `yolov8s-pose.pt` | ~23 MB | Hand position tracking |
| Fire/Smoke | `fire_smoke_yolov8.pt` | ~6 MB | Fire and smoke detection |

**Auto-Download Test**:
```bash
cd django_app

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\Activate.ps1  # Windows

# Test model download
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); YOLO('yolov8s-pose.pt'); print('Models downloaded successfully')"
```

**Expected Output**:
```
Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt to yolov8s.pt...
100%|████████████████████████| 21.5M/21.5M [00:02<00:00, 9.20MB/s]

Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt to yolov8s-pose.pt...
100%|████████████████████████| 23.0M/23.0M [00:02<00:00, 9.50MB/s]

Models downloaded successfully
```

Models will be stored in:
- Windows: `C:\Users\<username>\.cache\ultralytics\`
- Linux: `~/.cache/ultralytics/`

### Manual Download

If auto-download fails due to network issues:

```bash
# Create models directory
cd django_app
mkdir -p models

# Download models manually
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -O models/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt -O models/yolov8s-pose.pt
```

**Fire/Smoke Model** (Custom-Trained):

Contact the development team for the `fire_smoke_yolov8.pt` model file. Place it in `django_app/models/`.

**Update `.env`**:
```env
YOLO_MODEL=models/yolov8s.pt
POSE_MODEL=models/yolov8s-pose.pt
FIRE_MODEL=models/fire_smoke_yolov8.pt
```

---

## Database Setup

### Development: SQLite (Default)

SQLite is used by default for development. No setup required.

**Database File**: `django_app/db.sqlite3` (auto-created on first run)

**Advantages**:
- Zero configuration
- Portable (single file)
- Perfect for development

**Limitations**:
- Single connection (not suitable for production)
- Limited concurrency
- No advanced features

### Production: PostgreSQL (Recommended)

#### Install PostgreSQL

**Ubuntu/Debian**:
```bash
# Update package list
sudo apt update

# Install PostgreSQL 16
sudo apt install -y postgresql postgresql-contrib

# Start and enable service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Verify installation
sudo systemctl status postgresql
```

**Windows**:
Download PostgreSQL installer from [postgresql.org/download/windows](https://www.postgresql.org/download/windows/).

#### Create Database and User

```bash
# Switch to postgres user
sudo -u postgres psql

# Run these commands in psql
CREATE DATABASE cctv;
CREATE USER orange WITH PASSWORD '00oo00oo';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE cctv TO orange;

# For PostgreSQL 15+ (Ubuntu 24.04), grant schema privileges
\c cctv
GRANT ALL ON SCHEMA public TO orange;
GRANT CREATE ON SCHEMA public TO orange;
GRANT USAGE ON SCHEMA public TO orange;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO orange;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO orange;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO orange;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO orange;
ALTER DATABASE cctv OWNER TO orange;
ALTER SCHEMA public OWNER TO orange;

# Exit psql
\q
```

#### Configure PostgreSQL Authentication

```bash
# Edit PostgreSQL config
sudo nano /etc/postgresql/16/main/pg_hba.conf

# Find this line:
# local   all             all                                     peer

# Change to:
# local   all             all                                     md5

# Also ensure this line exists for TCP/IP:
# host    all             all             127.0.0.1/32            md5

# Restart PostgreSQL
sudo systemctl restart postgresql
```

#### Test Database Connection

```bash
# Test connection
psql -h localhost -U orange -d cctv -W
# Enter password: 00oo00oo

# If successful, you'll see:
# cctv=>

# Exit with \q
```

#### Update Django Settings

Edit `django_app/.env`:
```env
# Database (PostgreSQL)
DB_ENGINE=postgresql
DB_NAME=cctv
DB_USER=orange
DB_PASSWORD=00oo00oo
DB_HOST=localhost
DB_PORT=5432
```

#### Migrate from SQLite to PostgreSQL

If you have existing SQLite data:

```bash
cd django_app
source venv/bin/activate

# Dump SQLite data
python manage.py dumpdata --natural-foreign --natural-primary \
  --exclude contenttypes --exclude auth.Permission \
  --exclude sessions.session > data_backup.json

# Update .env to use PostgreSQL
# (edit DB_ENGINE=postgresql)

# Run migrations on PostgreSQL
python manage.py migrate

# Load data
python manage.py loaddata data_backup.json
```

---

## Environment Configuration

### Step 1: Create `.env` File

```bash
cd django_app

# Copy example file
cp .env.example .env

# Edit with your configuration
nano .env
```

### Step 2: Configure Required Variables

**Minimal Configuration**:
```env
# Django Settings
SECRET_KEY=your-super-secret-key-change-in-production
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# GPU Settings
USE_GPU=auto

# Detection Confidence Thresholds
CASH_DETECTION_CONFIDENCE=0.5
VIOLENCE_DETECTION_CONFIDENCE=0.6
FIRE_DETECTION_CONFIDENCE=0.5

# Model Paths (relative to django_app/)
YOLO_MODEL=models/yolov8s.pt
POSE_MODEL=models/yolov8s-pose.pt
FIRE_MODEL=models/fire_smoke_yolov8.pt
```

**Optional Gemini AI Validation**:
```env
# Gemini AI (Optional - reduces false positives)
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_VALIDATION_ENABLED=True
GEMINI_USE_VIDEO_VALIDATION=False
```

Get Gemini API key: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

**Complete Configuration**: See [03-env.md](03-env.md) for all environment variables.

### Step 3: Run Database Migrations

```bash
cd django_app
source venv/bin/activate

# Create database tables
python manage.py migrate

# Expected output:
# Operations to perform:
#   Apply all migrations: admin, auth, contenttypes, cctv, sessions
# Running migrations:
#   Applying contenttypes.0001_initial... OK
#   Applying auth.0001_initial... OK
#   Applying cctv.0001_initial... OK
#   ...
```

### Step 4: Create Admin User

```bash
# Create superuser
python manage.py createsuperuser

# Enter:
# Username: admin
# Email: admin@example.com
# Password: (your-password)
# Password (again): (your-password)
```

### Step 5: Seed Initial Data (Optional)

```bash
# Create default regions and branches
python manage.py shell
```

```python
from cctv.models import Region, Branch

# Create default region
region = Region.objects.create(name='Default', code='DEF')

# Create test branch
branch = Branch.objects.create(
    name='Test Hotel',
    region=region,
    address='123 Test Street',
    status='confirmed'
)

print(f"Created region: {region.name}")
print(f"Created branch: {branch.name}")
exit()
```

---

## RTSP Camera Testing

### Test RTSP URL Format

RTSP URL format:
```
rtsp://username:password@ip:port/stream/path
```

**Examples**:
```bash
# Hikvision
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101

# Dahua
rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0

# Uniview
rtsp://admin:password@192.168.1.120:554/unicast/c1/s0/live

# Generic
rtsp://admin:adminadmin!@175.213.55.16:554
```

### Test with VLC

**Using VLC Media Player**:
1. Open VLC
2. Media → Open Network Stream
3. Enter RTSP URL
4. Click Play
5. If video appears, RTSP URL is correct

### Test with FFplay

**If FFmpeg installed**:
```bash
ffplay rtsp://admin:password@192.168.1.64:554/stream
```

### Test with Python

Create `test_rtsp.py`:
```python
import cv2

rtsp_url = "rtsp://admin:password@192.168.1.64:554/stream"
cap = cv2.VideoCapture(rtsp_url)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"✅ Camera connected: {frame.shape}")
        cv2.imwrite('test_frame.jpg', frame)
        print("Saved test_frame.jpg")
    else:
        print("❌ Could not read frame")
else:
    print("❌ Could not open stream")

cap.release()
```

```bash
python test_rtsp.py
```

---

## Running the Application

### Step 1: Start ML Service (Terminal 1)

```bash
cd ml_service
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Step 2: Start Django Development Server (Terminal 2)

```bash
source venv/bin/activate

# Run Django server
python manage.py runserver 0.0.0.0:8000
```

**Expected Output**:
```
Watching for file changes with StatReloader
Performing system checks...

System check identified no issues (0 silenced).
January 25, 2026 - 10:30:00
Django version 5.2.7, using settings 'hotel_cctv.settings'
Starting development server at http://0.0.0.0:8000/
Quit the server with CTRL-BREAK.
```

**Access Application**:
- Dashboard: [http://localhost:8000](http://localhost:8000)
- Admin Panel: [http://localhost:8000/admin](http://localhost:8000/admin)

### Step 2: Add Camera in Dashboard

1. Log in with admin credentials
2. Navigate to **Cameras** section
3. Click **Add Camera**
4. Fill in:
   - **Name**: "Front Desk Camera"
   - **Branch**: Select branch
   - **Camera ID**: "CAM001"
   - **RTSP URL**: Your camera's RTSP URL
   - **Status**: "online"
5. Click **Save**

### Step 3: Configure Cashier Zone

1. Go to **Camera Settings** for your camera
2. Click **Draw Cashier Zone**
3. Use mouse to draw rectangle around cashier counter area
4. Save zone coordinates

### Step 4: Start Detection Worker

Detection workers run automatically when you access the **Monitor** page:

1. Navigate to **Monitor All Cameras**
2. Click **Start** on your camera
3. Worker will start processing RTSP stream

**Manual Worker Start** (for debugging):
```bash
cd django_app
source venv/bin/activate

python manage.py shell
```

```python
from cctv.models import Camera
from detectors.unified_detector import BackgroundCameraWorker

# Get camera
camera = Camera.objects.get(camera_id='CAM001')

# Start worker
worker = BackgroundCameraWorker(camera)
worker.start()

# Worker runs in background
# Press Ctrl+C to stop
```

### Step 5: View Events

1. Navigate to **Event Logs**
2. View detected cash transactions, violence, or fire events
3. Click on event to view video clip and details

---

## Production Setup

### Option 1: Dual Systemd Services (Ubuntu)

Production requires **two systemd services**: `hotel-cctv` (Django) and `hotel-ml-service` (FastAPI).

See [08-deployment.md](08-deployment.md) for complete systemd service configurations.

**Quick Start**:
```bash
# Start both services
sudo systemctl start hotel-ml-service
sudo systemctl start hotel-cctv

# Check status
sudo systemctl status hotel-cctv
sudo systemctl status hotel-ml-service

# View logs
sudo journalctl -u hotel-cctv -f        # Django + detection MODE indicators
sudo journalctl -u hotel-ml-service -f   # ML service
```

### Option 2: Docker Compose

See [08-deployment.md](08-deployment.md) for Docker Compose configuration with all services (Django, ML Service, PostgreSQL, Nginx).

### Option 3: AWS EC2 with GPU

See [08-deployment.md](08-deployment.md) for AWS g4dn instance setup.

---

## Verification Checklist

After setup, verify:

- ✅ ML service runs: `curl http://localhost:8001/health`
- ✅ Django server runs: `python manage.py runserver`
- ✅ Admin panel accessible: [http://localhost:8000/admin](http://localhost:8000/admin)
- ✅ Database migrations applied: `python manage.py showmigrations`
- ✅ PyTorch CUDA available (if GPU): `python -c "import torch; print(torch.cuda.is_available())"`
- ✅ YOLO models downloaded: Check `~/.cache/ultralytics/`
- ✅ RTSP camera connects: Test with VLC or Python script
- ✅ Detection worker starts: Access Monitor page
- ✅ MODE indicator shows: Check logs for `[MODE: ML_SERVICE]` or `[MODE: LOCAL]`
- ✅ Events are logged: Check Event Logs after detection

---

## Quick Start Commands

**Development Workflow**:
```bash
# Terminal 1: Start ML Service
cd ml_service
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2: Start Django
cd Hotel-Cash-Detector
source venv/bin/activate
python manage.py runserver 0.0.0.0:8000

# 3. Open browser: http://localhost:8000
# 4. Log in with admin credentials
# 5. Add camera and start monitoring
# 6. Check logs for [MODE: ML_SERVICE] indicator
```

---

## Related Documentation

- [00 - Overview](00-overview.md) - System overview
- [03 - Environment Variables](03-env.md) - Complete .env reference
- [06 - Integrations](06-integrations.md) - RTSP, GPU, PMS integration
- [09 - Troubleshooting](09-troubleshooting.md) - Common issues

---

**Previous**: [← 00 - Overview](00-overview.md) | **Next**: [03 - Environment Variables →](03-env.md)
