# Integrations - Hotel Cash Detector

Complete reference for all external system integrations.

## Table of Contents

1. [Integration Overview](#integration-overview)
2. [ML Service Integration](#ml-service-integration)
3. [HotelPMS Integration](#hotelpms-integration)
4. [RTSP Camera Integration](#rtsp-camera-integration)
5. [GPU/CUDA Integration](#gpucuda-integration)
6. [Troubleshooting](#troubleshooting)

---

## Integration Overview

The CCTV system integrates with four main components:

```mermaid
graph TB
    subgraph CCTV["Hotel Cash Detector"]
        Django[Django Backend<br/>port 8000]
        MLProxy[MLDetectorProxy]
        Workers[Background Workers]
    end

    subgraph Services["Internal Services"]
        MLService[ML Service<br/>FastAPI port 8001]
        Detector[AI Detectors]
    end

    subgraph External["External Systems"]
        PMS[HotelPMS<br/>Central Auth]
        Cameras[RTSP Cameras<br/>8+ Streams]
        GPU[GPU/CUDA<br/>AI Acceleration]
    end

    Django -.->|"Optional<br/>Project Sync"| PMS
    Workers -->|"RTSP/TCP<br/>Video Streams"| Cameras
    MLProxy -->|"REST API<br/>HTTP"| MLService
    MLService --> Detector
    Detector -->|"CUDA<br/>Inference"| GPU

    style CCTV fill:#e8f5e9
    style Services fill:#e0f7fa
    style External fill:#ffe0b2
```

| Integration | Purpose | Required | Protocol |
|------------|---------|----------|----------|
| ML Service | AI detection microservice | ⚠️ Recommended | REST API (HTTP) |
| HotelPMS | Central authentication, project sync | ❌ Optional | REST API |
| RTSP Cameras | Real-time video streams | ✅ Yes | RTSP over TCP |
| GPU/CUDA | AI model acceleration | ⚠️ Recommended | CUDA API |

---

## ML Service Integration

### Overview

The **ML Service** is a standalone FastAPI microservice that handles all AI detection (cash, violence, fire). Django communicates with it via REST API through the `MLDetectorProxy` class.

### Architecture

```mermaid
sequenceDiagram
    participant Worker as Django BackgroundWorker
    participant Proxy as MLDetectorProxy
    participant ML as ML Service (port 8001)
    participant Local as Local UnifiedDetector

    Worker->>Proxy: Initialize(camera_id, config)
    Proxy->>ML: POST /detectors/{camera_id}/initialize
    alt ML Service Available
        ML-->>Proxy: 200 OK
        Proxy->>Proxy: Set MODE = ML_SERVICE
    else ML Service Down
        Proxy->>Local: Load UnifiedDetector
        Proxy->>Proxy: Set MODE = LOCAL
    end

    loop Every frame
        Worker->>Proxy: process_frame(frame)
        alt Using ML Service
            Proxy->>ML: POST /detect (base64 frame)
            ML-->>Proxy: DetectionResponse
        else Using Local
            Proxy->>Local: process_frame(frame)
            Local-->>Proxy: Detection result
        end
        Proxy-->>Worker: Detections + alerts
    end

    Note over Proxy: Every 100 frames (when LOCAL):<br/>Try reconnecting to ML service
```

### Configuration

**Environment Variables**:
```env
# In Django .env
USE_ML_SERVICE=True
ML_SERVICE_URL=http://localhost:8001
ML_SERVICE_TIMEOUT=30
```

### ML Service API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Service status + GPU info |
| `/detectors/{camera_id}/initialize` | POST | Initialize detector for camera |
| `/detectors/{camera_id}` | DELETE | Remove detector |
| `/detectors` | GET | List active detectors |
| `/detect` | POST | Process single frame |
| `/detect/batch` | POST | Process multiple frames |
| `/validate` | POST | Gemini image validation |
| `/validate/video` | POST | Gemini video validation |
| `/detectors/{camera_id}/zones` | POST | Update zone polygons |

### Auto-Fallback Behavior

The `MLDetectorProxy` (in `cctv/ml_client.py`) provides transparent fallback:

1. **Startup**: Tries to connect to ML service first
2. **ML Service Available**: All frames sent to ML service via HTTP (`[MODE: ML_SERVICE]`)
3. **ML Service Fails**: Automatically loads local `UnifiedDetector` (`[MODE: LOCAL]`)
4. **Periodic Reconnect**: Every 100 frames, checks if ML service recovered
5. **Recovery**: Switches back to ML service when available (`[MODE: ML_SERVICE] ML service recovered`)

### Key Files

| File | Purpose |
|------|---------|
| `cctv/ml_client.py` | `MLServiceClient` (HTTP) + `MLDetectorProxy` (fallback logic) |
| `ml_service/app/main.py` | FastAPI application entry point |
| `ml_service/app/api/routes.py` | API endpoint handlers |
| `ml_service/app/services/detector_manager.py` | Detector lifecycle management |
| `ml_service/app/models/schemas.py` | Pydantic request/response schemas |

### Testing ML Service

```bash
# Health check
curl http://localhost:8001/health

# Service status
curl http://localhost:8001/status

# List active detectors
curl http://localhost:8001/detectors
```

---

## HotelPMS Integration

### Overview

**HotelPMS** can serve as the **central authentication provider** for the CCTV system. Projects (branches) created in PMS can be synchronized to the CCTV system.

### Architecture

```mermaid
sequenceDiagram
    participant Admin as PMS Admin
    participant PMS as HotelPMS
    participant CCTV as CCTV System
    participant DB as CCTV Database

    Admin->>PMS: Create/update project
    PMS->>PMS: Save to PMS database
    PMS->>CCTV: POST /api/pms/sync-project/<br/>(api_key, project data)
    CCTV->>CCTV: Verify API key
    CCTV->>DB: INSERT/UPDATE Branch
    CCTV-->>PMS: {success: true}
    PMS-->>Admin: Project synced
```

### Configuration

**Environment Variables**:
```env
# In CCTV .env
PMS_API_URL=http://localhost:8000
PMS_API_KEY=your-pms-api-key
```

### API Endpoints

#### Project Sync Webhook

**Endpoint**: `POST /api/pms/sync-project/`

**Request** (from PMS):
```json
{
  "api_key": "your-pms-api-key",
  "project": {
    "id": "uuid",
    "name": "Seoul Hotel",
    "type": "hotel",
    "address": "123 Seoul Street",
    "total_rooms": 50,
    "status": "confirmed"
  }
}
```

**Response**:
```json
{
  "success": true,
  "branch_id": 24,
  "message": "Project synced as branch 'Seoul Hotel'"
}
```

**CCTV Implementation**:
```python
# cctv/views.py
@csrf_exempt
def pms_sync_project(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    data = json.loads(request.body)

    # Verify API key
    if data.get('api_key') != os.getenv('PMS_API_KEY'):
        return JsonResponse({'error': 'Invalid API key'}, status=401)

    project = data.get('project')

    # Create or update branch
    region, _ = Region.objects.get_or_create(name='Default', code='DEF')

    branch, created = Branch.objects.update_or_create(
        id=project['id'],
        defaults={
            'name': project['name'],
            'region': region,
            'address': project.get('address', ''),
            'status': project.get('status', 'confirmed')
        }
    )

    return JsonResponse({
        'success': True,
        'branch_id': branch.id,
        'message': f"Project synced as branch '{branch.name}'"
    })
```

### Authentication (Optional)

**CCTV can use PMS for user authentication**:

**Login Flow**:
1. User enters credentials in CCTV login
2. CCTV sends to PMS: `POST /api/v1/auth/login`
3. PMS verifies credentials
4. PMS returns JWT token
5. CCTV stores token, creates local user session

**Implementation**:
```python
import requests

def authenticate_with_pms(email, password):
    pms_url = os.getenv('PMS_API_URL')
    response = requests.post(
        f'{pms_url}/api/v1/auth/login',
        json={'email': email, 'password': password}
    )

    if response.status_code == 200:
        data = response.json()
        # Check user has CCTV access
        if 'cctv' in data['user']['allowed_systems']:
            return data['user']

    return None
```

---

## RTSP Camera Integration

### Overview

**RTSP (Real-Time Streaming Protocol)** is used to connect to IP cameras for video streams.

### RTSP URL Format

```
rtsp://username:password@ip:port/stream/path
```

**Examples**:
```
# Hikvision
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101

# Dahua
rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0

# Uniview
rtsp://admin:password@192.168.1.120:554/unicast/c1/s0/live

# Generic
rtsp://admin:adminadmin!@175.213.55.16:554
```

### Connection Configuration

**TCP Transport (Recommended)**:

```python
import os
import cv2

def create_rtsp_capture(rtsp_url):
    """Create RTSP capture with TCP transport for stability."""

    # FFmpeg options for stable RTSP streaming
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
        'rtsp_transport;tcp|'              # Use TCP (more reliable)
        'stimeout;60000000|'                # 60s socket timeout
        'max_delay;1000000|'                # 1s max frame delay
        'fflags;nobuffer+discardcorrupt|'  # Low latency, handle errors
        'analyzeduration;2000000|'          # 2s to analyze stream
        'probesize;2000000|'                # 2MB probe size
        'buffer_size;4096000'               # 4MB network buffer
    )

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # OpenCV timeout properties
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)  # 30s connection
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)  # 15s read
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)             # Buffer 5 frames

    return cap
```

### Connection Retry Logic

```python
def connect_with_retry(rtsp_url, max_retries=5):
    """Connect to RTSP with automatic retry."""

    for attempt in range(max_retries):
        print(f"Connecting... (attempt {attempt + 1}/{max_retries})")

        cap = create_rtsp_capture(rtsp_url)

        if cap.isOpened():
            # Test read to verify stream works
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ Connected successfully")
                return cap
            else:
                cap.release()

        time.sleep(5)  # Wait before retry

    return None  # Connection failed
```

### Automatic Reconnection

```python
def worker_main_loop(camera):
    """Main worker loop with automatic reconnection."""

    cap = connect_with_retry(camera.rtsp_url)
    if cap is None:
        print(f"❌ Failed to connect after retries")
        return

    consecutive_failures = 0
    max_failures = 20
    last_success_time = time.time()

    while running:
        ret, frame = cap.read()

        if not ret or frame is None:
            consecutive_failures += 1
            time_since_success = time.time() - last_success_time

            # Reconnect if too many failures or 30s without frames
            if consecutive_failures >= max_failures or time_since_success > 30:
                print(f"📡 Stream lost, reconnecting...")
                cap.release()
                time.sleep(3)

                cap = connect_with_retry(camera.rtsp_url)
                if cap and cap.isOpened():
                    consecutive_failures = 0
                    last_success_time = time.time()
                    print(f"✅ Reconnected")
            continue

        # Successful frame read
        consecutive_failures = 0
        last_success_time = time.time()

        # Process frame...
```

### Camera Setup Checklist

- [ ] Camera supports RTSP
- [ ] RTSP URL is correct (test with VLC or ffplay)
- [ ] Network connectivity (ping camera IP)
- [ ] Firewall allows port 554 (RTSP)
- [ ] Camera stream resolution (720p or 1080p)
- [ ] Camera FPS (15+ recommended, 25-30 ideal)

### Testing Camera Connection

**Using VLC**:
1. Open VLC Media Player
2. Media → Open Network Stream
3. Enter RTSP URL
4. Click Play

**Using FFplay** (if FFmpeg installed):
```bash
ffplay rtsp://admin:password@192.168.1.64:554/stream
```

**Using Python**:
```python
import cv2

rtsp_url = "rtsp://admin:password@192.168.1.64:554/stream"
cap = cv2.VideoCapture(rtsp_url)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"✅ Camera connected: {frame.shape}")
        cv2.imwrite('test_frame.jpg', frame)
    else:
        print("❌ Could not read frame")
else:
    print("❌ Could not open stream")

cap.release()
```

---

## GPU/CUDA Integration

### Overview

**CUDA** enables GPU acceleration for AI model inference, providing 10-20x speedup compared to CPU.

### Requirements

**Hardware**:
- NVIDIA GPU with CUDA support
- Compute Capability 3.5+ (GTX 750+, Tesla K40+)
- Recommended: GTX 1650+, Tesla T4, RTX series

**Software**:
- CUDA Toolkit (11.0+)
- cuDNN (compatible with CUDA version)
- PyTorch with CUDA support

### Check CUDA Availability

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

### Model Initialization

```python
from ultralytics import YOLO
import torch

# Determine device
if os.getenv('USE_GPU', 'auto').lower() == 'cuda':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif os.getenv('USE_GPU', 'auto').lower() == 'cpu':
    device = 'cpu'
else:  # auto
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# Load models on GPU
yolo_model = YOLO('models/yolov8s.pt').to(device)
pose_model = YOLO('models/yolov8s-pose.pt').to(device)
fire_model = YOLO('models/fire_smoke_yolov8.pt').to(device)
```

### Inference

```python
# Run inference on GPU
results = pose_model(frame, device=device, verbose=False)

# Results are automatically on GPU
# Convert to CPU for OpenCV operations
for r in results:
    keypoints = r.keypoints.xy.cpu().numpy()  # Move to CPU
    boxes = r.boxes.xyxy.cpu().numpy()        # Move to CPU
```

### Performance Benchmarks

| Hardware | FPS (1080p) | Inference Time |
|----------|-------------|----------------|
| **i7 CPU** | 5-8 | ~150ms |
| **GTX 1650** | 25-30 | ~30ms |
| **GTX 1660** | 30+ | ~20ms |
| **RTX 3060** | 30+ | ~15ms |
| **Tesla T4** | 30+ | ~20ms |

### GPU Memory Management

```python
import torch

# Clear GPU cache periodically
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Monitor GPU memory
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

---

## Troubleshooting

### PMS Integration Issues

**Cannot connect to PMS**:
1. Verify PMS is running: `curl http://localhost:8000/docs`
2. Check `PMS_API_URL` in `.env`
3. Check network connectivity

**API key invalid**:
- Verify `PMS_API_KEY` matches on both systems
- Check for extra spaces or quotes

### RTSP Camera Issues

**Connection timeout**:
```
Stream timeout triggered after 30043.255000 ms
```

**Solution**: Extended timeout configured (60s), automatic reconnection implemented.

**Frames showing 0**:
- Check RTSP URL is correct
- Test with VLC or ffplay
- Verify network connectivity: `ping camera-ip`
- Check firewall allows port 554

**Poor video quality**:
- Check camera resolution settings
- Verify network bandwidth (1-5 Mbps per camera)
- Reduce number of cameras if bandwidth limited

### GPU/CUDA Issues

**CUDA not available**:
```python
torch.cuda.is_available()  # Returns False
```

**Solutions**:
1. Install CUDA Toolkit
2. Install compatible PyTorch:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Verify GPU drivers: `nvidia-smi`

**Out of GPU memory**:
```
CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions**:
- Use smaller models (`yolov8n.pt` instead of `yolov8s.pt`)
- Reduce number of simultaneous cameras
- Clear GPU cache: `torch.cuda.empty_cache()`
- Restart workers

---

## Related Documentation

- [02 - Setup](02-setup.md) - Development setup
- [03 - Environment Variables](03-env.md) - Configuration
- [09 - Troubleshooting](09-troubleshooting.md) - Common issues

---

**Previous**: [← 05 - Data Models](05-data-models.md) | **Next**: [07 - Flows →](07-flows.md)
