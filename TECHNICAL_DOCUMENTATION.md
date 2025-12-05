# Hotel Cash Detector - Technical Documentation

> **Version:** 1.0.0  
> **Last Updated:** December 5, 2025  
> **Author:** Loop-Dimension  

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Stack](#2-technology-stack)
3. [Architecture](#3-architecture)
4. [Database Schema](#4-database-schema)
5. [Detection Models](#5-detection-models)
6. [Processing Pipeline](#6-processing-pipeline)
7. [API Endpoints](#7-api-endpoints)
8. [Configuration](#8-configuration)
9. [Deployment](#9-deployment)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. System Overview

### 1.1 Purpose
The Hotel Cash Detector is an AI-powered CCTV monitoring system designed for hotel cashier surveillance. It detects:
- **Cash Transactions** - Hand-to-hand exchanges between cashier and customer
- **Violence/Disturbances** - Physical altercations or aggressive behavior
- **Fire/Smoke** - Fire and smoke detection for safety

### 1.2 Key Features
- Real-time RTSP video stream processing
- Multi-camera support with individual settings
- Background detection workers (continuous monitoring)
- Event logging with video clip recording
- Multi-language support (English, Korean, Thai, Vietnamese, Chinese)
- Role-based access control (Admin, Project Manager)
- Developer mode for debugging and tuning

---

## 2. Technology Stack

### 2.1 Backend Framework

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Web Framework | Django | 5.2.7 | Main web application, ORM, admin |
| Python | Python | 3.10+ | Core programming language |
| ASGI Server | Daphne/Uvicorn | Latest | Async request handling |
| Task Queue | Threading | Built-in | Background detection workers |

### 2.2 Frontend

| Component | Technology | Purpose |
|-----------|------------|---------|
| Templates | Django Templates | Server-side rendering |
| Styling | Custom CSS | Dark theme UI |
| JavaScript | Vanilla JS | Interactive components |
| Video | HTML5 Video + MJPEG | Live stream display |

### 2.3 AI/ML Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Deep Learning | PyTorch | 2.0+ | Model inference backend |
| Object Detection | Ultralytics YOLOv8 | 8.0+ | Person, fire detection |
| Pose Estimation | YOLOv8-Pose | 8.0+ | Hand position tracking |
| Computer Vision | OpenCV | 4.8+ | Frame processing, video I/O |
| Array Operations | NumPy | 1.24+ | Numerical computations |

### 2.4 Database

| Component | Technology | Purpose |
|-----------|------------|---------|
| Database | SQLite3 | Development database |
| ORM | Django ORM | Database abstraction |
| Migrations | Django Migrations | Schema versioning |

### 2.5 Video Processing

| Component | Technology | Purpose |
|-----------|------------|---------|
| Stream Protocol | RTSP over TCP | Camera connection |
| Video Codec | H.264 (libx264) | Clip encoding |
| Transcoding | FFmpeg | Video conversion |
| Container | MP4 (faststart) | Web-compatible video |

---

## 3. Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Browser   │  │   Mobile    │  │   Admin Dashboard       │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DJANGO WEB LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    Views    │  │     API     │  │    Video Streaming      │  │
│  │  (HTML)     │  │  (JSON)     │  │    (MJPEG/MP4)          │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
└─────────┼────────────────┼─────────────────────┼────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKGROUND WORKERS                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              BackgroundCameraWorker (per camera)            ││
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐││
│  │  │  RTSP    │─▶│   Unified    │─▶│   Event/Clip Saving    │││
│  │  │ Capture  │  │  Detector    │  │   (async)              │││
│  │  └──────────┘  └──────────────┘  └────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DETECTION LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    Cash     │  │  Violence   │  │         Fire            │  │
│  │  Detector   │  │  Detector   │  │       Detector          │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                     │                 │
│         └────────────────┼─────────────────────┘                 │
│                          ▼                                       │
│                   ┌─────────────┐                                │
│                   │ YOLOv8 +    │                                │
│                   │ YOLOv8-Pose │                                │
│                   └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   SQLite    │  │   Media     │  │       Models            │  │
│  │  Database   │  │   Files     │  │   (YOLO weights)        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
Hotel-Cash-Detector/
├── django_app/                    # Main Django application
│   ├── manage.py                  # Django management script
│   ├── db.sqlite3                 # SQLite database
│   ├── hotel_cctv/                # Django project settings
│   │   ├── settings.py            # Configuration
│   │   ├── urls.py                # Root URL routing
│   │   ├── wsgi.py                # WSGI entry point
│   │   └── asgi.py                # ASGI entry point
│   ├── cctv/                      # Main application
│   │   ├── models.py              # Database models
│   │   ├── views.py               # Views and API endpoints
│   │   ├── urls.py                # App URL routing
│   │   ├── admin.py               # Admin configuration
│   │   ├── translations.py        # Multi-language support
│   │   └── context_processors.py  # Template context
│   ├── templates/cctv/            # HTML templates
│   │   ├── base.html              # Base template
│   │   ├── home.html              # Dashboard
│   │   ├── monitor_all.html       # Multi-camera view
│   │   ├── monitor_local.html     # Single camera view
│   │   ├── camera_settings.html   # Camera configuration
│   │   ├── video_logs.html        # Event logs
│   │   └── ...
│   ├── static/                    # Static assets
│   │   ├── css/style.css          # Styles
│   │   └── js/main.js             # JavaScript
│   ├── media/                     # User uploads
│   │   ├── clips/                 # Event video clips
│   │   └── thumbnails/            # Event thumbnails
│   └── models/                    # AI model weights
│       ├── yolov8n.pt             # YOLOv8 Nano
│       ├── yolov8n-pose.pt        # YOLOv8 Pose
│       └── fire_smoke_yolov8.pt   # Fire/smoke model
│
└── flask/                         # Detection modules (shared)
    ├── detectors/                 # Detection algorithms
    │   ├── base_detector.py       # Base class
    │   ├── unified_detector.py    # Main detector
    │   ├── cash_detector.py       # Cash detection
    │   ├── violence_detector.py   # Violence detection
    │   └── fire_detector.py       # Fire detection
    └── models/                    # Model weights (duplicate)
```

---

## 4. Database Schema

### 4.1 Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│      User       │       │     Region      │       │     Branch      │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ id (PK)         │       │ id (PK)         │       │ id (PK)         │
│ username        │       │ name            │       │ name            │
│ email           │       │ code            │       │ region_id (FK)  │
│ password        │       └────────┬────────┘       │ address         │
│ role            │                │                │ status          │
│ phone           │                │                │ created_at      │
└────────┬────────┘                │                └────────┬────────┘
         │                         │                         │
         │    ┌────────────────────┘                         │
         │    │                                              │
         ▼    ▼                                              ▼
┌─────────────────────────┐                    ┌─────────────────────────┐
│   managers (M2M)        │                    │        Camera           │
│   Branch ←──────────────┼────────────────────├─────────────────────────┤
│        → User           │                    │ id (PK)                 │
└─────────────────────────┘                    │ branch_id (FK)          │
                                               │ camera_id               │
                                               │ name                    │
                                               │ rtsp_url                │
                                               │ status                  │
                                               │ cashier_zone_*          │
                                               │ cash_confidence         │
                                               │ violence_confidence     │
                                               │ fire_confidence         │
                                               │ hand_touch_distance     │
                                               │ detect_cash             │
                                               │ detect_violence         │
                                               │ detect_fire             │
                                               └───────────┬─────────────┘
                                                           │
                                                           ▼
                                               ┌─────────────────────────┐
                                               │         Event           │
                                               ├─────────────────────────┤
                                               │ id (PK)                 │
                                               │ branch_id (FK)          │
                                               │ camera_id (FK)          │
                                               │ event_type              │
                                               │ status                  │
                                               │ confidence              │
                                               │ frame_number            │
                                               │ bbox_*                  │
                                               │ clip_path               │
                                               │ thumbnail_path          │
                                               │ notes                   │
                                               │ reviewed_by (FK)        │
                                               │ created_at              │
                                               └─────────────────────────┘
```

### 4.2 Model Definitions

#### User Model
```python
class User(AbstractUser):
    ROLE_CHOICES = [
        ('admin', 'Admin (Master)'),
        ('project_manager', 'Project Manager'),
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    phone = models.CharField(max_length=20, blank=True, null=True)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | AutoField | Primary key |
| `username` | CharField | Login username |
| `email` | EmailField | Email address |
| `password` | CharField | Hashed password |
| `role` | CharField | `admin` or `project_manager` |
| `phone` | CharField | Phone number (optional) |

#### Region Model
```python
class Region(models.Model):
    name = models.CharField(max_length=50, unique=True)
    code = models.CharField(max_length=10, unique=True)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | AutoField | Primary key |
| `name` | CharField | Region name (e.g., "Bangkok") |
| `code` | CharField | Short code (e.g., "BKK") |

#### Branch Model
```python
class Branch(models.Model):
    name = models.CharField(max_length=100)
    region = models.ForeignKey(Region, on_delete=models.CASCADE)
    address = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    managers = models.ManyToManyField(User, related_name='managed_branches')
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | AutoField | Primary key |
| `name` | CharField | Branch name |
| `region_id` | ForeignKey | Reference to Region |
| `address` | TextField | Physical address |
| `status` | CharField | `confirmed`, `reviewing`, `pending` |
| `managers` | ManyToMany | Assigned project managers |

#### Camera Model
```python
class Camera(models.Model):
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    camera_id = models.CharField(max_length=50)
    name = models.CharField(max_length=100)
    rtsp_url = models.CharField(max_length=500)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    
    # Cashier zone coordinates
    cashier_zone_x = models.IntegerField(default=0)
    cashier_zone_y = models.IntegerField(default=0)
    cashier_zone_width = models.IntegerField(default=640)
    cashier_zone_height = models.IntegerField(default=480)
    cashier_zone_enabled = models.BooleanField(default=False)
    
    # Detection thresholds
    cash_confidence = models.FloatField(default=0.5)
    violence_confidence = models.FloatField(default=0.6)
    fire_confidence = models.FloatField(default=0.5)
    hand_touch_distance = models.IntegerField(default=100)
    
    # Detection toggles
    detect_cash = models.BooleanField(default=True)
    detect_violence = models.BooleanField(default=True)
    detect_fire = models.BooleanField(default=True)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | AutoField | Primary key |
| `branch_id` | ForeignKey | Reference to Branch |
| `camera_id` | CharField | Unique camera identifier |
| `name` | CharField | Display name |
| `rtsp_url` | CharField | RTSP stream URL |
| `status` | CharField | `online`, `offline`, `maintenance` |
| `cashier_zone_*` | Integer | Zone coordinates (x, y, width, height) |
| `cash_confidence` | Float | Cash detection threshold (0.0-1.0) |
| `violence_confidence` | Float | Violence detection threshold |
| `fire_confidence` | Float | Fire detection threshold |
| `hand_touch_distance` | Integer | Max pixels between hands for detection |

#### Event Model
```python
class Event(models.Model):
    TYPE_CHOICES = [
        ('cash', '현금'),
        ('fire', '화재'),
        ('violence', '난동'),
    ]
    
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE)
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE)
    event_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    confidence = models.FloatField(default=0.0)
    clip_path = models.CharField(max_length=500, blank=True, null=True)
    thumbnail_path = models.CharField(max_length=500, blank=True, null=True)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | AutoField | Primary key |
| `branch_id` | ForeignKey | Reference to Branch |
| `camera_id` | ForeignKey | Reference to Camera |
| `event_type` | CharField | `cash`, `violence`, `fire` |
| `status` | CharField | `confirmed`, `reviewing`, `pending` |
| `confidence` | Float | Detection confidence (0.0-1.0) |
| `frame_number` | Integer | Frame number in stream |
| `bbox_*` | Integer | Bounding box coordinates |
| `clip_path` | CharField | Path to video clip |
| `thumbnail_path` | CharField | Path to thumbnail image |
| `reviewed_by` | ForeignKey | User who reviewed |
| `created_at` | DateTime | Event timestamp |

---

## 5. Detection Models

### 5.1 Model Overview

| Model | File | Size | Purpose |
|-------|------|------|---------|
| YOLOv8n | `yolov8n.pt` | ~6 MB | Person detection (backup) |
| YOLOv8n-Pose | `yolov8n-pose.pt` | ~7 MB | Pose estimation (17 keypoints) |
| YOLOv8s-Pose | `yolov8s-pose.pt` | ~23 MB | Higher accuracy pose (optional) |
| Fire/Smoke | `fire_smoke_yolov8.pt` | ~6 MB | Fire and smoke detection |

### 5.2 Cash Transaction Detection

**Algorithm:** Pose-based hand proximity detection

```
┌────────────────────────────────────────────────────────────┐
│                 CASH DETECTION PIPELINE                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. FRAME INPUT                                            │
│     └── RTSP stream frame (1920x1080 typical)              │
│                                                            │
│  2. POSE ESTIMATION (YOLOv8-Pose)                          │
│     └── Detect all people in frame                         │
│     └── Extract 17 keypoints per person                    │
│     └── Focus on wrists: LEFT_WRIST(9), RIGHT_WRIST(10)    │
│                                                            │
│  3. ZONE CLASSIFICATION                                    │
│     └── Check if person bbox overlaps cashier zone         │
│     └── Classify as: CASHIER (in zone) or CUSTOMER (out)   │
│                                                            │
│  4. HAND PROXIMITY CHECK                                   │
│     └── For each cashier-customer pair:                    │
│         └── Calculate distance between all hand combos     │
│         └── distance = √((x1-x2)² + (y1-y2)²)              │
│                                                            │
│  5. DETECTION CRITERIA                                     │
│     └── ONE person IN cashier zone (cashier)               │
│     └── ONE person OUTSIDE zone (customer)                 │
│     └── Hand distance < hand_touch_distance threshold      │
│                                                            │
│  6. EVENT GENERATION                                       │
│     └── If criteria met → Generate CASH event              │
│     └── Save 30-second video clip                          │
│     └── Create thumbnail from detection frame              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Key Parameters:**
- `hand_touch_distance`: Maximum pixel distance between hands (default: 100px)
- `pose_confidence`: Minimum keypoint confidence (default: 0.3)
- `min_transaction_frames`: Frames before confirming (default: 1)
- `transaction_cooldown`: Frames between detections (default: 45)

**Keypoint Indices (COCO Format):**
```
0: nose          5: left_shoulder   10: right_wrist
1: left_eye      6: right_shoulder  11: left_hip
2: right_eye     7: left_elbow      12: right_hip
3: left_ear      8: right_elbow     13: left_knee
4: right_ear     9: left_wrist      14: right_knee
                                    15: left_ankle
                                    16: right_ankle
```

### 5.3 Violence Detection

**Algorithm:** Pose-based close combat detection

```
┌────────────────────────────────────────────────────────────┐
│               VIOLENCE DETECTION PIPELINE                   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. POSE ESTIMATION                                        │
│     └── Detect all people with keypoints                   │
│                                                            │
│  2. CLOSE COMBAT DETECTION                                 │
│     └── Find pairs of people in close proximity            │
│     └── Check for overlapping bounding boxes               │
│     └── Calculate inter-person distance                    │
│                                                            │
│  3. AGGRESSIVE POSE ANALYSIS                               │
│     └── Check arm positions (raised/swinging)              │
│     └── Detect rapid motion between frames                 │
│     └── Both people must show aggressive indicators        │
│                                                            │
│  4. SUSTAINED DETECTION                                    │
│     └── Require min_violence_frames consecutive frames     │
│     └── Confidence must meet threshold                     │
│                                                            │
│  5. EXCLUSIONS                                             │
│     └── Ignore cashier zone (normal transactions)          │
│     └── Single person actions NOT violence                 │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Key Parameters:**
- `violence_confidence`: Detection threshold (default: 0.8)
- `min_violence_frames`: Consecutive frames required (default: 15)
- `motion_threshold`: Motion magnitude threshold (default: 100)
- `violence_cooldown`: Frames between alerts (default: 90)

### 5.4 Fire Detection

**Algorithm:** YOLO + Color-based detection with flickering analysis

```
┌────────────────────────────────────────────────────────────┐
│                 FIRE DETECTION PIPELINE                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  PRIMARY METHOD: YOLO Fire/Smoke Model                     │
│  ├── Classes: {0: 'Fire', 1: 'default', 2: 'smoke'}        │
│  ├── Run inference with conf=0.25                          │
│  └── Filter by fire_confidence threshold                   │
│                                                            │
│  FALLBACK METHOD: Color-Based Detection                    │
│  ├── Convert to HSV color space                            │
│  ├── Fire colors: Bright orange/yellow (H:5-25, S:150+)    │
│  ├── Exclude skin tones (prevent false positives)          │
│  ├── Analyze flickering (temporal variation)               │
│  └── Require significant area + flickering score           │
│                                                            │
│  SMOKE DETECTION:                                          │
│  ├── Background subtraction (MOG2)                         │
│  ├── Gray/white color mask                                 │
│  └── Motion detection for rising smoke                     │
│                                                            │
│  CONFIRMATION:                                             │
│  ├── min_fire_frames consecutive detections                │
│  ├── Confidence meets camera threshold                     │
│  └── fire_cooldown between alerts                          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Key Parameters:**
- `fire_confidence`: Detection threshold (default: 0.7)
- `min_fire_frames`: Consecutive frames required (default: 10)
- `min_fire_area`: Minimum fire region area (default: 3000 px²)
- `fire_cooldown`: Frames between alerts (default: 60)

**Color Ranges (HSV):**
```python
# Fire colors (very bright orange/yellow)
fire_lower1 = [5, 150, 200]    fire_upper1 = [25, 255, 255]
fire_lower2 = [0, 200, 220]    fire_upper2 = [5, 255, 255]

# Skin exclusion (to prevent false positives)
skin_lower = [0, 20, 70]       skin_upper = [25, 170, 200]

# Smoke (gray/white)
smoke_lower = [0, 0, 150]      smoke_upper = [180, 30, 255]
```

---

## 6. Processing Pipeline

### 6.1 Background Worker Architecture

```python
class BackgroundCameraWorker:
    """Continuous detection worker for each camera"""
    
    def __init__(self, camera, models_dir, output_dir):
        self.camera_id = camera.id
        self.detector = None          # UnifiedDetector instance
        self.clip_buffer = []         # Last 30 seconds of frames
        self.clip_buffer_size = 900   # 30 sec × 30 fps
        self.event_cooldown = 30      # Seconds between events
```

### 6.2 Frame Processing Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                    FRAME PROCESSING LOOP                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  while worker.running:                                           │
│  │                                                               │
│  ├─► 1. CAPTURE FRAME                                            │
│  │      cap.read() from RTSP stream                              │
│  │      └── TCP transport for stability                          │
│  │                                                               │
│  ├─► 2. PROCESS WITH DETECTOR                                    │
│  │      detector.process_frame(frame, draw_overlay=True)         │
│  │      └── Returns: frame_with_overlay, detections[]            │
│  │                                                               │
│  ├─► 3. BUFFER FRAME                                             │
│  │      clip_buffer.append(frame_with_overlay)                   │
│  │      └── Keep last 900 frames (30 seconds)                    │
│  │                                                               │
│  ├─► 4. UPDATE SHARED FRAME                                      │
│  │      current_frame = frame  (for live viewing)                │
│  │      current_frame_with_overlay = frame_with_overlay          │
│  │                                                               │
│  ├─► 5. PROCESS DETECTIONS                                       │
│  │      for detection in detections:                             │
│  │      │   └── event_type = 'cash' | 'violence' | 'fire'        │
│  │      │                                                        │
│  │      ├─► 5a. SAVE VIDEO CLIP                                  │
│  │      │       save_clip(clip_buffer, camera, event_type)       │
│  │      │       └── MJPG temp → FFmpeg H.264 → MP4               │
│  │      │                                                        │
│  │      └─► 5b. SAVE EVENT TO DATABASE                           │
│  │              Event.objects.create(...)                        │
│  │                                                               │
│  └─► 6. SLEEP (throttle to ~30 FPS)                              │
│         time.sleep(0.01)                                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 6.3 Video Clip Saving

```python
def save_clip(frames, camera, detection_type, fps=30):
    """
    1. Create temp AVI with MJPG codec (fast, reliable)
    2. Convert to H.264 MP4 with FFmpeg
    3. Save thumbnail from last frame
    4. Return paths for database storage
    """
    
    # Step 1: Write temp file
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    
    # Step 2: Convert to H.264
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-r', str(fps),
        final_path
    ])
    
    # Step 3: Save thumbnail
    cv2.imwrite(thumb_path, frames[-1])
    
    return clip_url, thumb_url
```

### 6.4 RTSP Connection

```python
def _create_rtsp_capture(self, rtsp_url):
    """Create RTSP capture with TCP transport"""
    
    # Use TCP to avoid UDP packet loss/reordering
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000'
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10s timeout
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Low latency
    
    return cap
```

---

## 7. API Endpoints

### 7.1 Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/login/` | Login page |
| POST | `/login/` | Authenticate user |
| GET | `/logout/` | Logout user |

### 7.2 Dashboard & Views

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home dashboard |
| GET | `/monitor/all/` | All cameras view |
| GET | `/monitor/local/<id>/` | Single camera view |
| GET | `/camera/<id>/settings/` | Camera settings page |
| GET | `/video/logs/` | Event logs page |
| GET | `/manage/branches/` | Branch management |

### 7.3 Camera API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cameras/` | List all cameras |
| GET | `/api/cameras/<id>/` | Get camera details |
| PUT | `/api/cameras/<id>/` | Update camera settings |
| POST | `/api/cameras/<id>/zone/` | Update cashier zone |

**Example: Update Camera Settings**
```json
PUT /api/cameras/24/
{
    "cash_confidence": 0.5,
    "violence_confidence": 0.6,
    "fire_confidence": 0.7,
    "hand_touch_distance": 100,
    "detect_cash": true,
    "detect_violence": true,
    "detect_fire": true
}
```

### 7.4 Event API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/events/` | List events with filters |
| GET | `/api/events/<id>/` | Get event details |
| PUT | `/api/events/<id>/` | Update event status |
| DELETE | `/api/events/<id>/` | Delete event |

**Query Parameters:**
- `region` - Filter by region ID
- `branch` - Filter by branch ID
- `type` - Filter by event type (cash/violence/fire)
- `from` - Start date (YYYY-MM-DD)
- `to` - End date (YYYY-MM-DD)

### 7.5 Video Streaming

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/video-feed/<id>/` | MJPEG live stream |
| GET | `/video-feed-hq/<id>/` | High quality stream |
| GET | `/video-feed-raw/<id>/` | Raw stream (no overlay) |

### 7.6 Worker Control

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/workers/status/` | Get all worker status |
| POST | `/api/workers/start-all/` | Start all workers |
| POST | `/api/workers/<id>/start/` | Start specific worker |
| POST | `/api/workers/<id>/stop/` | Stop specific worker |
| POST | `/api/workers/<id>/restart/` | Restart worker |

**Response Example:**
```json
{
    "workers": [
        {
            "camera_id": 24,
            "camera_name": "Lobby Camera",
            "status": "running",
            "uptime": "02:45:30",
            "events_detected": 15,
            "frames_processed": 298500
        }
    ]
}
```

### 7.7 Developer Mode

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/cameras/<id>/dev-mode/verify/` | Enter dev mode |
| GET | `/api/cameras/<id>/dev-mode/status/` | Get dev mode status |
| GET | `/api/cameras/<id>/dev-mode/debug-info/` | Get detection debug info |

---

## 8. Configuration

### 8.1 Django Settings

**File:** `django_app/hotel_cctv/settings.py`

```python
# Security
SECRET_KEY = os.getenv('SECRET_KEY', 'change-in-production')
DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 'yes')
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Static & Media Files
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Detection Settings
DETECTION_CONFIG = {
    'MODELS_DIR': BASE_DIR.parent / 'flask' / 'models',
    'CASH_CONFIDENCE': 0.5,
    'VIOLENCE_CONFIDENCE': 0.6,
    'FIRE_CONFIDENCE': 0.5,
    'HAND_TOUCH_DISTANCE': 100,
    'MIN_TRANSACTION_FRAMES': 1,
}
```

### 8.2 Environment Variables

Create `.env` file in `django_app/`:

```bash
# Django Settings
SECRET_KEY=your-super-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,192.168.1.100

# Detection Defaults
CASH_DETECTION_CONFIDENCE=0.5
VIOLENCE_DETECTION_CONFIDENCE=0.6
FIRE_DETECTION_CONFIDENCE=0.5
HAND_TOUCH_DISTANCE=100
```

### 8.3 Camera-Specific Settings

Each camera has independent settings stored in the database:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `cash_confidence` | Float | 0.5 | Cash detection threshold |
| `violence_confidence` | Float | 0.6 | Violence detection threshold |
| `fire_confidence` | Float | 0.5 | Fire detection threshold |
| `hand_touch_distance` | Integer | 100 | Max hand distance (pixels) |
| `pose_confidence` | Float | 0.3 | Pose keypoint confidence |
| `cashier_zone_*` | Integer | varies | Zone coordinates |
| `detect_cash` | Boolean | True | Enable cash detection |
| `detect_violence` | Boolean | True | Enable violence detection |
| `detect_fire` | Boolean | True | Enable fire detection |

---

## 9. Deployment

### 9.1 Development Setup

```bash
# Clone repository
git clone https://github.com/Loop-Dimension/Hotel-Cash-Detector.git
cd Hotel-Cash-Detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
cd django_app
pip install -r requirements.txt

# Download YOLO models (auto-downloads on first run)
# Or manually place in django_app/models/ or flask/models/

# Run migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Seed demo data (optional)
python manage.py seed_data

# Run development server
python manage.py runserver 0.0.0.0:8000
```

### 9.2 Production Deployment

**Requirements:**
- Python 3.10+
- FFmpeg (for video encoding)
- CUDA-capable GPU (recommended for real-time processing)
- 8GB+ RAM per 4 cameras

**Using Gunicorn + Nginx:**

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn hotel_cctv.wsgi:application --bind 0.0.0.0:8000 --workers 4

# Nginx configuration
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /static/ {
        alias /path/to/django_app/static/;
    }
    
    location /media/ {
        alias /path/to/django_app/media/;
    }
}
```

### 9.3 Docker Deployment

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "hotel_cctv.wsgi:application", "--bind", "0.0.0.0:8000"]
```

### 9.4 AWS Deployment

#### Option A: EC2 Instance (Recommended for GPU)

**1. Launch EC2 Instance**
```bash
# Recommended instance types:
# - 8.xlarge (1 GPU, 4 vCPU, 16GB RAM) - Best for 4-8 cameras
# - g4dn.2xlarge (1 GPU, 8 vCPU, 32GB RAM) - Best for 8-16 cameras
# - p3.2xlarge (1 GPU, 8 vCPU, 61GB RAM) - High performance

# AMI: Deep Learning AMI (Ubuntu) - includes CUDA, cuDNN, PyTorch
```

**2. Security Group Configuration**
```
Inbound Rules:
- SSH (22): Your IP
- HTTP (80): 0.0.0.0/0
- HTTPS (443): 0.0.0.0/0
- Custom TCP (8000): 0.0.0.0/0 (Django dev server)
- RTSP (554): Camera IPs only
```

**3. Install Dependencies**
```bash
# Connect to EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install FFmpeg
sudo apt install -y ffmpeg

# Clone repository
git clone https://github.com/Loop-Dimension/Hotel-Cash-Detector.git
cd Hotel-Cash-Detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
cd django_app
pip install -r requirements.txt
pip install gunicorn

# Download YOLO models
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
```

**4. Configure Environment**
```bash
# Create .env file
cat > .env << EOF
SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
DEBUG=False
ALLOWED_HOSTS=your-domain.com,your-ec2-ip
EOF
```

**5. Setup Systemd Service**
```bash
# Create service file
sudo nano /etc/systemd/system/hotel-cctv.service
```

```ini
[Unit]
Description=Hotel CCTV Detection Service
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/Hotel-Cash-Detector/django_app
Environment="PATH=/home/ubuntu/Hotel-Cash-Detector/venv/bin"
ExecStart=/home/ubuntu/Hotel-Cash-Detector/venv/bin/gunicorn \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    hotel_cctv.wsgi:application
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable hotel-cctv
sudo systemctl start hotel-cctv
```

**6. Setup Nginx Reverse Proxy**
```bash
sudo apt install -y nginx

sudo nano /etc/nginx/sites-available/hotel-cctv
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
    }

    location /static/ {
        alias /home/ubuntu/Hotel-Cash-Detector/django_app/static/;
    }

    location /media/ {
        alias /home/ubuntu/Hotel-Cash-Detector/django_app/media/;
    }

    # For video streaming - increase buffer sizes
    location /video-feed/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/hotel-cctv /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**7. Setup SSL with Let's Encrypt**
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

#### Option B: AWS ECS (Container-based)

**1. Create ECR Repository**
```bash
aws ecr create-repository --repository-name hotel-cctv
```

**2. Build and Push Docker Image**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t hotel-cctv .

# Tag and push
docker tag hotel-cctv:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/hotel-cctv:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/hotel-cctv:latest
```

**3. Create ECS Task Definition**
```json
{
  "family": "hotel-cctv",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "hotel-cctv",
      "image": "YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/hotel-cctv:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "DEBUG", "value": "False"},
        {"name": "ALLOWED_HOSTS", "value": "*"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hotel-cctv",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Option C: AWS Lambda + API Gateway (Serverless - Limited)

> ⚠️ **Note:** Not recommended for real-time video processing. Only suitable for event API and dashboard.

```yaml
# serverless.yml
service: hotel-cctv-api

provider:
  name: aws
  runtime: python3.10
  region: us-east-1

functions:
  api:
    handler: wsgi_handler.handler
    events:
      - http: ANY /
      - http: ANY /{proxy+}
```

### 9.5 AWS Architecture Diagram

```
                                    ┌─────────────────┐
                                    │   CloudFront    │
                                    │   (CDN/SSL)     │
                                    └────────┬────────┘
                                             │
                                             ▼
┌─────────────┐                    ┌─────────────────┐
│   Route 53  │───────────────────▶│      ALB        │
│   (DNS)     │                    │ (Load Balancer) │
└─────────────┘                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
           ┌───────────────┐        ┌───────────────┐        ┌───────────────┐
           │  EC2 (GPU)    │        │  EC2 (GPU)    │        │  EC2 (GPU)    │
           │  Worker 1-4   │        │  Worker 5-8   │        │  Worker 9-12  │
           └───────┬───────┘        └───────┬───────┘        └───────┬───────┘
                   │                        │                        │
                   └────────────────────────┼────────────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
           ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
           │     RDS       │       │      S3       │       │  ElastiCache  │
           │  (PostgreSQL) │       │ (Media/Clips) │       │   (Redis)     │
           └───────────────┘       └───────────────┘       └───────────────┘
```

### 9.6 AWS Cost Estimation

> 💡 Pricing based on AWS Calculator (US East - N. Virginia, December 2024)

#### EC2 Instance Pricing (On-Demand, 730 hours/month)

| Instance Type | vCPU | Memory | GPU | Storage | Hourly Cost | Monthly Cost (USD) |
|---------------|------|--------|-----|---------|-------------|-------------------|
| t3.medium | 2 | 4 GB | - | EBS | $0.0416 | ~$30 |
| t3.large | 2 | 8 GB | - | EBS | $0.0832 | ~$61 |
| t3.xlarge | 4 | 16 GB | - | EBS | $0.1664 | ~$122 |
| m5.large | 2 | 8 GB | - | EBS | $0.096 | ~$70 |
| m5.xlarge | 4 | 16 GB | - | EBS | $0.192 | ~$140 |
| c5.xlarge | 4 | 8 GB | - | EBS | $0.17 | ~$124 |
| **g4dn.xlarge** | 4 | 16 GB | 1x T4 | 125 GB NVMe | $0.526 | **~$384** |
| **g4dn.2xlarge** | 8 | 32 GB | 1x T4 | 225 GB NVMe | $0.752 | **~$549** |
| g4dn.4xlarge | 16 | 64 GB | 1x T4 | 225 GB NVMe | $1.204 | ~$879 |
| g5.xlarge | 4 | 16 GB | 1x A10G | 250 GB NVMe | $1.006 | ~$734 |
| p3.2xlarge | 8 | 61 GB | 1x V100 | EBS | $3.06 | ~$2,234 |

#### EC2 Savings Plans (1-Year, No Upfront)

| Instance Type | On-Demand | Savings Plan | Savings |
|---------------|-----------|--------------|---------|
| g4dn.xlarge | $384/mo | ~$243/mo | 37% |
| g4dn.2xlarge | $549/mo | ~$347/mo | 37% |
| t3.xlarge | $122/mo | ~$77/mo | 37% |

#### EC2 Spot Instances (Variable, Up to 90% Off)

| Instance Type | On-Demand | Spot Price* | Savings |
|---------------|-----------|-------------|---------|
| g4dn.xlarge | $384/mo | ~$115/mo | 70% |
| g4dn.2xlarge | $549/mo | ~$165/mo | 70% |
| t3.xlarge | $122/mo | ~$37/mo | 70% |

*Spot prices vary by region and availability

#### Other AWS Services

| Component | Configuration | Monthly Cost (USD) |
|-----------|---------------|-------------------|
| RDS PostgreSQL | db.t3.micro (Free Tier) | $0 |
| RDS PostgreSQL | db.t3.small (2 vCPU, 2GB) | ~$25 |
| RDS PostgreSQL | db.t3.medium (2 vCPU, 4GB) | ~$50 |
| S3 Storage | 100 GB Standard | ~$2.30 |
| S3 Storage | 500 GB Standard | ~$11.50 |
| S3 Data Transfer | 100 GB/month out | ~$9 |
| CloudFront | 100 GB transfer | ~$8.50 |
| CloudFront | 1 TB transfer | ~$85 |
| ALB | Standard + 1 LCU | ~$22 |
| Elastic IP | 1 IP (in use) | $0 |
| Elastic IP | 1 IP (idle) | ~$3.60 |
| CloudWatch | Basic monitoring | Free |
| CloudWatch | Detailed (1-min) | ~$3/instance |

#### Complete Deployment Scenarios

| Scenario | Components | Monthly Cost |
|----------|------------|--------------|
| **Dev/Test** | t3.medium + RDS Free + 50GB S3 | **~$35/mo** |
| **Small (1-4 cameras)** | g4dn.xlarge + RDS t3.small + 100GB S3 + ALB | **~$440/mo** |
| **Medium (5-10 cameras)** | g4dn.2xlarge + RDS t3.medium + 250GB S3 + ALB + CloudFront | **~$640/mo** |
| **Large (10-20 cameras)** | 2x g4dn.xlarge + RDS t3.large + 500GB S3 + ALB + CloudFront | **~$950/mo** |
| **Enterprise (20+ cameras)** | 3x g4dn.2xlarge + RDS m5.large + 1TB S3 + ALB + CloudFront | **~$1,900/mo** |

#### Cost Optimization Tips

1. **Use Spot Instances** for non-critical workers (70% savings)
2. **Reserved Instances/Savings Plans** for production (37% savings)
3. **Right-size instances** - start small and scale up
4. **Use S3 Intelligent-Tiering** for video clips
5. **Set lifecycle policies** to delete old clips after 30/60/90 days
6. **Use CloudFront caching** to reduce origin requests

### 9.7 Search Functionality

The system includes event search and filtering capabilities:

#### Event Log Filters

```javascript
// Frontend filter application
function applyFilters() {
    const params = new URLSearchParams();
    
    // Region filter (by ID)
    if (region) params.append('region', region);
    
    // Branch filter (by ID)
    if (branch) params.append('branch', branch);
    
    // Event type filter (cash/violence/fire)
    if (eventType) params.append('type', eventType);
    
    // Date range filters
    if (dateFrom) params.append('from', dateFrom);
    if (dateTo) params.append('to', dateTo);
    
    window.location.href = `/video/logs/?${params.toString()}`;
}
```

#### Backend Query Building

```python
# views.py - video_logs()
def video_logs(request):
    events = Event.objects.select_related('branch', 'camera', 'branch__region')
    
    # Filter by region ID
    if region_filter:
        events = events.filter(branch__region_id=int(region_filter))
    
    # Filter by branch ID
    if branch_filter:
        events = events.filter(branch_id=int(branch_filter))
    
    # Filter by event type
    if type_filter:
        events = events.filter(event_type=type_filter)
    
    # Filter by date range
    if date_from:
        events = events.filter(created_at__date__gte=date_from)
    if date_to:
        events = events.filter(created_at__date__lte=date_to)
    
    return events.order_by('-created_at')[:100]
```

#### Search API Endpoints

| Endpoint | Parameters | Description |
|----------|------------|-------------|
| `/video/logs/` | `region`, `branch`, `type`, `from`, `to` | Filter events |
| `/api/events/` | Same as above | JSON event list |
| `/manage/branches/` | `search` | Search branches by name |

#### Adding Full-Text Search (Future Enhancement)

```python
# Using Django PostgreSQL full-text search
from django.contrib.postgres.search import SearchVector, SearchQuery

events = Event.objects.annotate(
    search=SearchVector('notes', 'camera__name', 'branch__name')
).filter(search=SearchQuery(search_term))
```

---

## 10. Troubleshooting

### 10.1 Common Issues

#### RTSP Stream Errors
```
[ WARN:0] global cap_ffmpeg_impl.hpp:453 Stream timeout triggered
[h264] error while decoding MB
```
**Solution:** The system uses TCP transport which is more reliable. If issues persist:
- Check camera network connectivity
- Verify RTSP URL is correct
- Increase timeout values in `_create_rtsp_capture()`

#### Video Clips Not Playing
**Cause:** Browser may not support codec
**Solution:** 
- Ensure FFmpeg is installed
- Check clip is H.264 encoded
- Verify `movflags +faststart` is applied

#### Detection Not Working
**Checklist:**
1. Check if detection is enabled for camera
2. Verify confidence thresholds aren't too high
3. Ensure cashier zone is properly configured
4. Check worker status in dashboard
5. Review terminal logs for errors

#### High CPU/GPU Usage
**Solutions:**
- Use smaller YOLO models (yolov8n instead of yolov8s)
- Reduce frame processing rate
- Lower video resolution
- Limit number of simultaneous cameras

### 10.2 Log Locations

| Log | Location | Description |
|-----|----------|-------------|
| Django | Console/stdout | Web requests, errors |
| Detection | Console/stdout | Model loading, detections |
| Worker | Console/stdout | Frame processing, events |

### 10.3 Debug Mode

Access developer mode in Camera Settings:
1. Click "Developer" button
2. Enter password: `dev123`
3. View real-time detection info
4. Toggle pose overlay
5. Adjust thresholds live

---

## Appendix A: Model Performance

| Model | Inference Time (GPU) | Inference Time (CPU) | Accuracy |
|-------|---------------------|---------------------|----------|
| YOLOv8n-Pose | ~15ms | ~100ms | Good |
| YOLOv8s-Pose | ~25ms | ~200ms | Better |
| Fire/Smoke YOLO | ~10ms | ~80ms | Trained |

## Appendix B: Supported Languages

| Code | Language | File |
|------|----------|------|
| `en` | English | Default |
| `ko` | Korean | translations.py |
| `th` | Thai | translations.py |
| `vi` | Vietnamese | translations.py |
| `zh` | Chinese | translations.py |

## Appendix C: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Dec 2025 | Initial release |

---

*Document generated for Hotel Cash Detector v1.0.0*
