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
│       ├── yolov8s.pt             # YOLOv8 Small (person detection)
│       ├── yolov8s-pose.pt        # YOLOv8 Pose (hand tracking)
│       └── fire_smoke_yolov8.pt   # Fire/smoke detection
│
└── detectors/                     # Detection modules (root level)
    ├── base_detector.py           # Base class
    ├── unified_detector.py        # Main detector
    ├── cash_detector.py           # Cash detection
    ├── violence_detector.py       # Violence detection
    └── fire_detector.py           # Fire detection
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

**Algorithm:** Pose-based hand proximity detection with strict cashier-customer validation

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
│     └── Confidence threshold: pose_confidence (default 0.5) │
│                                                            │
│  3. PERSON CENTER POINT CALCULATION                        │
│     └── Priority: Hip center (most stable)                 │
│         └── keypoints[11] (left_hip) + keypoints[12]       │
│     └── Fallback: Shoulder center                          │
│         └── keypoints[5] (left_shoulder) + keypoints[6]    │
│     └── Last resort: Bounding box center                   │
│         └── center = ((x1+x2)/2, (y1+y2)/2)                │
│                                                            │
│  4. ZONE CLASSIFICATION (STRICT)                           │
│     └── Use CENTER POINT only (not bbox overlap)           │
│     └── if center in cashier_zone → CASHIER                │
│     └── if center outside zone → CUSTOMER                  │
│     └── This ensures ONE person = ONE classification       │
│                                                            │
│  5. HAND POSITION EXTRACTION                               │
│     └── For each person:                                   │
│         ├── left_wrist = keypoints[9]                      │
│         ├── right_wrist = keypoints[10]                    │
│         └── Only use if confidence >= 0.3                  │
│                                                            │
│  6. HAND PROXIMITY CHECK (STRICT XOR VALIDATION)           │
│     └── For each person pair (i, j):                       │
│         ├── Skip if both IN zone (cashier-cashier)         │
│         ├── Skip if both OUT zone (customer-customer)      │
│         ├── Only accept if XOR: (p1_in XOR p2_in)          │
│         │   └── Exactly ONE in zone, ONE outside           │
│         │                                                  │
│         └── For valid cashier-customer pairs:              │
│             └── Check all hand combinations:               │
│                 ├── cashier.left ↔ customer.left           │
│                 ├── cashier.left ↔ customer.right          │
│                 ├── cashier.right ↔ customer.left          │
│                 └── cashier.right ↔ customer.right         │
│                 └── distance = √((x1-x2)² + (y1-y2)²)      │
│                                                            │
│  7. DETECTION CRITERIA (ALL MUST BE TRUE)                  │
│     ✓ ONE person center IN cashier zone (cashier)          │
│     ✓ ONE person center OUTSIDE zone (customer)            │
│     ✓ Both have visible hands (confidence >= 0.3)          │
│     ✓ Hand distance < hand_touch_distance                  │
│     ✓ Distance score = 1 - (distance/threshold)            │
│     ✓ consecutive_detections >= min_transaction_frames     │
│     ✓ Cooldown period elapsed (transaction_cooldown)       │
│                                                            │
│  8. METADATA COLLECTION                                    │
│     └── Cashier info:                                      │
│         ├── center: [x, y]                                 │
│         ├── bbox: [x1, y1, x2, y2]                         │
│         ├── hands: {left: [x,y,conf], right: [x,y,conf]}  │
│         ├── in_zone: true                                  │
│         └── hand_used: "left" or "right"                   │
│     └── Customer info:                                     │
│         ├── center: [x, y]                                 │
│         ├── bbox: [x1, y1, x2, y2]                         │
│         ├── hands: {left: [x,y,conf], right: [x,y,conf]}  │
│         ├── in_zone: false                                 │
│         └── hand_used: "left" or "right"                   │
│     └── Detection info:                                    │
│         ├── distance: actual pixel distance                │
│         ├── distance_threshold: threshold setting          │
│         ├── interaction_point: midpoint [x, y]             │
│         └── people_count: total in frame                   │
│                                                            │
│  9. EVENT GENERATION                                       │
│     └── Create Detection object with metadata              │
│     └── Trigger clip save (async, 30s buffer)              │
│     └── Save JSON file with full metadata                  │
│     └── Create database Event record                       │
│     └── Generate thumbnail from detection frame            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### Key Logic Improvements (December 2025)

**1. Center Point Classification (vs. Bbox Overlap)**
```python
# OLD: Bbox overlap could split one person into two zones
if self.is_box_in_cashier_zone(bbox, threshold=0.3):
    # Problem: 30% overlap = ambiguous

# NEW: Single center point = definitive classification
center = self.get_person_center(keypoints, bbox)  # Hip or shoulder center
if self.is_in_cashier_zone(center):
    # One person = one zone = no ambiguity
```

**2. Strict XOR Validation**
```python
# Enforce cashier-customer pairs ONLY
p1_in = person1['in_cashier_zone']
p2_in = person2['in_cashier_zone']
is_valid_pair = (p1_in and not p2_in) or (not p1_in and p2_in)

if not is_valid_pair:
    continue  # Skip: both cashiers OR both customers
```

**3. Comprehensive Metadata**
- Cashier position (center, bbox, hands)
- Customer position (center, bbox, hands)
- Actual measured distance
- Threshold that triggered detection
- Interaction point (hand midpoint)
- All stored in JSON for analysis

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

### 6.3 JSON Event Metadata Structure

Every detected event generates a JSON file with comprehensive metadata, stored in `media/json/`.

#### Cash Transaction Event JSON

```json
{
  "timestamp": "2025-12-07T14:30:45.123456",
  "frame_number": 12543,
  "confidence": 0.87,
  "bbox": [450, 380, 550, 480],
  "camera_id": "1",
  "camera_name": "Lobby Counter Camera",
  "event_type": "cash",
  "clip_path": "/media/clips/cash_1_20251207_143045.mp4",
  "thumbnail_path": "/media/thumbnails/cash_1_20251207_143045.jpg",
  
  "cash_detection": {
    "hand_touch_distance_threshold": 200,
    "cashier_zone": [2, 354, 1273, 359],
    "pose_confidence": 0.5
  },
  
  "cashier": {
    "center": [640, 540],
    "bbox": [520, 380, 760, 700],
    "hands": {
      "left": [580, 460, 0.92],
      "right": [695, 455, 0.88]
    },
    "in_zone": true,
    "hand_used": "right"
  },
  
  "customer": {
    "center": [920, 510],
    "bbox": [820, 350, 1020, 670],
    "hands": {
      "left": [865, 445, 0.85],
      "right": [975, 520, 0.79]
    },
    "in_zone": false,
    "hand_used": "left"
  },
  
  "measured_hand_distance": 85.5,
  "distance_threshold": 200,
  "interaction_point": [780, 450],
  "people_count": 2,
  
  "trigger_time": "2025-12-07T14:30:45.123456",
  "frames_saved": 450,
  "duration_sec": 30.0
}
```

#### Metadata Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO 8601 | Event detection timestamp |
| `frame_number` | Integer | Frame number in stream |
| `confidence` | Float | Detection confidence (0.0-1.0) |
| `bbox` | [x1, y1, x2, y2] | Bounding box of detection |
| `camera_id` | String | Camera identifier |
| `camera_name` | String | Camera display name |
| `event_type` | String | `cash`, `violence`, or `fire` |
| `clip_path` | String | Relative path to video clip |
| `thumbnail_path` | String | Relative path to thumbnail |

#### Cash Detection Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `cash_detection.hand_touch_distance_threshold` | Integer | Max pixels between hands for detection |
| `cash_detection.cashier_zone` | [x, y, w, h] | Cashier zone coordinates |
| `cash_detection.pose_confidence` | Float | Minimum keypoint confidence |
| `cashier.center` | [x, y] | Center point of cashier person |
| `cashier.bbox` | [x1, y1, x2, y2] | Cashier bounding box |
| `cashier.hands` | Object | Left/right hand positions [x, y, confidence] |
| `cashier.in_zone` | Boolean | Always `true` (in cashier zone) |
| `cashier.hand_used` | String | Which hand was used (`left` or `right`) |
| `customer.center` | [x, y] | Center point of customer person |
| `customer.bbox` | [x1, y1, x2, y2] | Customer bounding box |
| `customer.hands` | Object | Left/right hand positions [x, y, confidence] |
| `customer.in_zone` | Boolean | Always `false` (outside zone) |
| `customer.hand_used` | String | Which hand was used (`left` or `right`) |
| `measured_hand_distance` | Float | Actual pixel distance between hands |
| `distance_threshold` | Integer | Threshold that triggered detection |
| `interaction_point` | [x, y] | Midpoint between hands |
| `people_count` | Integer | Total people detected in frame |

#### Violence Detection JSON

```json
{
  "timestamp": "2025-12-07T15:22:10.987654",
  "event_type": "violence",
  "violence_detection": {
    "min_violence_frames": 10,
    "violence_confidence": 0.8,
    "motion_threshold": 100
  },
  "people_involved": 2,
  "close_combat_detected": true,
  "motion_magnitude": 235.6
}
```

#### Fire Detection JSON

```json
{
  "timestamp": "2025-12-07T16:45:33.555555",
  "event_type": "fire",
  "fire_detection": {
    "min_fire_frames": 3,
    "fire_confidence": 0.5,
    "detection_method": "yolo"  // or "color_based"
  },
  "fire_area": 4500,  // pixels
  "smoke_detected": false,
  "flickering_score": 0.75
}
```

### 6.4 Video Clip Saving

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

### 6.4 RTSP Connection & Stream Stability

```python
def _create_rtsp_capture(self, rtsp_url):
    """Create RTSP capture with optimized settings for stability.
    
    Uses TCP transport to avoid RTP packet ordering issues.
    Extended timeouts prevent premature disconnections.
    """
    import os
    
    # FFmpeg options for stable RTSP streaming:
    # - rtsp_transport=tcp: Use TCP instead of UDP (prevents packet loss)
    # - stimeout=60000000: 60 second socket timeout (microseconds)
    # - max_delay=1000000: Max delay 1 second before dropping frames
    # - fflags=nobuffer+discardcorrupt: Minimize latency, handle errors
    # - analyzeduration=2000000: Spend 2s analyzing stream
    # - probesize=2000000: Read 2MB to detect stream properties
    # - buffer_size=4096000: 4MB network buffer for stability
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
        'rtsp_transport;tcp|'
        'stimeout;60000000|'
        'max_delay;1000000|'
        'fflags;nobuffer+discardcorrupt|'
        'analyzeduration;2000000|'
        'probesize;2000000|'
        'buffer_size;4096000'
    )
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    
    # OpenCV timeout properties (balanced for reliability)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)  # 30s connection timeout
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)  # 15s read timeout
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # Buffer 5 frames for stability
    
    return cap
```

#### Connection Retry Logic

```python
def run(self):
    """Main worker loop with automatic reconnection"""
    
    # Try to connect with retries (5 attempts, 5s between)
    cap = None
    max_connect_retries = 5
    for attempt in range(max_connect_retries):
        print(f"[Worker] Connecting... (attempt {attempt + 1}/{max_connect_retries})")
        cap = self._create_rtsp_capture(camera.rtsp_url)
        
        if cap.isOpened():
            # Verify stream works by reading a test frame
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"[Worker] ✅ Connected successfully")
                break
            else:
                cap.release()
                cap = None
        time.sleep(5)  # Wait before retry
    
    if cap is None or not cap.isOpened():
        self.status = 'error'
        self.last_error = f'Cannot open stream after {max_connect_retries} attempts'
        return
    
    # Frame reading loop with automatic reconnection
    consecutive_failures = 0
    max_failures = 20  # Max consecutive failures before reconnect
    last_success_time = time.time()
    
    while self.running:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            consecutive_failures += 1
            time_since_success = time.time() - last_success_time
            
            # Reconnect if too many failures or 30s without frames
            if consecutive_failures >= max_failures or time_since_success > 30:
                self.status = 'reconnecting'
                self.last_error = f'Stream lost ({consecutive_failures} failures, {time_since_success:.0f}s)'
                print(f"[Worker] {self.last_error}")
                
                cap.release()
                time.sleep(3)  # Wait before reconnect
                
                cap = self._create_rtsp_capture(camera.rtsp_url)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.status = 'running'
                        self.last_error = None
                        consecutive_failures = 0
                        last_success_time = time.time()
                        print(f"[Worker] ✅ Reconnected")
            continue
        
        # Successful frame read
        consecutive_failures = 0
        last_success_time = time.time()
        # ... process frame ...
```

**Key Features:**
- **TCP Transport**: More reliable than UDP for RTSP
- **Extended Timeouts**: 60s socket timeout prevents premature disconnection
- **Automatic Reconnection**: Recovers from network issues
- **Test Frame Verification**: Ensures stream works before proceeding
- **Failure Tracking**: Distinguishes between temporary glitches and real failures

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
pip install -r requirements.txt
pip install gunicorn

# Download YOLO models
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); YOLO('yolov8s-pose.pt')"
```

**4. Configure Environment**
```bash
# Create .env file
cat > .env << EOF
SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
DEBUG=False
ALLOWED_HOSTS=your-domain.com,your-ec2-ip
YOLO_MODEL=yolov8s.pt
POSE_MODEL=yolov8s-pose.pt
FIRE_MODEL=fire_smoke_yolov8.pt
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

## 10. Testing & Verification

### 10.1 Test Script

The system includes a comprehensive test script: `test_worker_streaming.py`

```bash
python test_worker_streaming.py
```

#### Test Coverage

| Test | Description | Success Criteria |
|------|-------------|------------------|
| **Camera Model** | Database connection and camera config | Camera found with valid RTSP URL |
| **RTSP Connection** | Stream connectivity | Opens in <10s, valid resolution |
| **Frame Reading** | Stream stability | All frames read, FPS > 20 |
| **Detector Init** | Model loading | All detectors initialized |
| **Detection** | Frame processing | Frame processed without errors |
| **Cash Metadata** | Metadata structure | Expected keys present |
| **JSON Output** | File generation | Valid JSON structure |
| **Event Model** | Database operations | Events can be created/queried |

#### Sample Output

```
╔════════════════════════════════════════════════════════════╗
║     WORKER & STREAMING TEST SUITE                         ║
║     Hotel Cash Detector System                            ║
╚════════════════════════════════════════════════════════════╝

============================================================
  Test 1: Camera Model
============================================================

  ✅ PASS: Found 1 camera(s) in database
  ℹ️ INFO: Camera ID: 1
  ℹ️ INFO: Camera Name: test
  ℹ️ INFO: RTSP URL: rtsp://admin:adminadmin!@175.213.55.16:554
  ℹ️ INFO: Hand Touch Distance: 200px
  ℹ️ INFO: Cashier Zone: x=2, y=354, w=1273, h=359

============================================================
  Test 2: RTSP Stream Connection
============================================================

  ℹ️ INFO: Connecting to: rtsp://admin:adminadmin!@...
  ✅ PASS: Stream opened in 4.0s
  ℹ️ INFO: Stream FPS: 25.0
  ℹ️ INFO: Resolution: 1280x720

============================================================
  Test 3: Frame Reading
============================================================

  ℹ️ INFO: Reading 30 frames...
  ✅ PASS: Read all 30/30 frames successfully
  ℹ️ INFO: Actual FPS: 108.4
  ℹ️ INFO: Frame shape: (720, 1280, 3)

============================================================
  TEST SUMMARY
============================================================

  ✅ PASS: camera_model
  ✅ PASS: rtsp_connection
  ✅ PASS: frame_reading
  ✅ PASS: detector_init
  ✅ PASS: detection
  ✅ PASS: cash_metadata
  ✅ PASS: json_output
  ✅ PASS: event_model

  Results: 8/8 tests passed
  Time: 6.2s

  ✅ ALL TESTS PASSED!
```

### 10.2 Manual Testing Checklist

#### Camera Setup
- [ ] Camera appears in dashboard
- [ ] RTSP URL is correct
- [ ] Status shows "online"
- [ ] Live feed displays correctly

#### Detection Testing
- [ ] Cashier zone is properly configured
- [ ] Hand touch distance is appropriate (100-200px typical)
- [ ] Confidence thresholds are reasonable (0.5-0.8)
- [ ] Detection toggles work (enable/disable)

#### Cash Detection Validation
- [ ] Detects cashier-customer hand exchange
- [ ] Does NOT trigger on cashier-cashier contact
- [ ] Does NOT trigger on customer-customer contact
- [ ] Distance threshold works as expected
- [ ] Cooldown period prevents duplicates

#### Event Verification
- [ ] Events appear in logs
- [ ] Video clips are playable
- [ ] Thumbnails display correctly
- [ ] JSON files contain all metadata
- [ ] Metadata includes cashier/customer positions
- [ ] Measured distance is accurate

#### JSON Metadata Validation

```python
import json
from pathlib import Path

# Read latest JSON
json_dir = Path('media/json')
latest = max(json_dir.glob('*.json'), key=lambda p: p.stat().st_mtime)

with open(latest) as f:
    data = json.load(f)

# Verify required fields
assert 'cashier' in data, "Missing cashier position"
assert 'customer' in data, "Missing customer position"
assert 'measured_hand_distance' in data, "Missing distance"
assert 'interaction_point' in data, "Missing interaction point"

# Verify cashier data
assert data['cashier']['in_zone'] == True
assert 'center' in data['cashier']
assert 'hands' in data['cashier']

# Verify customer data
assert data['customer']['in_zone'] == False
assert 'center' in data['customer']
assert 'hands' in data['customer']

print("✅ JSON metadata validation passed!")
```

### 10.3 Performance Benchmarks

| Hardware | Cameras | FPS | CPU Usage | GPU Usage | Notes |
|----------|---------|-----|-----------|-----------|-------|
| **i7 + GTX 1660** | 1 | 30 | 25% | 40% | Smooth operation |
| **i7 + GTX 1660** | 4 | 30 | 60% | 75% | Acceptable |
| **i5 + No GPU** | 1 | 15 | 80% | - | Use yolov8n models |
| **AWS g4dn.xlarge** | 4 | 30 | 30% | 50% | Recommended |
| **AWS g4dn.2xlarge** | 10 | 30 | 40% | 60% | Production ready |

### 10.4 Debug Mode Testing

**Enable Debug Overlay:**
1. Go to Camera Settings
2. Click "Developer" button
3. Enter password: `dev123`
4. Enable "Show Pose Overlay"

**Verify Overlay Shows:**
- Person bounding boxes (green=cashier, orange=customer)
- Center point marker ("C")
- Hand circles (magenta)
- Distance lines between hands:
  - Green: Valid detection (cashier-customer, close)
  - Gray: Ignored (cashier-cashier or customer-customer)
  - Red: Too far apart
- Distance labels with pixel values

---

## 11. Troubleshooting

### 11.1 Common Issues

#### RTSP Stream Timeouts
```
[ WARN:0@30.044] global cap_ffmpeg_impl.hpp:453 Stream timeout triggered after 30043.255000 ms
[h264 @ 0000023ce8cd0940] error while decoding MB 36 28, bytestream -11
```

**Root Cause:** Default OpenCV FFmpeg timeout is 30 seconds, causing disconnects during brief network issues.

**Solution (Fixed in v1.0.1):**
```python
# Extended timeout configuration (60 seconds)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp|'
    'stimeout;60000000|'  # 60s timeout (microseconds)
    'max_delay;1000000|'  # 1s max frame delay
    'buffer_size;4096000' # 4MB buffer
)
```

**Additional Troubleshooting:**
- Check camera network connectivity: `ping camera-ip`
- Verify RTSP URL: `ffplay rtsp://...` (if ffmpeg installed)
- Check firewall rules for port 554
- Ensure camera supports TCP transport
- Monitor network bandwidth (1-5 Mbps per camera typical)

#### Frames Showing 0
```
📹 test
⏱️ Uptime: 00:00:53
📊 Frames: 0
🎯 Events: 0
```

**Root Cause:** Stream failed to connect or initial read failed.

**Solution (Fixed in v1.0.1):**
- Added connection retry logic (5 attempts)
- Test frame verification after connection
- Automatic reconnection on sustained failures
- Better failure tracking and reporting

**Debug Steps:**
```bash
# Check worker status
curl http://localhost:8000/api/workers/status/ | python -m json.tool

# Look for:
# - status: "running" vs "error" vs "reconnecting"
# - last_error: error message if any
# - frames_processed: should increase over time
```

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
- For low-resource systems, use yolov8n models (set in .env)
- Reduce frame processing rate (FRAME_SKIP in settings)
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
| YOLOv8s | ~20ms | ~150ms | Better (default) |
| YOLOv8s-Pose | ~25ms | ~200ms | Better (default) |
| YOLOv8n-Pose | ~15ms | ~100ms | Good (low resources) |
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
