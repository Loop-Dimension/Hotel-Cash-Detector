# Environment Variables - Hotel Cash Detector

Complete reference for all environment variables.

## Table of Contents

1. [Environment File Location](#environment-file-location)
2. [Complete .env Template](#complete-env-template)
3. [Variable Details](#variable-details)
4. [Production Configuration](#production-configuration)

---

## Environment File Location

**File**: `django_app/.env` or project root `.env`

**Important**:
- Never commit `.env` files to version control
- Add `.env` to `.gitignore`
- Use `.env.example` as a template

---

## Complete .env Template

```env
# =============================================================================
# Hotel Cash Detector - Environment Configuration
# =============================================================================

# Django Settings
SECRET_KEY=your-super-secret-key-change-in-production
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,192.168.1.100

# Gemini AI Configuration (Optional Validation)
GEMINI_API_KEY=your-gemini-api-key-here
GEMINI_VALIDATION_ENABLED=True

# Gemini Validation Mode
# False = Use single image frame (default, faster, less accurate)
# True = Use 3-second video clip (slower, more accurate with motion context)
GEMINI_USE_VIDEO_VALIDATION=False

# Detection Logging
ENABLE_DETECTION_LOGS=False

# GPU Settings
# Options: auto, cuda, cpu
USE_GPU=auto

# Detection Confidence Thresholds (0.0 - 1.0)
CASH_DETECTION_CONFIDENCE=0.5
VIOLENCE_DETECTION_CONFIDENCE=0.6
FIRE_DETECTION_CONFIDENCE=0.5

# Cash Detection Parameters
HAND_TOUCH_DISTANCE=100
MIN_TRANSACTION_FRAMES=1
TRANSACTION_COOLDOWN=45

# Violence Detection Parameters
MIN_VIOLENCE_FRAMES=15
MOTION_THRESHOLD=100
VIOLENCE_COOLDOWN=90

# Fire Detection Parameters
MIN_FIRE_FRAMES=10
MIN_FIRE_AREA=3000
FIRE_COOLDOWN=60

# Model Paths (relative to django_app/)
YOLO_MODEL=models/yolov8s.pt
POSE_MODEL=models/yolov8s-pose.pt
FIRE_MODEL=models/fire_smoke_yolov8.pt

# PMS Integration (Optional)
PMS_API_URL=http://localhost:8000
PMS_API_KEY=your-pms-api-key

# Database (Production only - SQLite used in dev)
# DATABASE_URL=postgresql://user:password@host:5432/cctv_db
```

---

## Variable Details

### Django Settings

#### SECRET_KEY

- **Value**: Strong random string (50+ characters)
- **Purpose**: Django security (sessions, CSRF, signing)
- **Required**: ✅ Yes
- **Default**: Demo key (MUST change for production)
- **Security**: ⚠️ **CRITICAL** - Rotating this invalidates all sessions

**Generate Secret**:
```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

#### DEBUG

- **Value**: `True` | `False`
- **Purpose**: Enable debug mode
- **Required**: ✅ Yes
- **Default**: `True`
- **Production**: Set to `False`

**Debug Mode Effects**:
- Shows detailed error pages
- Serves static files automatically
- Enables Django Debug Toolbar (if installed)

**Production**: MUST be `False` (security risk if enabled)

#### ALLOWED_HOSTS

- **Value**: Comma-separated hostnames/IPs
- **Purpose**: Valid hostnames for Django
- **Required**: ✅ Yes (production)
- **Default**: `localhost,127.0.0.1`
- **Example**: `localhost,cctv.hio.ai.kr,192.168.1.100`

---

### Gemini AI Configuration

#### GEMINI_API_KEY

- **Value**: Google Gemini API key
- **Purpose**: AI validation of detections (optional)
- **Required**: ❌ No (feature is optional)
- **Default**: None
- **Get Key**: https://aistudio.google.com/app/apikey

**What it does**:
- Sends detection frames to Gemini AI for validation
- Reduces false positives
- Provides natural language description
- Can validate cash/violence/fire events

#### GEMINI_VALIDATION_ENABLED

- **Value**: `True` | `False`
- **Purpose**: Enable/disable Gemini validation
- **Required**: ❌ No
- **Default**: `False`
- **Note**: Requires `GEMINI_API_KEY` to function

#### GEMINI_USE_VIDEO_VALIDATION

- **Value**: `True` | `False`
- **Purpose**: Use video clip vs. single frame for validation
- **Required**: ❌ No
- **Default**: `False`

**Validation Modes**:

| Mode | Speed | Accuracy | Best For |
|------|-------|----------|----------|
| Image (`False`) | Fast (1-2s) | Good | Static scenes |
| Video (`True`) | Slow (3-5s) | Better | Motion analysis |

**Image Mode**: Sends single frame to Gemini
**Video Mode**: Creates 3-second clip, sends to Gemini

---

### Detection Logging

#### ENABLE_DETECTION_LOGS

- **Value**: `True` | `False`
- **Purpose**: Enable verbose detection logging
- **Required**: ❌ No
- **Default**: `False`
- **Production**: `False` (performance impact)

**When enabled**:
- Logs every detection to console
- Logs frame processing time
- Logs confidence scores
- Useful for debugging

---

### GPU Settings

#### USE_GPU

- **Value**: `auto` | `cuda` | `cpu`
- **Purpose**: Control GPU usage for AI inference
- **Required**: ✅ Yes
- **Default**: `auto`

**Options**:
- `auto`: Use GPU if available, fallback to CPU
- `cuda`: Force GPU (fails if no CUDA)
- `cpu`: Force CPU only

**GPU Requirements**:
- CUDA-capable GPU (NVIDIA)
- CUDA Toolkit installed
- PyTorch with CUDA support

**Check CUDA availability**:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # GPU name
```

---

### Detection Confidence Thresholds

#### CASH_DETECTION_CONFIDENCE

- **Value**: Float (0.0 - 1.0)
- **Purpose**: Minimum confidence for cash detection
- **Required**: ✅ Yes
- **Default**: `0.5`
- **Recommended**: `0.4 - 0.6`

**Tuning**:
- **Lower (0.3-0.4)**: More detections, more false positives
- **Higher (0.6-0.8)**: Fewer detections, miss some events

#### VIOLENCE_DETECTION_CONFIDENCE

- **Value**: Float (0.0 - 1.0)
- **Purpose**: Minimum confidence for violence detection
- **Required**: ✅ Yes
- **Default**: `0.6`
- **Recommended**: `0.6 - 0.8`

**Note**: Violence detection uses multiple factors (pose, motion, proximity), so higher threshold recommended.

#### FIRE_DETECTION_CONFIDENCE

- **Value**: Float (0.0 - 1.0)
- **Purpose**: Minimum confidence for fire detection
- **Required**: ✅ Yes
- **Default**: `0.5`
- **Recommended**: `0.5 - 0.7`

---

### Cash Detection Parameters

#### HAND_TOUCH_DISTANCE

- **Value**: Integer (pixels)
- **Purpose**: Maximum pixel distance between hands for cash detection
- **Required**: ✅ Yes
- **Default**: `100`
- **Recommended**: `80 - 200`

**Tuning Guide**:
- **1080p camera**: 100-150px
- **720p camera**: 60-100px
- **4K camera**: 150-250px

**Formula**: `distance = sqrt((x1-x2)² + (y1-y2)²)`

#### MIN_TRANSACTION_FRAMES

- **Value**: Integer (frames)
- **Purpose**: Consecutive frames before confirming cash detection
- **Required**: ✅ Yes
- **Default**: `1`
- **Recommended**: `1 - 3`

**Purpose**: Prevent single-frame false positives

#### TRANSACTION_COOLDOWN

- **Value**: Integer (frames)
- **Purpose**: Frames to wait before next cash detection
- **Required**: ✅ Yes
- **Default**: `45`
- **Recommended**: `30 - 60` (1-2 seconds at 30fps)

**Purpose**: Prevent duplicate events for same transaction

---

### Violence Detection Parameters

#### MIN_VIOLENCE_FRAMES

- **Value**: Integer (frames)
- **Purpose**: Consecutive frames before confirming violence
- **Required**: ✅ Yes
- **Default**: `15`
- **Recommended**: `10 - 20` (0.3-0.7 seconds at 30fps)

**Purpose**: Ensure sustained aggressive behavior, not momentary motion

#### MOTION_THRESHOLD

- **Value**: Integer (magnitude)
- **Purpose**: Motion magnitude threshold for violence
- **Required**: ✅ Yes
- **Default**: `100`
- **Recommended**: `80 - 150`

#### VIOLENCE_COOLDOWN

- **Value**: Integer (frames)
- **Purpose**: Frames between violence alerts
- **Required**: ✅ Yes
- **Default**: `90`
- **Recommended**: `60 - 120` (2-4 seconds at 30fps)

---

### Fire Detection Parameters

#### MIN_FIRE_FRAMES

- **Value**: Integer (frames)
- **Purpose**: Consecutive frames before confirming fire
- **Required**: ✅ Yes
- **Default**: `10`
- **Recommended**: `5 - 15`

**Purpose**: Reduce false positives from reflections/lights

#### MIN_FIRE_AREA

- **Value**: Integer (pixels²)
- **Purpose**: Minimum fire region area
- **Required**: ✅ Yes
- **Default**: `3000`
- **Recommended**: `2000 - 5000`

**Purpose**: Ignore small bright spots (lights, reflections)

#### FIRE_COOLDOWN

- **Value**: Integer (frames)
- **Purpose**: Frames between fire alerts
- **Required**: ✅ Yes
- **Default**: `60`
- **Recommended**: `45 - 90` (1.5-3 seconds at 30fps)

---

### Model Paths

#### YOLO_MODEL

- **Value**: Path to YOLOv8 object detection model
- **Purpose**: Person detection for all scenarios
- **Required**: ✅ Yes
- **Default**: `models/yolov8s.pt`
- **Options**:
  - `yolov8n.pt` - Nano (fastest, less accurate)
  - `yolov8s.pt` - Small (recommended)
  - `yolov8m.pt` - Medium (slower, more accurate)

#### POSE_MODEL

- **Value**: Path to YOLOv8-Pose model
- **Purpose**: Keypoint detection for cash transactions
- **Required**: ✅ Yes
- **Default**: `models/yolov8s-pose.pt`
- **Options**:
  - `yolov8n-pose.pt` - Nano (fastest)
  - `yolov8s-pose.pt` - Small (recommended)

#### FIRE_MODEL

- **Value**: Path to fire detection YOLO model
- **Purpose**: Fire and smoke detection
- **Required**: ✅ Yes
- **Default**: `models/fire_smoke_yolov8.pt`
- **Note**: Custom-trained model for fire/smoke

---

### PMS Integration

#### PMS_API_URL

- **Value**: HotelPMS API base URL
- **Purpose**: Project synchronization from PMS
- **Required**: ❌ No (only if using PMS integration)
- **Example**:
  - Local: `http://localhost:8000`
  - Production: `https://pmsapi.hio.ai.kr`

#### PMS_API_KEY

- **Value**: API key for PMS authentication
- **Purpose**: Server-to-server authentication
- **Required**: ❌ No (only if using PMS integration)
- **Security**: ⚠️ Keep secret, rotate quarterly

---

## Production Configuration

### Production .env

```env
# Django (Production)
SECRET_KEY=<GENERATE-WITH-secrets.token_hex-32>
DEBUG=False
ALLOWED_HOSTS=cctv.hio.ai.kr,your-ip-address

# Gemini AI (Optional)
GEMINI_API_KEY=<YOUR-GEMINI-KEY>
GEMINI_VALIDATION_ENABLED=True
GEMINI_USE_VIDEO_VALIDATION=False

# Detection Logging (Disable in production)
ENABLE_DETECTION_LOGS=False

# GPU (Force CUDA in production)
USE_GPU=cuda

# Detection Thresholds (Tuned for your cameras)
CASH_DETECTION_CONFIDENCE=0.5
VIOLENCE_DETECTION_CONFIDENCE=0.7
FIRE_DETECTION_CONFIDENCE=0.6

# Cash Detection
HAND_TOUCH_DISTANCE=120
MIN_TRANSACTION_FRAMES=2
TRANSACTION_COOLDOWN=45

# Models (Use 's' models for production)
YOLO_MODEL=models/yolov8s.pt
POSE_MODEL=models/yolov8s-pose.pt
FIRE_MODEL=models/fire_smoke_yolov8.pt

# PMS Integration (Production)
PMS_API_URL=https://pmsapi.hio.ai.kr
PMS_API_KEY=<PRODUCTION-API-KEY>

# Database (PostgreSQL in production)
DATABASE_URL=postgresql://cctv_user:STRONG_PASSWORD@production-db:5432/cctv_production
```

---

## Related Documentation

- [02 - Setup](02-setup.md) - Development setup
- [04 - Features](04-features.md) - Detection features
- [09 - Troubleshooting](09-troubleshooting.md) - Common issues

---

**Previous**: [← 02 - Setup](02-setup.md) | **Next**: [04 - Features →](04-features.md)
