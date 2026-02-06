# System Overview - Hotel Cash Detector

Comprehensive overview of the AI-powered CCTV monitoring system.

## Table of Contents

1. [What is Hotel Cash Detector?](#what-is-hotel-cash-detector)
2. [Who Uses This System?](#who-uses-this-system)
3. [Key Features](#key-features)
4. [Business Value](#business-value)
5. [Use Cases](#use-cases)
6. [System Requirements](#system-requirements)

---

## What is Hotel Cash Detector?

The **Hotel Cash Detector** is an AI-powered CCTV monitoring system designed for hotel cashier surveillance and safety monitoring. It uses advanced computer vision and deep learning to detect:

### Primary Detection Capabilities

1. **Cash Transaction Detection**
   - Hand-to-hand exchanges between cashier and customer
   - Pose-based hand proximity detection (YOLOv8-Pose)
   - Strict cashier zone validation
   - Prevents false positives (cashier-cashier, customer-customer)

2. **Violence/Disturbance Detection**
   - Physical altercations or aggressive behavior
   - Close combat detection
   - Rapid motion analysis
   - Multi-person interaction tracking

3. **Fire/Smoke Detection**
   - Real-time fire detection (YOLO + color-based)
   - Smoke detection with motion analysis
   - Flickering flame analysis
   - Background subtraction for smoke

### System Architecture

- **Microservice architecture**: Django backend (port 8000) + FastAPI ML service (port 8001)
- **Real-time RTSP video stream processing** via TCP
- **Background detection workers** (one per camera)
- **Auto-fallback detection**: ML service with automatic fallback to local detectors
- **Multi-camera support** (8+ simultaneous streams)
- **Event logging with video clip recording** (30-second clips)
- **Multi-language support** (English, Korean, Thai, Vietnamese, Chinese)
- **Role-based access control** (Admin, Project Manager)
- **Developer mode** for debugging and threshold tuning

---

## Who Uses This System?

### Primary Users

#### 1. Hotel Managers
- **Use**: Monitor cashier transactions and safety
- **Goal**: Prevent theft, ensure safety compliance
- **Actions**:
  - Review event logs (cash, violence, fire)
  - Watch video clips of detected events
  - Manage camera settings per location
  - Generate reports by branch/region

#### 2. Security Staff
- **Use**: Real-time monitoring dashboard
- **Role**: Respond to safety alerts
- **Actions**:
  - Monitor live camera feeds
  - Respond to violence/fire alerts
  - Review flagged events
  - Mark events as confirmed/false positive

#### 3. System Administrators
- **Use**: Multi-region management
- **Role**: System oversight and configuration
- **Actions**:
  - Manage regions and branches
  - Configure camera RTSP URLs
  - Set detection thresholds
  - Assign project managers to branches
  - Monitor system health

#### 4. Project Managers
- **Use**: Branch-level management
- **Role**: Single property oversight
- **Actions**:
  - View assigned branch cameras
  - Review events for their branch
  - Adjust camera-specific settings
  - Generate branch reports

---

## Key Features

### 1. AI Detection Models

**YOLOv8 Object Detection**:
- Person detection for all scenarios
- Fire and smoke object detection
- Real-time inference (20-25ms on GPU)

**YOLOv8-Pose Estimation**:
- 17 keypoint detection per person
- Hand position tracking (wrists)
- Center point calculation (hip/shoulder)
- Pose confidence scoring

**Cash Transaction Algorithm**:
```
1. Detect all people via YOLOv8-Pose
2. Calculate person center point (hip or shoulder)
3. Classify: IN cashier zone or OUT of zone
4. Extract hand positions (left/right wrist)
5. Check valid pair: ONE in zone, ONE out (XOR)
6. Measure hand distance (pixels)
7. Trigger if distance < threshold
8. Generate event with full metadata
```

### 2. Multi-Camera Support

- **Simultaneous processing**: 8+ cameras (GPU-dependent)
- **Independent settings**: Per-camera thresholds
- **Cashier zone configuration**: Custom zones per camera
- **Detection toggles**: Enable/disable cash, violence, fire per camera

### 3. Real-Time Video Processing

**RTSP Streaming**:
- TCP transport for reliability
- Extended timeouts (60s connection)
- Automatic reconnection on failure
- Frame buffering (30 seconds)

**Video Clip Saving**:
- 30-second clips centered on detection
- H.264 encoding (web-compatible MP4)
- Thumbnail generation
- FFmpeg transcoding

### 4. Event Logging

**Comprehensive Metadata**:
- Cashier position (center, bbox, hands)
- Customer position (center, bbox, hands)
- Measured hand distance
- Interaction point
- Full JSON export

**Event Management**:
- Status workflow (pending → reviewing → confirmed)
- Reviewer assignment
- Notes and annotations
- Search and filtering (region, branch, type, date)

### 5. Developer Mode

**Debug Features**:
- Real-time pose overlay
- Distance visualization
- Person classification markers
- Hand circle indicators
- Detection threshold tuning

**Access**:
- Password-protected: `dev123`
- Per-camera activation
- Live debug info display

---

## Business Value

### For Hotel Owners

**Loss Prevention**:
- Detect unauthorized cash handling
- Monitor cashier-customer exchanges
- Reduce internal theft
- Video evidence for investigations

**Safety Compliance**:
- Early fire detection
- Violence alert system
- 24/7 automated monitoring
- Regulatory compliance (safety systems)

**Cost Reduction**:
- Automated monitoring (vs. manual review)
- Reduce security staffing needs
- Lower insurance premiums (with safety systems)

### For Security Teams

**Efficiency**:
- Automated alert system
- Focus on real incidents (not false alarms)
- Multi-location monitoring from central dashboard
- Quick incident review (video clips)

**Response Time**:
- Real-time alerts (cash, violence, fire)
- Immediate video access
- Faster incident resolution

### For Management

**Analytics**:
- Transaction frequency tracking
- Incident reports by branch/region
- Performance metrics
- Trend analysis

**Accountability**:
- Video evidence
- Audit trails
- Reviewer assignment tracking

---

## Use Cases

### Use Case 1: Cash Transaction Monitoring

**Scenario**: Hotel lobby cashier handling payments.

**Setup**:
1. Camera positioned above cashier counter
2. Cashier zone configured around counter area
3. Hand touch distance: 100-200px
4. Detection enabled

**Operation**:
1. Guest approaches to pay
2. System detects two people (cashier + guest)
3. Classifies: cashier (center IN zone), guest (center OUT zone)
4. Tracks hands via pose estimation
5. Measures hand proximity
6. When hands touch (< 100px), triggers detection
7. Saves 30-second clip with metadata
8. Logs event to database

**Outcome**: Manager reviews all cash exchanges, identifies patterns, prevents theft.

---

### Use Case 2: Violence Detection

**Scenario**: Disturbance in hotel lobby.

**Flow**:
1. Two guests arguing
2. Argument escalates to physical contact
3. System detects close combat (overlapping bboxes)
4. Analyzes aggressive poses (raised arms, rapid motion)
5. Sustained detection over 10 frames
6. Triggers violence alert
7. Security receives real-time notification
8. Reviews live feed and responds

**Outcome**: Faster response, video evidence, incident documentation.

---

### Use Case 3: Fire Detection

**Scenario**: Small fire in back office.

**Flow**:
1. Fire starts from electrical fault
2. YOLO model detects fire object (confidence 0.8)
3. Color-based detection confirms (orange/yellow flickering)
4. Sustained detection over 5 frames
5. Triggers fire alert
6. Staff receives immediate notification
7. Checks camera, confirms real fire
8. Evacuates and calls fire department

**Outcome**: Early detection, faster evacuation, potential lives saved.

---

### Use Case 4: Multi-Branch Management

**Scenario**: Hotel chain with 20 branches across 5 regions.

**Setup**:
1. Admin creates 5 regions (Bangkok, Phuket, etc.)
2. Creates 20 branches under regions
3. Registers 40 cameras (2 per branch)
4. Assigns project managers per region

**Operations**:
- **Admin**: View all events across all branches
- **Project Manager**: View only assigned region
- **Filter events**: By region, branch, type, date
- **Generate reports**: Monthly cash transaction count

**Outcome**: Centralized monitoring with delegated management.

---

## System Requirements

### Server Requirements

**Minimum (Development)**:
- CPU: Intel i5 or equivalent
- RAM: 8 GB
- Storage: 50 GB
- GPU: CUDA-capable (GTX 1650+)
- OS: Ubuntu 20.04+ or Windows 10+

**Recommended (Production 4-8 cameras)**:
- CPU: Intel i7 or AWS g4dn.xlarge
- RAM: 16 GB
- Storage: 100+ GB SSD
- GPU: GTX 1660 or Tesla T4
- OS: Ubuntu 20.04 LTS

**High-Performance (8-16 cameras)**:
- CPU: Intel i9 or AWS g4dn.2xlarge
- RAM: 32 GB
- Storage: 250+ GB SSD
- GPU: RTX 3060 or Tesla T4
- OS: Ubuntu 20.04 LTS

### Camera Requirements

**RTSP Cameras**:
- Protocol: RTSP over TCP
- Resolution: 720p minimum, 1080p recommended
- FPS: 15+ (25-30 ideal)
- Codec: H.264
- Network: Stable connection (1-5 Mbps per camera)

**Camera Positioning**:
- **Cash detection**: Overhead view of cashier counter
- **Violence detection**: Wide angle of common areas
- **Fire detection**: Coverage of high-risk areas

### Network Requirements

**Bandwidth**:
- 1-5 Mbps per camera (depends on resolution/codec)
- 10+ Mbps for dashboard access
- Low latency (<50ms to cameras)

**Ports**:
- 8000 (Django web server)
- 8001 (FastAPI ML service)
- 554 (RTSP from cameras)
- 80/443 (Nginx reverse proxy in production)

---

## Integration Overview

The CCTV system integrates with:

1. **ML Service** - FastAPI microservice for AI detection (port 8001)
2. **HotelPMS** - Central authentication and project synchronization
3. **RTSP Cameras** - Real-time video streams
4. **GPU** - CUDA acceleration for AI inference (used by ML service)

For detailed integration documentation, see:
- [06 - Integrations](06-integrations.md)

---

## Related Documentation

- [01 - Architecture](01-architecture.md) - AI models and technical design
- [02 - Setup](02-setup.md) - Development setup
- [04 - Features](04-features.md) - Detection algorithms

---

**Next**: [01 - Architecture →](01-architecture.md)
