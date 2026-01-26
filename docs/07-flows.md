# Flows - Hotel Cash Detector

End-to-end workflows and sequence diagrams for detection processes.

## Table of Contents

1. [Cash Transaction Detection Flow](#cash-transaction-detection-flow)
2. [Violence Detection Flow](#violence-detection-flow)
3. [Fire Detection Flow](#fire-detection-flow)
4. [Event Saving Flow](#event-saving-flow)
5. [Video Clip Generation](#video-clip-generation)
6. [User Interaction Flows](#user-interaction-flows)

---

## Cash Transaction Detection Flow

### Complete Detection Sequence

```mermaid
sequenceDiagram
    participant Worker as Background Worker
    participant RTSP as RTSP Camera
    participant Detector as UnifiedDetector
    participant Pose as YOLOv8-Pose Model
    participant Cash as CashDetector
    participant DB as Database
    participant FS as File System

    Worker->>RTSP: Read frame
    RTSP-->>Worker: Frame (1920x1080)

    Worker->>Worker: Add to buffer<br/>(last 900 frames)

    Worker->>Detector: process_frame(frame)

    Detector->>Pose: Run inference
    Pose-->>Detector: Keypoints (17 per person)

    Detector->>Cash: detect(frame, pose_results)

    Cash->>Cash: Extract people with keypoints
    Cash->>Cash: Calculate center points<br/>(hip or shoulder)

    loop For each person
        Cash->>Cash: Classify zone:<br/>IN (cashier) or OUT (customer)
    end

    Cash->>Cash: Extract hand positions<br/>(wrists)

    loop For each person pair
        Cash->>Cash: Check XOR:<br/>(p1_in XOR p2_in)?

        alt Valid pair (cashier-customer)
            Cash->>Cash: Measure hand distance:<br/>√((x1-x2)² + (y1-y2)²)

            alt Distance < threshold
                Cash->>Cash: Increment consecutive_frames

                alt consecutive_frames >= min_frames
                    Cash-->>Detector: Detection {<br/>  cashier: {...},<br/>  customer: {...},<br/>  distance: 87.3<br/>}
                end
            else Distance >= threshold
                Cash->>Cash: Reset consecutive_frames
            end
        else Invalid pair (skip)
            Cash->>Cash: Skip (both IN or both OUT)
        end
    end

    alt Detection triggered
        Detector-->>Worker: Event detected

        Worker->>DB: Create Event record
        DB-->>Worker: event_id

        Worker->>FS: Save 30s video clip<br/>(async)
        Worker->>FS: Save thumbnail
        Worker->>FS: Save JSON metadata

        FS-->>Worker: File paths

        Worker->>DB: Update Event<br/>(clip_path, thumbnail_path)
    end

    Worker->>Worker: Update frame counter
    Worker->>Worker: Sleep (throttle to 30fps)
```

### Step-by-Step Breakdown

#### 1. Frame Acquisition
- Worker reads frame from RTSP camera (TCP transport)
- Frame buffered for 30 seconds (900 frames at 30fps)

#### 2. Pose Estimation
- YOLOv8-Pose detects all people in frame
- Extracts 17 keypoints per person
- Returns keypoints with confidence scores

#### 3. Zone Classification
- Calculate person center (hip or shoulder average)
- Check if center point is inside cashier zone
- Classify: IN zone = cashier, OUT zone = customer

#### 4. Hand Position Extraction
- Extract wrist positions (keypoints 9, 10)
- Filter by confidence (>= 0.3)
- Store left and right hand coordinates

#### 5. Pair Validation (XOR)
- Check all person pairs
- **Skip** if both IN zone (cashier-cashier)
- **Skip** if both OUT zone (customer-customer)
- **Valid** if XOR: exactly ONE in, ONE out

#### 6. Distance Measurement
- Calculate Euclidean distance between hands
- Check all hand combinations (left-left, left-right, right-left, right-right)
- Compare to `hand_touch_distance` threshold

#### 7. Consecutive Frame Tracking
- Increment counter if detection valid
- Reset counter if detection lost
- Trigger event if counter >= `min_transaction_frames`

#### 8. Event Creation
- Save to database with full metadata
- Trigger async clip saving
- Generate thumbnail and JSON

---

## Violence Detection Flow

### Detection Sequence

```mermaid
sequenceDiagram
    participant Worker as Background Worker
    participant Detector as UnifiedDetector
    participant Pose as YOLOv8-Pose Model
    participant Violence as ViolenceDetector
    participant DB as Database

    Worker->>Detector: process_frame(frame)
    Detector->>Pose: Run inference
    Pose-->>Detector: Keypoints + Boxes

    Detector->>Violence: detect(frame, pose_results)

    Violence->>Violence: Extract all people

    alt Less than 2 people
        Violence-->>Detector: No detection<br/>(violence requires 2+ people)
    else 2 or more people
        loop For each person pair
            Violence->>Violence: Check proximity:<br/>Bboxes overlap?

            alt Boxes overlap (close combat)
                Violence->>Violence: Analyze poses:<br/>Raised arms?
                Violence->>Violence: Calculate motion magnitude

                alt Aggressive pose AND high motion
                    Violence->>Violence: Increment violence_frames

                    alt violence_frames >= min_violence_frames
                        Violence-->>Detector: Violence detected {<br/>  people: 2,<br/>  motion: 235.6,<br/>  confidence: 0.85<br/>}

                        Detector-->>Worker: Event detected
                        Worker->>DB: Create Event<br/>(type='violence')
                        Worker->>Worker: Trigger clip save
                    end
                else Not aggressive
                    Violence->>Violence: Reset violence_frames
                end
            else Boxes not overlapping
                Violence->>Violence: Not close combat
            end
        end
    end
```

### Key Checks

1. **Minimum 2 People**: Violence requires at least 2 people in frame
2. **Close Proximity**: Bounding boxes must overlap
3. **Aggressive Poses**: Raised arms, rapid motion
4. **High Motion**: Motion magnitude > threshold
5. **Sustained**: 15+ consecutive frames (0.5s at 30fps)

---

## Fire Detection Flow

### Detection Sequence

```mermaid
sequenceDiagram
    participant Worker as Background Worker
    participant Detector as UnifiedDetector
    participant FireYOLO as Fire YOLO Model
    participant Fire as FireDetector
    participant DB as Database

    Worker->>Detector: process_frame(frame)

    Detector->>FireYOLO: Run inference
    FireYOLO-->>Detector: Detections {<br/>  class: 'fire',<br/>  conf: 0.82<br/>}

    Detector->>Fire: detect(frame, yolo_results)

    alt YOLO detected fire
        Fire->>Fire: Check confidence >= threshold

        alt Confidence sufficient
            Fire->>Fire: Increment fire_frames

            alt fire_frames >= min_fire_frames
                Fire-->>Detector: Fire detected {<br/>  method: 'yolo',<br/>  confidence: 0.82<br/>}

                Detector-->>Worker: Event detected
                Worker->>DB: Create Event<br/>(type='fire')
                Worker->>Worker: Trigger clip save
            end
        else Low confidence
            Fire->>Fire: Try fallback method
        end
    else YOLO no detection
        Fire->>Fire: FALLBACK: Color-based detection

        Fire->>Fire: Convert to HSV
        Fire->>Fire: Apply fire color mask<br/>(orange/yellow, 5-25° hue)
        Fire->>Fire: Exclude skin tones
        Fire->>Fire: Calculate fire area

        alt Fire area >= min_fire_area
            Fire->>Fire: Analyze flickering
            Fire->>Fire: Increment fire_frames

            alt fire_frames >= min_fire_frames
                Fire-->>Detector: Fire detected {<br/>  method: 'color_based',<br/>  area: 4500<br/>}

                Detector-->>Worker: Event detected
                Worker->>DB: Create Event
            end
        else Area too small
            Fire->>Fire: Reset fire_frames
        end
    end
```

### Detection Methods

#### Primary: YOLO Model
- Custom-trained YOLOv8 model
- Classes: Fire (0), Smoke (2)
- Fast and accurate

#### Fallback: Color-Based
- HSV color space analysis
- Fire colors: H=5-25°, S=150-255, V=200-255
- Skin tone exclusion to prevent false positives
- Flickering analysis for motion validation

---

## Event Saving Flow

### Complete Event Lifecycle

```mermaid
sequenceDiagram
    participant Detector as Detection Triggered
    participant Worker as Background Worker
    participant DB as Django ORM
    participant FS as File System
    participant FFmpeg as FFmpeg Process

    Detector->>Worker: Detection event {<br/>  type: 'cash',<br/>  metadata: {...}<br/>}

    Worker->>Worker: Check cooldown period

    alt Cooldown expired
        Worker->>DB: Event.objects.create({<br/>  camera_id: 1,<br/>  event_type: 'cash',<br/>  confidence: 0.87,<br/>  status: 'pending'<br/>})

        DB-->>Worker: event (id=123)

        par Async clip saving
            Worker->>Worker: Extract 30s from buffer<br/>(frames[current-450:current+450])

            Worker->>FS: Write temp MJPG file<br/>(temp_123.avi)

            Worker->>FFmpeg: Convert to H.264<br/>ffmpeg -i temp_123.avi \<br/>  -c:v libx264 \<br/>  -movflags +faststart \<br/>  event_123_clip.mp4

            FFmpeg-->>FS: event_123_clip.mp4

            Worker->>FS: Save thumbnail<br/>(last frame as JPG)

            Worker->>FS: Save JSON metadata {<br/>  cashier: {...},<br/>  customer: {...},<br/>  distance: 87.3<br/>}

            FS-->>Worker: File paths

            Worker->>DB: Event.update({<br/>  clip_path: 'clips/event_123_clip.mp4',<br/>  thumbnail_path: 'thumbnails/event_123.jpg'<br/>})
        end

        Worker->>Worker: Set cooldown timer<br/>(transaction_cooldown frames)

    else Cooldown active
        Worker->>Worker: Skip event<br/>(prevent duplicates)
    end
```

### Cooldown Mechanism

**Purpose**: Prevent duplicate events for same transaction/incident.

**Configuration**:
- `TRANSACTION_COOLDOWN=45` frames (1.5s at 30fps)
- `VIOLENCE_COOLDOWN=90` frames (3s at 30fps)
- `FIRE_COOLDOWN=60` frames (2s at 30fps)

**Logic**:
```python
# Check cooldown
frames_since_last_event = current_frame - last_event_frame

if frames_since_last_event >= cooldown_frames:
    # Cooldown expired, allow new event
    save_event()
    last_event_frame = current_frame
else:
    # Still in cooldown, skip
    pass
```

---

## Video Clip Generation

### Frame Buffer to MP4

```mermaid
flowchart LR
    Buffer[Frame Buffer<br/>900 frames deque]
    --> Extract[Extract 30s<br/>450 before + 450 after]
    --> TempAVI[Write Temp AVI<br/>MJPG codec]
    --> FFmpeg[FFmpeg Transcode<br/>H.264 + faststart]
    --> MP4[Final MP4<br/>Web-compatible]

    Extract --> Thumbnail[Extract Last Frame<br/>Save as JPG]

    style Buffer fill:#e3f2fd
    style MP4 fill:#c8e6c9
    style Thumbnail fill:#fff9c4
```

### Implementation

```python
def save_clip(frames, camera, event_type, event_id):
    """Save 30-second video clip from frame buffer."""

    # Step 1: Create temporary AVI with MJPG
    temp_path = f'temp_{event_id}.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    height, width = frames[0].shape[:2]
    fps = 30

    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    # Step 2: Transcode to H.264 MP4 with FFmpeg
    final_path = f'media/clips/{event_type}_{camera.id}_{timestamp}.mp4'

    subprocess.run([
        'ffmpeg', '-y', '-i', temp_path,
        '-c:v', 'libx264',      # H.264 codec
        '-preset', 'fast',      # Encoding speed
        '-crf', '23',           # Quality (18-28 typical)
        '-pix_fmt', 'yuv420p',  # Pixel format (universal support)
        '-movflags', '+faststart',  # Optimize for web streaming
        '-r', str(fps),         # Frame rate
        final_path
    ], check=True)

    # Step 3: Save thumbnail (last frame)
    thumb_path = f'media/thumbnails/{event_type}_{camera.id}_{timestamp}.jpg'
    cv2.imwrite(thumb_path, frames[-1])

    # Step 4: Save JSON metadata
    json_path = f'media/json/{event_type}_{camera.id}_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Clean up temp file
    os.remove(temp_path)

    return final_path, thumb_path, json_path
```

### FFmpeg Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `-c:v libx264` | H.264 codec | Universal browser support |
| `-preset fast` | Fast encoding | Balance speed/compression |
| `-crf 23` | Constant Rate Factor | Quality setting (lower = better) |
| `-pix_fmt yuv420p` | Pixel format | Maximum compatibility |
| `-movflags +faststart` | Move metadata | Enable progressive streaming |
| `-r 30` | Frame rate | Match source FPS |

---

## User Interaction Flows

### 1. View Live Camera Feed

```mermaid
sequenceDiagram
    participant User as User Browser
    participant Django as Django View
    participant Worker as Background Worker

    User->>Django: GET /monitor/CAM001/
    Django->>Django: Check camera exists
    Django->>Worker: Check worker status

    alt Worker not running
        Django->>Worker: Start worker
        Worker-->>Django: Worker started
    end

    Django-->>User: Render monitor.html

    loop Live stream
        User->>Django: GET /api/stream/CAM001/
        Django->>Worker: Get latest frame
        Worker-->>Django: MJPEG frame
        Django-->>User: Stream frame
    end
```

### 2. Review Event

```mermaid
sequenceDiagram
    participant User as User Browser
    participant Django as Django View
    participant DB as Database
    participant FS as File System

    User->>Django: GET /video-logs/
    Django->>DB: Event.objects.filter(...).order_by('-created_at')
    DB-->>Django: Events list
    Django-->>User: Render event table

    User->>Django: Click event (id=123)
    Django->>DB: Event.objects.get(id=123)
    DB-->>Django: Event details
    Django->>FS: Check clip exists
    FS-->>Django: Clip path
    Django-->>User: Render event detail<br/>(video player + metadata)

    User->>User: Watch video clip
    User->>Django: POST /event/123/confirm/
    Django->>DB: Event.update({<br/>  status: 'confirmed',<br/>  reviewed_by: current_user<br/>})
    DB-->>Django: Updated
    Django-->>User: Redirect to event list
```

### 3. Configure Camera Settings

```mermaid
sequenceDiagram
    participant User as User Browser
    participant Django as Django View
    participant DB as Database
    participant Worker as Background Worker

    User->>Django: GET /camera/1/settings/
    Django->>DB: Camera.objects.get(id=1)
    DB-->>Django: Camera config
    Django-->>User: Render settings form

    User->>User: Draw cashier zone<br/>(mouse drag on canvas)
    User->>User: Adjust thresholds<br/>(sliders)

    User->>Django: POST /camera/1/settings/ {<br/>  cashier_zone_x: 200,<br/>  cashier_zone_y: 300,<br/>  hand_touch_distance: 120<br/>}

    Django->>DB: Camera.update({...})
    DB-->>Django: Updated

    Django->>Worker: Reload camera config
    Worker-->>Django: Config reloaded

    Django-->>User: Redirect to camera list
```

---

## Related Documentation

- [01 - Architecture](01-architecture.md) - System architecture and background workers
- [04 - Features](04-features.md) - Detection algorithms in detail
- [05 - Data Models](05-data-models.md) - Database schema and Event model
- [06 - Integrations](06-integrations.md) - RTSP streaming and video processing

---

**Previous**: [← 06 - Integrations](06-integrations.md) | **Next**: [08 - Deployment →](08-deployment.md)
