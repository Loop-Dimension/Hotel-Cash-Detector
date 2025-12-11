# Multiprocessing Worker Architecture

## Overview

The Hotel-Cash-Detector system uses **separate CPU cores** for each camera detection worker to ensure:
- **Isolation**: Each camera runs in its own process - crashes don't affect other cameras
- **Performance**: Dedicated CPU cores prevent interference between workers
- **Stability**: No shared memory issues or GIL (Global Interpreter Lock) contention
- **Scalability**: Can easily utilize all 4 cores on g4dn.xlarge instance

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Django Main Process (Core 0)            │
│  • Web requests, API, database                      │
│  • Worker management                                │
└──────────────┬──────────────────────────────────────┘
               │
               ├─────────────────────────────────────┐
               │                                     │
        ┌──────▼──────┐                    ┌────────▼──────┐
        │  Worker 1   │                    │   Worker 2    │
        │  (Core 1)   │                    │   (Core 2)    │
        │             │                    │               │
        │ • Camera-1  │                    │ • Camera-2    │
        │ • RTSP      │                    │ • RTSP        │
        │ • Detection │                    │ • Detection   │
        │ • Clip Save │                    │ • Clip Save   │
        └─────────────┘                    └───────────────┘
```

## Key Components

### 1. CameraWorkerProcess (`cctv/worker_process.py`)
- Manages individual worker process lifecycle
- Sets CPU affinity to dedicated core
- Provides shared memory for status communication
- Handles graceful shutdown and cleanup

### 2. Shared Memory Communication
- **Status**: running/stopped/error/reconnecting
- **Counters**: frames_processed, events_detected
- **Timing**: start_timestamp for uptime calculation
- **Frames**: Queue for live video feed (2-frame buffer)

### 3. CPU Core Assignment
```python
# Assign to specific CPU core (round-robin based on camera_id)
core_id = camera_id % cpu_count
p.cpu_affinity([core_id])
```

On 4-core system:
- Camera 1 → Core 1
- Camera 2 → Core 2  
- Camera 3 → Core 3
- Camera 4 → Core 0 (wraps around)

## Worker Process Lifecycle

### Starting a Worker
```python
# User clicks "Start" button in dashboard
POST /camera/<id>/start-worker/

# Creates new process
worker = CameraWorkerProcess(camera_id)
worker.start()

# Worker connects to RTSP stream
# Worker loads AI models (YOLOv8)
# Worker starts frame processing loop
```

### Worker Running
```python
while running and not stop_event.is_set():
    # 1. Read frame from RTSP stream
    ret, frame = cap.read()
    
    # 2. Buffer every 2nd frame for clips (reduces memory)
    frame_buffer.append(frame.copy())
    
    # 3. Send frame to live video queue (every 4th frame)
    frame_queue.put_nowait(frame.copy())
    
    # 4. Process detection (every 4th frame)
    if frame_count % 4 == 0:
        result = detector.process_frame(frame)
        
        # 5. Save event + clip if detection found
        if detection_found:
            save_clip(frame_buffer)
            save_event(detection_data)
```

### Stopping a Worker
```python
# User clicks "Stop" button
POST /camera/<id>/stop-worker/

# Graceful shutdown (10 second timeout)
worker.stop(timeout=10)

# 1. Set stop_event flag
# 2. Worker exits loop
# 3. Releases RTSP connection
# 4. Updates camera status to offline
# 5. Process terminates
```

## Advantages Over Threading

| Feature | Multiprocessing | Threading (Old) |
|---------|----------------|-----------------|
| **CPU Cores** | Uses all 4 cores | Limited by GIL |
| **Isolation** | Complete separation | Shared memory |
| **Crashes** | Isolated to one worker | Can crash entire app |
| **Performance** | ~4x throughput | Limited scaling |
| **Memory** | Separate heap per process | Shared heap |
| **Debugging** | Each process has own PID | Hard to track |

## Status Monitoring

Dashboard polls status every 2 seconds:
```javascript
GET /background-workers/status/

Response:
{
  "workers": {
    "1": {
      "camera_id": "cam-1",
      "status": "running",
      "running": true,
      "frames_processed": 1234,
      "events_detected": 5,
      "uptime": "00:15:32"
    }
  }
}
```

## Video Streaming Integration

When user views live feed:
1. Check if worker exists for camera
2. If yes, get frames from worker's queue (no new RTSP connection!)
3. If no, create temporary connection (legacy mode)

```python
worker = background_workers.get(camera_id)
if worker and worker.is_alive():
    frame = worker.get_current_frame()  # From shared queue
else:
    # Fall back to direct connection
```

## Resource Management

### Memory Usage
- Each worker: ~500MB (model + frame buffer)
- 4 workers: ~2GB total
- Django: ~500MB
- **Total: ~2.5GB / 16GB available** ✅

### CPU Usage (g4dn.xlarge)
- Core 0: Django web + worker manager (20-30%)
- Core 1: Camera 1 worker (70-90%)
- Core 2: Camera 2 worker (70-90%)
- Core 3: Camera 3 worker (70-90%)

### GPU Usage (NVIDIA T4)
- Shared across all workers
- YOLOv8 inference: ~2GB VRAM per model
- Total: ~4GB / 16GB VRAM ✅

## Error Handling

### Worker Crashes
```python
# Main process detects dead worker
if not worker.is_alive():
    del background_workers[camera_id]
    # User sees "Stopped" in dashboard
```

### Stream Disconnects
```python
# Worker attempts reconnection (5 retries)
for attempt in range(5):
    cap = create_rtsp_capture(url)
    if cap.isOpened():
        break
    time.sleep(5)

# If all retries fail, worker exits
```

### Memory Leaks
Each process is isolated - if leak occurs:
1. Stop worker
2. Process terminates completely
3. Start new worker - fresh memory

## Configuration

### Frame Processing
- Read: Every frame (30fps)
- Buffer: Every 2nd frame (15fps, for clips)
- Live stream: Every 4th frame (7.5fps, reduces network load)
- Detection: Every 4th frame (7.5fps, reduces CPU load)

### Clip Settings
- Duration: 30 seconds
- Buffer size: 450 frames (30s × 15fps)
- Format: H.264 MP4 (browser compatible)
- Quality: CRF 23 (balanced size/quality)

### Event Cooldown
- Same event type: 15 seconds minimum gap
- Prevents duplicate detections
- Allows multiple event types simultaneously

## Deployment Notes

### AWS Setup
```bash
# Install psutil for CPU affinity
pip install psutil

# Verify CPU cores
python -c "import psutil; print(f'CPUs: {psutil.cpu_count()}')"

# Start Django with workers
python manage.py runserver 0.0.0.0:8000
# OR
gunicorn hotel_cctv.wsgi:application --bind 0.0.0.0:8000
```

### Auto-Start on Boot
Workers start automatically when Django starts:
```python
# In ready() method of AppConfig
start_all_background_workers_internal()
```

### Monitoring
```bash
# View all processes
ps aux | grep python

# Check CPU usage per process
htop -p <worker_pid>

# Monitor GPU
watch -n 1 nvidia-smi
```

## Troubleshooting

### Worker Won't Start
```bash
# Check logs
sudo journalctl -u hotel-cctv -f

# Common issues:
# 1. RTSP URL unreachable
# 2. Models not downloaded
# 3. GPU drivers missing
```

### High CPU Usage
```python
# Reduce detection frequency
if frame_count % 8 == 0:  # Changed from 4 to 8
    result = detector.process_frame(frame)
```

### Memory Issues
```python
# Reduce frame buffer size
buffer_size = 300  # 20 seconds instead of 30
```

## Future Enhancements

1. **Dynamic Core Assignment**: Assign workers to least-busy cores
2. **GPU Scheduling**: Queue GPU operations to prevent contention
3. **Distributed Workers**: Run workers on separate machines
4. **Load Balancing**: Distribute cameras across multiple servers
5. **Health Checks**: Auto-restart crashed workers

---

**Version**: 2.0 (Multiprocessing)  
**Date**: December 11, 2025  
**Author**: Loop-Dimension
