# Troubleshooting - Hotel Cash Detector

Common issues, solutions, and debugging strategies.

## Table of Contents

1. [RTSP Stream Issues](#rtsp-stream-issues)
2. [Detection Problems](#detection-problems)
3. [Performance Issues](#performance-issues)
4. [GPU/CUDA Issues](#gpucuda-issues)
5. [Database Issues](#database-issues)
6. [Video Clip Issues](#video-clip-issues)
7. [Debug Mode](#debug-mode)
8. [Log Locations](#log-locations)

---

## RTSP Stream Issues

### Stream Timeout Errors

**Symptom**:
```
[ WARN:0@30.044] global cap_ffmpeg_impl.hpp:453 Stream timeout triggered after 30043.255000 ms
[h264 @ 0000023ce8cd0940] error while decoding MB 36 28, bytestream -11
```

**Root Cause**: Default OpenCV FFmpeg timeout is 30 seconds, causing disconnects during brief network issues.

**Solution (Fixed in v1.0.1)**:

Extended timeout configuration is already implemented in the system:

```python
# In unified_detector.py
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp|'              # Use TCP (more reliable than UDP)
    'stimeout;60000000|'                # 60s socket timeout (microseconds)
    'max_delay;1000000|'                # 1s max frame delay
    'fflags;nobuffer+discardcorrupt|'  # Low latency, handle errors
    'analyzeduration;2000000|'          # 2s to analyze stream
    'probesize;2000000|'                # 2MB probe size
    'buffer_size;4096000'               # 4MB network buffer
)
```

**Additional Troubleshooting**:

1. **Check camera network connectivity**:
   ```bash
   ping 192.168.1.64
   ```

2. **Verify RTSP URL with VLC**:
   - Open VLC Media Player
   - Media → Open Network Stream
   - Enter RTSP URL
   - If video plays, URL is correct

3. **Test with FFplay** (if FFmpeg installed):
   ```bash
   ffplay rtsp://admin:password@192.168.1.64:554/stream
   ```

4. **Check firewall rules**:
   ```bash
   # Ubuntu/Linux
   sudo ufw status

   # Ensure port 554 is allowed
   sudo ufw allow 554/tcp
   ```

5. **Verify camera supports TCP transport**:
   - Some cameras only support UDP (less reliable)
   - Check camera documentation
   - Try accessing camera web interface

6. **Monitor network bandwidth**:
   - Typical bandwidth: 1-5 Mbps per camera
   - Reduce camera resolution if network is congested
   - Check for network packet loss: `ping -c 100 camera-ip`

---

### Frames Showing 0

**Symptom**:
```
📹 Front Desk Camera
⏱️ Uptime: 00:00:53
📊 Frames: 0
🎯 Events: 0
```

**Root Cause**: Stream failed to connect or initial frame read failed.

**Solution**:

The system includes automatic retry logic (5 attempts) and automatic reconnection.

**Debug Steps**:

1. **Check worker status API**:
   ```bash
   curl http://localhost:8000/api/workers/status/ | python -m json.tool
   ```

   **Look for**:
   ```json
   {
     "camera_id": "CAM001",
     "status": "error",  // Should be "running"
     "last_error": "Failed to read frame after 20 consecutive attempts",
     "frames_processed": 0,
     "uptime_seconds": 53
   }
   ```

2. **Review terminal logs**:
   ```bash
   # If running with systemd
   sudo journalctl -u hotel-cctv -f

   # Look for connection errors
   ```

3. **Test RTSP connection manually**:
   ```bash
   cd django_app
   source venv/bin/activate
   python
   ```

   ```python
   import cv2

   rtsp_url = "rtsp://admin:password@192.168.1.64:554/stream"
   cap = cv2.VideoCapture(rtsp_url)

   if cap.isOpened():
       ret, frame = cap.read()
       if ret:
           print(f"✅ Success: {frame.shape}")
       else:
           print("❌ Could not read frame")
   else:
       print("❌ Could not open stream")

   cap.release()
   ```

4. **Common Fixes**:
   - **Wrong RTSP URL**: Verify format and credentials
   - **Camera offline**: Check camera power and network
   - **Port blocked**: Firewall blocking RTSP port 554
   - **Codec incompatibility**: Camera using unsupported codec
   - **Network congestion**: Too many cameras, insufficient bandwidth

---

### Stream Disconnects During Operation

**Symptom**: Worker shows "reconnecting" status, frames drop to 0 periodically.

**Root Cause**: Temporary network issues or camera restarts.

**Solution**:

The system includes automatic reconnection logic:

```python
# In BackgroundCameraWorker
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
        continue
```

**Best Practices**:
- Use wired Ethernet instead of WiFi for cameras
- Ensure cameras have static IP addresses
- Reduce camera stream resolution if network is unstable
- Monitor network packet loss

---

## Detection Problems

### Cash Detection Not Triggering

**Symptom**: No cash transaction events even when visible hand exchanges occur.

**Checklist**:

1. **Detection Enabled**:
   - Go to Camera Settings
   - Verify "Detect Cash Transactions" is enabled

2. **Cashier Zone Configured**:
   - Go to Camera Settings
   - Ensure cashier zone is drawn around cashier counter
   - Zone should capture cashier's working area

3. **Confidence Threshold Too High**:
   - Check `CASH_DETECTION_CONFIDENCE` in `.env`
   - Default: `0.5`
   - Try lowering to `0.4` if missing detections
   - See [03-env.md](03-env.md) for details

4. **Hand Touch Distance Too Small**:
   - Check `HAND_TOUCH_DISTANCE` in `.env` or Camera Settings
   - Default: `100` pixels
   - For 1080p cameras: Try `120-150`
   - For 720p cameras: Try `80-100`

5. **Pose Model Issues**:
   - Ensure `yolov8s-pose.pt` model is downloaded
   - Check terminal for model loading errors
   - Verify hands are visible in frame (not obscured)

6. **Enable Debug Mode**:
   - Go to Camera Settings → Developer Mode (password: `dev123`)
   - Enable "Show Pose Overlay"
   - Verify:
     - Hands are detected (magenta circles)
     - One person is IN zone (green box)
     - One person is OUT of zone (orange box)
     - Distance line appears between hands

**Debug Commands**:
```bash
# Enable detection logging
# Edit .env
ENABLE_DETECTION_LOGS=True

# Restart worker
# Terminal logs will show:
# [CASH] Cashier (ID 0) IN zone, Customer (ID 1) OUT zone
# [CASH] Hand distance: 87.3px (threshold: 100px) ✅ DETECTED
```

---

### False Positives (Too Many Detections)

**Symptom**: Detection triggers for non-transaction events (cashier adjusting items, customer waving, etc.).

**Solutions**:

1. **Increase Confidence Threshold**:
   ```env
   CASH_DETECTION_CONFIDENCE=0.6  # Higher = fewer detections
   ```

2. **Increase Hand Touch Distance Threshold**:
   ```env
   HAND_TOUCH_DISTANCE=80  # Smaller = hands must be closer
   ```

3. **Increase Minimum Transaction Frames**:
   ```env
   MIN_TRANSACTION_FRAMES=3  # Require 3 consecutive frames
   ```

4. **Enable Gemini AI Validation** (Optional):
   ```env
   GEMINI_API_KEY=your-gemini-api-key
   GEMINI_VALIDATION_ENABLED=True
   GEMINI_USE_VIDEO_VALIDATION=False
   ```

   Gemini AI validates detections and filters false positives.

5. **Adjust Cooldown Period**:
   ```env
   TRANSACTION_COOLDOWN=60  # Wait 60 frames (2s at 30fps) between events
   ```

---

### Violence Detection Not Working

**Symptom**: No violence events during test scenarios.

**Checklist**:

1. **Detection Enabled**:
   - Verify "Detect Violence" is enabled in Camera Settings

2. **Confidence Threshold**:
   ```env
   VIOLENCE_DETECTION_CONFIDENCE=0.6
   ```
   - Try lowering to `0.5` if missing events

3. **Motion Threshold**:
   ```env
   MOTION_THRESHOLD=100
   ```
   - Lower value = more sensitive
   - Try `80` for higher sensitivity

4. **Minimum Violence Frames**:
   ```env
   MIN_VIOLENCE_FRAMES=15  # 0.5s at 30fps
   ```
   - Lower value = faster detection
   - Try `10` for quicker alerts

5. **Scene Requirements**:
   - Violence detection requires:
     - Two or more people in frame
     - Close proximity (overlapping bounding boxes)
     - Aggressive poses (raised arms, rapid motion)
     - Sustained motion over multiple frames

---

### Fire Detection Not Working

**Symptom**: No fire events even when fire is visible.

**Checklist**:

1. **Detection Enabled**:
   - Verify "Detect Fire" is enabled in Camera Settings

2. **Fire Model Loaded**:
   - Verify `fire_smoke_yolov8.pt` exists in `models/` directory
   - Check terminal for model loading errors

3. **Confidence Threshold**:
   ```env
   FIRE_DETECTION_CONFIDENCE=0.5
   ```
   - Try lowering to `0.4`

4. **Minimum Fire Area**:
   ```env
   MIN_FIRE_AREA=3000  # pixels²
   ```
   - Lower value detects smaller fires
   - Try `2000` for earlier detection

5. **Minimum Fire Frames**:
   ```env
   MIN_FIRE_FRAMES=10
   ```
   - Lower value = faster alerts
   - Try `5` for quicker detection

6. **Lighting Conditions**:
   - Fire detection works best in normal lighting
   - Poor lighting may affect color-based detection
   - Ensure camera has adequate exposure

---

## Performance Issues

### High CPU Usage

**Symptom**: CPU usage at 80-100%, system slow or unresponsive.

**Solutions**:

1. **Use GPU Acceleration**:
   ```env
   USE_GPU=cuda  # Force GPU if available
   ```

   **Verify GPU is being used**:
   ```bash
   nvidia-smi

   # Look for python process using GPU memory
   ```

2. **Use Lighter Models**:
   ```env
   YOLO_MODEL=models/yolov8n.pt       # Nano (6MB, faster)
   POSE_MODEL=models/yolov8n-pose.pt  # Nano Pose (7MB)
   ```

   **Model Comparison**:
   | Model | Size | CPU Time | GPU Time | Accuracy |
   |-------|------|----------|----------|----------|
   | yolov8n | 6MB | ~100ms | ~15ms | Good |
   | yolov8s | 22MB | ~150ms | ~20ms | Better (default) |

3. **Reduce Number of Cameras**:
   - Each camera worker runs independently
   - Limit to 4-6 cameras on non-GPU systems
   - Use GPU for 8+ cameras

4. **Lower Camera Resolution**:
   - Configure camera to stream 720p instead of 1080p
   - Reduces processing load significantly

5. **Disable Unused Detections**:
   - If only cash detection needed, disable violence and fire
   - Go to Camera Settings and toggle off unused detections

---

### High GPU Memory Usage

**Symptom**:
```
RuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB
```

**Solutions**:

1. **Use Smaller Models**:
   ```env
   YOLO_MODEL=models/yolov8n.pt
   POSE_MODEL=models/yolov8n-pose.pt
   ```

2. **Reduce Number of Simultaneous Cameras**:
   - Each camera uses ~500MB-1GB GPU memory
   - GTX 1660 (6GB): Max 4-5 cameras
   - RTX 3060 (12GB): Max 8-10 cameras
   - Tesla T4 (16GB): Max 10-12 cameras

3. **Clear GPU Cache**:
   ```bash
   cd django_app
   source venv/bin/activate
   python
   ```

   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Restart Detection Workers**:
   - Stop all workers
   - Clear GPU cache
   - Start workers one at a time

---

### Low FPS (Frames Per Second)

**Symptom**: Processing < 15 FPS, video appears choppy.

**Expected Performance**:
| Hardware | Cameras | FPS | Notes |
|----------|---------|-----|-------|
| i7 + GTX 1660 | 1 | 25-30 | Smooth |
| i7 + GTX 1660 | 4 | 25-30 | Good |
| i5 (no GPU) | 1 | 10-15 | Use yolov8n |
| AWS g4dn.xlarge | 4 | 30 | Recommended |

**Solutions**:

1. **Enable GPU**:
   ```env
   USE_GPU=auto  # or cuda
   ```

2. **Use Faster Models**:
   ```env
   YOLO_MODEL=models/yolov8n.pt
   ```

3. **Reduce Camera Resolution**:
   - 1080p → 720p (significant speedup)

4. **Close Other Applications**:
   - Free up CPU/GPU resources

---

## GPU/CUDA Issues

### CUDA Not Available

**Symptom**:
```python
import torch
print(torch.cuda.is_available())  # Returns False
```

**Solutions**:

1. **Install CUDA Toolkit**:
   - Download from [nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Install CUDA 11.8 or 12.1

2. **Install PyTorch with CUDA Support**:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify NVIDIA Drivers**:
   ```bash
   nvidia-smi

   # If command not found, install NVIDIA drivers
   ```

   **Ubuntu**:
   ```bash
   sudo ubuntu-drivers autoinstall
   sudo reboot
   ```

4. **Check PyTorch CUDA Version**:
   ```python
   import torch
   print(torch.__version__)  # Should show +cu118 or +cu121
   print(torch.version.cuda)  # Should match installed CUDA
   ```

---

### GPU Out of Memory

**Symptom**:
```
RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB
```

**Solutions**:

1. **Use Smaller Models**:
   ```env
   YOLO_MODEL=models/yolov8n.pt
   POSE_MODEL=models/yolov8n-pose.pt
   ```

2. **Reduce Cameras**:
   - Stop some workers to free GPU memory

3. **Clear GPU Cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

4. **Restart Workers**:
   ```bash
   sudo systemctl restart hotel-cctv
   ```

5. **Check GPU Memory Usage**:
   ```bash
   nvidia-smi

   # Look for GPU memory usage
   # Kill other GPU processes if needed
   ```

---

## Database Issues

### Migration Errors

**Symptom**:
```
django.db.utils.ProgrammingError: relation "cctv_camera" does not exist
```

**Solution**:

1. **Run migrations**:
   ```bash
   cd django_app
   source venv/bin/activate
   python manage.py migrate
   ```

2. **Check migration status**:
   ```bash
   python manage.py showmigrations

   # All migrations should have [X]
   ```

3. **Reset database** (CAUTION: Deletes all data):
   ```bash
   # SQLite
   rm db.sqlite3
   python manage.py migrate
   python manage.py createsuperuser

   # PostgreSQL
   sudo -u postgres psql
   DROP DATABASE cctv;
   CREATE DATABASE cctv;
   GRANT ALL PRIVILEGES ON DATABASE cctv TO orange;
   \q

   python manage.py migrate
   python manage.py createsuperuser
   ```

---

### PostgreSQL Permission Denied

**Symptom**:
```
django.db.utils.ProgrammingError: permission denied for schema public
```

**Solution (PostgreSQL 15+)**:

```bash
sudo -u postgres psql

\c cctv
GRANT ALL ON SCHEMA public TO orange;
GRANT CREATE ON SCHEMA public TO orange;
GRANT USAGE ON SCHEMA public TO orange;
ALTER DATABASE cctv OWNER TO orange;
ALTER SCHEMA public OWNER TO orange;
\q
```

See [POSTGRESQL_SETUP.md](../POSTGRESQL_SETUP.md) for details.

---

### Database Connection Refused

**Symptom**:
```
django.db.utils.OperationalError: could not connect to server: Connection refused
```

**Solutions**:

1. **Check PostgreSQL is running**:
   ```bash
   sudo systemctl status postgresql

   # If not running:
   sudo systemctl start postgresql
   ```

2. **Check PostgreSQL is listening**:
   ```bash
   sudo netstat -plnt | grep 5432

   # Should show:
   # tcp        0      0 127.0.0.1:5432          0.0.0.0:*               LISTEN
   ```

3. **Check `.env` database configuration**:
   ```env
   DB_HOST=localhost  # Not 127.0.0.1 if using Unix socket
   DB_PORT=5432
   DB_USER=orange
   DB_PASSWORD=00oo00oo
   ```

4. **Check PostgreSQL logs**:
   ```bash
   sudo tail -f /var/log/postgresql/postgresql-16-main.log
   ```

---

## Video Clip Issues

### Video Clips Not Playing in Browser

**Symptom**: Video element shows loading spinner indefinitely.

**Solutions**:

1. **Check FFmpeg is installed**:
   ```bash
   ffmpeg -version

   # If not installed:
   # Ubuntu
   sudo apt install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **Verify clip is H.264 encoded**:
   ```bash
   ffprobe media/clips/event_123_clip.mp4

   # Look for:
   # Video: h264 (High)
   ```

3. **Check file permissions**:
   ```bash
   ls -la media/clips/

   # Files should be readable
   chmod 644 media/clips/*.mp4
   ```

4. **Verify movflags faststart**:

   Clips should be encoded with `movflags +faststart` for web streaming. This is handled automatically by the system.

5. **Test clip locally**:
   ```bash
   ffplay media/clips/event_123_clip.mp4

   # Or open in VLC
   ```

---

### Clips Not Being Saved

**Symptom**: Events logged, but `clip_path` is empty.

**Solutions**:

1. **Check FFmpeg is installed** (see above)

2. **Check disk space**:
   ```bash
   df -h

   # Ensure sufficient space in media directory
   ```

3. **Check media directory permissions**:
   ```bash
   ls -la django_app/media/

   # Should be writable
   chmod 755 django_app/media/
   chmod 755 django_app/media/clips/
   ```

4. **Check worker logs**:
   ```bash
   # Look for FFmpeg errors
   sudo journalctl -u hotel-cctv -f | grep -i ffmpeg
   ```

---

## Debug Mode

### Accessing Developer Mode

**Purpose**: Real-time detection visualization and threshold tuning.

**Steps**:

1. Navigate to **Camera Settings**
2. Click **"Developer"** button
3. Enter password: `dev123`
4. Enable features:
   - **Show Pose Overlay**: Visual detection overlay
   - **Detection Info**: Real-time detection logs

### Debug Overlay Features

**Pose Overlay Shows**:
- Person bounding boxes:
  - **Green**: Cashier (IN zone)
  - **Orange**: Customer (OUT of zone)
- **Center point marker**: "C" at person center
- **Hand circles**: Magenta circles at wrists
- **Distance lines** between hands:
  - **Green**: Valid detection (cashier-customer, close enough)
  - **Gray**: Ignored (cashier-cashier or customer-customer)
  - **Red**: Too far apart
- **Distance labels**: Pixel values

**Use Cases**:
- Verify cashier zone is correct
- Tune hand touch distance threshold
- Debug why detections aren't triggering
- Understand person classification logic

---

## Log Locations

| Log Type | Location | Access |
|----------|----------|--------|
| **Django Server** | Console/stdout | Terminal where `runserver` was started |
| **Detection Logs** | Console/stdout | Terminal (if `ENABLE_DETECTION_LOGS=True`) |
| **Worker Logs** | Console/stdout | Terminal or systemd journal |
| **Systemd Logs** | `/var/log/syslog` | `sudo journalctl -u hotel-cctv -f` |
| **PostgreSQL Logs** | `/var/log/postgresql/` | `sudo tail -f /var/log/postgresql/*.log` |
| **Nginx Logs** | `/var/log/nginx/` | `sudo tail -f /var/log/nginx/error.log` |

**Enable Detection Logging**:
```env
# In .env
ENABLE_DETECTION_LOGS=True
```

**Expected Output**:
```
[CASH] Cashier (ID 0) IN zone at (640, 360)
[CASH] Customer (ID 1) OUT zone at (1000, 400)
[CASH] Hand distance: 87.3px (threshold: 100px) ✅ DETECTED
[EVENT] Saving event: cash transaction, confidence 0.82
[CLIP] Saving 30-second clip from frame 450-1350
```

---

## Performance Benchmarks

**Expected Performance**:

| Hardware | Cameras | FPS | CPU % | GPU % | Notes |
|----------|---------|-----|-------|-------|-------|
| **i7 + GTX 1660** | 1 | 30 | 25% | 40% | Smooth operation |
| **i7 + GTX 1660** | 4 | 30 | 60% | 75% | Acceptable |
| **i5 (no GPU)** | 1 | 15 | 80% | - | Use yolov8n models |
| **AWS g4dn.xlarge** | 4 | 30 | 30% | 50% | Recommended |
| **AWS g4dn.2xlarge** | 10 | 30 | 40% | 60% | Production ready |

**Model Inference Times**:

| Model | GPU (GTX 1660) | CPU (i7) | Notes |
|-------|----------------|----------|-------|
| YOLOv8s | ~20ms | ~150ms | Default (better accuracy) |
| YOLOv8s-Pose | ~25ms | ~200ms | Default |
| YOLOv8n | ~15ms | ~100ms | Faster (good accuracy) |
| YOLOv8n-Pose | ~18ms | ~120ms | Faster |
| Fire/Smoke | ~10ms | ~80ms | Custom-trained |

---

## Getting Help

If you encounter issues not covered here:

1. **Check Documentation**:
   - [00 - Overview](00-overview.md)
   - [02 - Setup](02-setup.md)
   - [03 - Environment Variables](03-env.md)
   - [06 - Integrations](06-integrations.md)

2. **Enable Debug Logging**:
   ```env
   ENABLE_DETECTION_LOGS=True
   DEBUG=True
   ```

3. **Check System Status**:
   ```bash
   # Worker status
   curl http://localhost:8000/api/workers/status/

   # GPU status (if using GPU)
   nvidia-smi

   # Systemd status
   sudo systemctl status hotel-cctv
   ```

4. **Review Logs**:
   ```bash
   # Systemd logs (last 100 lines)
   sudo journalctl -u hotel-cctv -n 100

   # Real-time logs
   sudo journalctl -u hotel-cctv -f
   ```

5. **Contact Development Team**:
   - Provide error logs
   - Include system specifications
   - Describe steps to reproduce

---

## Related Documentation

- [02 - Setup](02-setup.md) - Development and production setup
- [03 - Environment Variables](03-env.md) - Configuration reference
- [06 - Integrations](06-integrations.md) - RTSP, GPU, PMS integration
- [08 - Deployment](08-deployment.md) - Production deployment

---

**Previous**: [← 08 - Deployment](08-deployment.md)
