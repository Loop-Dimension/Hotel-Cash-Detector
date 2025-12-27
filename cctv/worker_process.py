"""
Multiprocessing worker for camera detection - each camera gets dedicated CPU core(s)
This isolates workers completely to prevent interference and crashes.
"""
import os
import sys
import django
import cv2
import time
import json
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Manager

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hotel_cctv.settings')
django.setup()

from django.conf import settings
from cctv.models import Camera, Event

# Import detectors
try:
    from detectors import UnifiedDetector
    DETECTOR_AVAILABLE = True
except ImportError as e:
    DETECTOR_AVAILABLE = False
    print(f"Warning: Detectors not available - {e}")

# Import Gemini validator
try:
    from detectors.gemini_validator import GeminiValidator
    GEMINI_AVAILABLE = True
except ImportError as e:
    GEMINI_AVAILABLE = False
    print(f"Warning: Gemini validator not available - {e}")

# Global manager for all workers (create once, reuse)
_manager = None

def get_manager():
    """Get or create the global Manager instance"""
    global _manager
    if _manager is None:
        _manager = Manager()
    return _manager


class CameraWorkerProcess:
    """Isolated camera worker running in separate process with dedicated CPU core"""
    
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.process = None
        
        # Use global Manager for cross-platform shared state
        manager = get_manager()
        
        # Shared state dictionary (works on Windows + Linux)
        self.shared_state = manager.dict({
            'running': False,
            'frames_processed': 0,
            'events_detected': 0,
            'start_timestamp': 0.0,
            'status': 'stopped',
            'error': '',
        })
        
        # Queues for communication
        self.command_queue = manager.Queue(maxsize=10)
        self.frame_queue = manager.Queue(maxsize=2)  # Small queue for live frames
        
        # Stop event
        self.stop_flag = manager.Value('i', 0)  # 0=run, 1=stop
    
    def start(self):
        """Start worker process"""
        if self.process and self.process.is_alive():
            return False
        
        self.stop_flag.value = 0
        self.shared_state['running'] = True
        
        self.process = Process(
            target=_worker_main,
            args=(
                self.camera_id,
                self.shared_state,
                self.command_queue,
                self.frame_queue,
                self.stop_flag,
            ),
            daemon=True
        )
        self.process.start()
        return True
    
    def stop(self, timeout=10):
        """Stop worker process gracefully"""
        if not self.process or not self.process.is_alive():
            return True
        
        # Signal stop
        self.stop_flag.value = 1
        self.shared_state['running'] = False
        
        # Wait for graceful shutdown
        self.process.join(timeout=timeout)
        
        # Force terminate if still alive
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        
        if self.process.is_alive():
            self.process.kill()
        
        return True
    
    def get_status(self):
        """Get worker status"""
        state = dict(self.shared_state)
        
        uptime = None
        if state.get('start_timestamp', 0) > 0:
            elapsed = time.time() - state['start_timestamp']
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        return {
            'running': state.get('running', False),
            'status': state.get('status', 'unknown'),
            'error': state.get('error', None),
            'frames_processed': state.get('frames_processed', 0),
            'events_detected': state.get('events_detected', 0),
            'uptime': uptime or 'Not running',
            'alive': self.process.is_alive() if self.process else False,
        }
    
    def get_current_frame(self):
        """Get latest frame from worker (non-blocking)"""
        try:
            if not self.frame_queue.empty():
                return self.frame_queue.get_nowait()
        except:
            pass
        return None
    
    def is_alive(self):
        """Check if worker process is alive"""
        return self.process and self.process.is_alive()


def _worker_main(camera_id, shared_state, command_queue, frame_queue, stop_flag):
    """Main worker function running in separate process
    
    This runs with CPU affinity set to dedicated core(s) for isolation.
    """
    # Set CPU affinity for this process (Linux only)
    try:
        import psutil
        p = psutil.Process()
        cpu_count = psutil.cpu_count()
        # Assign to specific CPU core (round-robin based on camera_id)
        core_id = camera_id % cpu_count
        p.cpu_affinity([core_id])
        print(f"[Worker-{camera_id}] Assigned to CPU core {core_id}")
    except Exception as e:
        print(f"[Worker-{camera_id}] Could not set CPU affinity: {e}")
    
    # Close Django DB connection (will create new one in this process)
    from django.db import connection
    connection.close()
    
    # Initialize status
    shared_state['status'] = 'starting'
    shared_state['error'] = ''
    shared_state['start_timestamp'] = time.time()
    
    try:
        _run_worker_loop(
            camera_id, shared_state, command_queue, frame_queue, stop_flag
        )
    except Exception as e:
        error_msg = f"Worker crashed: {str(e)}"
        shared_state['error'] = error_msg[:199]
        shared_state['status'] = 'error'
        print(f"[Worker-{camera_id}] CRASH: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shared_state['running'] = False
        shared_state['status'] = 'stopped'
        print(f"[Worker-{camera_id}] Process terminated")


def _run_worker_loop(camera_id, shared_state, command_queue, frame_queue, stop_flag):
    """Main detection loop for worker process"""
    
    # Get camera from database
    try:
        camera = Camera.objects.get(id=camera_id)
    except Camera.DoesNotExist:
        shared_state['error'] = 'Camera not found'
        shared_state['status'] = 'error'
        return
    
    if not DETECTOR_AVAILABLE:
        shared_state['error'] = 'Detector not available'
        shared_state['status'] = 'error'
        return
    
    # Create detector
    zone = camera.get_cashier_zone()
    detector_config = {
        'models_dir': str(settings.BASE_DIR / 'models'),
        # GPU/CPU setting from environment
        'use_gpu': settings.DETECTION_CONFIG.get('USE_GPU', 'auto'),
        # Model selection - cash uses pose, violence/fire use nano
        'cash_pose_model': settings.DETECTION_CONFIG.get('CASH_POSE_MODEL', 'yolov8s-pose.pt'),
        'violence_pose_model': settings.DETECTION_CONFIG.get('VIOLENCE_POSE_MODEL', 'yolov8n-pose.pt'),
        'fire_yolo_model': settings.DETECTION_CONFIG.get('FIRE_YOLO_MODEL', 'yolov8n.pt'),
        'fire_model': settings.DETECTION_CONFIG.get('FIRE_MODEL', 'fire_smoke_yolov8.pt'),
        'cashier_zone': [zone['x'], zone['y'], zone['width'], zone['height']],
        'use_polygon_zones': True,  # POLYGON-ONLY MODE
        'cashier_zone_polygon': camera.get_cashier_zone_polygon_points(),
        'cash_drawer_zone': [camera.cash_drawer_zone_x, camera.cash_drawer_zone_y, 
                             camera.cash_drawer_zone_width, camera.cash_drawer_zone_height],
        'cash_drawer_zone_polygon': camera.get_cash_drawer_zone_polygon_points() if hasattr(camera, 'get_cash_drawer_zone_polygon_points') else None,
        'hand_touch_distance': camera.hand_touch_distance,
        'pose_confidence': 0.5,
        'min_transaction_frames': 1,
        'fire_confidence': camera.fire_confidence,
        'min_fire_frames': 3,
        'violence_confidence': camera.violence_confidence,
        'min_violence_frames': 10,
        'detect_cash': camera.detect_cash,
        'detect_violence': camera.detect_violence,
        'detect_fire': camera.detect_fire,
        'cash_confidence': camera.cash_confidence,
    }
    
    detector = UnifiedDetector(detector_config)
    
    # Connect to RTSP stream
    shared_state['status'] = 'connecting'
    cap = _create_rtsp_capture(camera.rtsp_url)
    
    max_retries = 5
    for attempt in range(max_retries):
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"[Worker-{camera_id}] Connected to stream")
                break
        
        print(f"[Worker-{camera_id}] Connection attempt {attempt + 1}/{max_retries}")
        time.sleep(5)
        cap.release()
        cap = _create_rtsp_capture(camera.rtsp_url)
    
    if not cap.isOpened():
        shared_state['error'] = 'Cannot connect to stream'
        shared_state['status'] = 'error'
        camera.status = 'offline'
        camera.save()
        return
    
    # Update camera status
    camera.status = 'online'
    camera.save()
    shared_state['status'] = 'running'
    
    # Frame buffer for clips
    frame_buffer = []
    buffer_size = 450  # 30 seconds at 15fps
    
    # Event cooldown tracking
    last_event_time = {}
    event_cooldown = 15  # seconds
    
    frame_count = 0
    consecutive_failures = 0
    max_failures = 20
    last_success_time = time.time()
    
    print(f"[Worker-{camera_id}] Starting detection loop")
    
    while shared_state['running'] and stop_flag.value == 0:
        # Check for commands
        try:
            if not command_queue.empty():
                cmd = command_queue.get_nowait()
                if cmd == 'stop':
                    break
                elif cmd == 'reload_settings':
                    # Reload camera settings
                    camera = Camera.objects.get(id=camera_id)
                    detector.detect_cash = camera.detect_cash
                    detector.detect_violence = camera.detect_violence
                    detector.detect_fire = camera.detect_fire
        except:
            pass
        
        # Read frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            consecutive_failures += 1
            time_since_success = time.time() - last_success_time
            
            if consecutive_failures >= max_failures or time_since_success > 30:
                shared_state['status'] = 'reconnecting'
                print(f"[Worker-{camera_id}] Stream lost, reconnecting...")
                cap.release()
                time.sleep(3)
                cap = _create_rtsp_capture(camera.rtsp_url)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        shared_state['status'] = 'running'
                        consecutive_failures = 0
                        last_success_time = time.time()
            continue
        
        # Successful frame read
        consecutive_failures = 0
        last_success_time = time.time()
        frame_count += 1
        shared_state['frames_processed'] = frame_count
        
        # Buffer every 2nd frame for clips
        if frame_count % 2 == 0:
            frame_buffer.append(frame.copy())
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)
        
        # Send frame for live viewing (every 4th frame to reduce queue pressure)
        if frame_count % 4 == 0:
            try:
                if frame_queue.full():
                    frame_queue.get_nowait()  # Remove old frame
                frame_queue.put_nowait(frame.copy())
            except:
                pass
        
        # Process detection (every 4th frame)
        if frame_count % 4 == 0:
            try:
                result = detector.process_frame(frame.copy(), draw_overlay=False)
                
                # Handle detections
                if result.get('detections'):
                    for det in result['detections']:
                        det_label = det.get('label', '').lower()
                        confidence = det.get('confidence', 0)
                        bbox = det.get('bbox')
                        
                        if 'cash' in det_label:
                            event_type = 'cash'
                        elif 'violence' in det_label:
                            event_type = 'violence'
                        elif 'fire' in det_label:
                            event_type = 'fire'
                        else:
                            continue
                        
                        # Check cooldown
                        now = datetime.now()
                        last_time = last_event_time.get(event_type)
                        if last_time and (now - last_time).total_seconds() < event_cooldown:
                            continue
                        
                        # Gemini AI Validation - verify detection before saving
                        gemini_validated = True
                        gemini_confidence = 1.0
                        gemini_reason = "Validation skipped"
                        
                        if GEMINI_AVAILABLE and settings.DETECTION_CONFIG.get('GEMINI_VALIDATION_ENABLED', True):
                            try:
                                gemini_api_key = getattr(settings, 'GEMINI_API_KEY', '')
                                if gemini_api_key:
                                    # Create validator with camera_id for logging
                                    validator = GeminiValidator(api_key=gemini_api_key, camera_id=camera.id)
                                    
                                    # Set custom prompts if defined for this camera
                                    custom_prompts = {}
                                    if camera.gemini_cash_prompt:
                                        custom_prompts['cash'] = camera.gemini_cash_prompt
                                    if camera.gemini_violence_prompt:
                                        custom_prompts['violence'] = camera.gemini_violence_prompt
                                    if camera.gemini_fire_prompt:
                                        custom_prompts['fire'] = camera.gemini_fire_prompt
                                    if custom_prompts:
                                        validator.set_custom_prompts(custom_prompts)
                                    
                                    gemini_validated, gemini_confidence, gemini_reason = validator.validate_event(frame, event_type)
                                    print(f"[Worker-{camera_id}] Gemini validation: {event_type} = {gemini_validated} ({gemini_reason})")
                                    
                                    if not gemini_validated:
                                        print(f"[Worker-{camera_id}] Event rejected by Gemini: {event_type} - {gemini_reason}")
                                        continue  # Skip saving this event
                            except Exception as e:
                                print(f"[Worker-{camera_id}] Gemini validation error: {e}")
                                # On error, allow the event (don't block on validation errors)
                        
                        # Save event with clip (only if Gemini validated)
                        clip_path, thumb_path = _save_clip(
                            frame_buffer[-150:] if len(frame_buffer) >= 150 else frame_buffer,
                            camera, event_type
                        )
                        
                        if clip_path:
                            _save_event(camera, event_type, confidence, frame_count, bbox, clip_path, thumb_path, 
                                       gemini_validated=gemini_validated, gemini_confidence=gemini_confidence, gemini_reason=gemini_reason)
                            shared_state['events_detected'] = shared_state.get('events_detected', 0) + 1
                            last_event_time[event_type] = now
                            print(f"[Worker-{camera_id}] Event saved: {event_type} (Gemini: {gemini_reason})")
                        
            except Exception as e:
                error_msg = f"Detection error: {str(e)}"
                shared_state['error'] = error_msg[:199]
                print(f"[Worker-{camera_id}] {error_msg}")
    
    # Cleanup
    cap.release()
    camera.status = 'offline'
    camera.save()
    print(f"[Worker-{camera_id}] Loop ended")


def _create_rtsp_capture(rtsp_url):
    """Create RTSP capture with optimized settings"""
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;60000000|max_delay;1000000|fflags;nobuffer+discardcorrupt'
    
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
    
    return cap


def _save_clip(frames, camera, event_type):
    """Save video clip"""
    if not frames or len(frames) == 0:
        return None, None
    
    import subprocess
    import uuid
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:6]
    
    clip_dir = Path(settings.MEDIA_ROOT) / 'clips'
    clip_dir.mkdir(parents=True, exist_ok=True)
    
    thumb_dir = Path(settings.MEDIA_ROOT) / 'thumbnails'
    thumb_dir.mkdir(parents=True, exist_ok=True)
    
    temp_filename = f"{camera.camera_id}_{event_type}_{timestamp}_{unique_id}_temp.avi"
    final_filename = f"{camera.camera_id}_{event_type}_{timestamp}.mp4"
    
    temp_path = clip_dir / temp_filename
    final_path = clip_dir / final_filename
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(temp_path), fourcc, 15, (width, height))
    
    if not out.isOpened():
        return None, None
    
    for frame in frames:
        if frame is None:
            continue
        label = f"{event_type.upper()} DETECTED"
        color = {'cash': (0, 255, 0), 'violence': (0, 0, 255), 'fire': (0, 165, 255)}.get(event_type, (255, 255, 255))
        cv2.rectangle(frame, (10, 10), (250, 45), (0, 0, 0), -1)
        cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        out.write(frame)
    
    out.release()
    
    # Convert to H.264
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', str(temp_path),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-r', '15',
            str(final_path)
        ], capture_output=True, timeout=180, check=True)
        
        temp_path.unlink()
        
    except Exception as e:
        print(f"[Clip] FFmpeg error: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return None, None
    
    # Save thumbnail
    thumb_filename = f"{camera.camera_id}_{event_type}_{timestamp}.jpg"
    thumb_path = thumb_dir / thumb_filename
    
    thumb_frame = frames[-1].copy()
    label = f"{event_type.upper()}"
    color = {'cash': (0, 255, 0), 'violence': (0, 0, 255), 'fire': (0, 165, 255)}.get(event_type, (255, 255, 255))
    cv2.rectangle(thumb_frame, (10, 10), (150, 45), (0, 0, 0), -1)
    cv2.putText(thumb_frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imwrite(str(thumb_path), thumb_frame)
    
    return f'/media/clips/{final_filename}', f'/media/thumbnails/{thumb_filename}'


def _save_event(camera, event_type, confidence, frame_number, bbox, clip_path, thumbnail_path,
                gemini_validated=True, gemini_confidence=1.0, gemini_reason=""):
    """Save event to database with Gemini validation metadata"""
    try:
        import json
        from datetime import datetime
        
        # Save JSON metadata file
        json_dir = Path(settings.MEDIA_ROOT) / 'json'
        json_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now()
        json_filename = f"{event_type}_{camera.camera_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        json_path = json_dir / json_filename
        
        metadata = {
            'timestamp': timestamp.isoformat(),
            'event_type': event_type,
            'camera_id': camera.camera_id,
            'camera_name': camera.name,
            'confidence': round(confidence, 3),
            'frame_number': frame_number,
            'bbox': list(bbox) if bbox else None,
            'clip_path': clip_path,
            'thumbnail_path': thumbnail_path,
            'gemini_validation': {
                'validated': gemini_validated,
                'confidence': round(gemini_confidence, 3),
                'reason': gemini_reason
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        json_relative_path = f"json/{json_filename}"
        
        Event.objects.create(
            branch=camera.branch,
            camera=camera,
            event_type=event_type,
            confidence=confidence,
            frame_number=frame_number,
            bbox_x1=bbox[0] if bbox else 0,
            bbox_y1=bbox[1] if bbox else 0,
            bbox_x2=bbox[2] if bbox else 0,
            bbox_y2=bbox[3] if bbox else 0,
            clip_path=clip_path,
            thumbnail_path=thumbnail_path,
            metadata=json_relative_path,
        )
        print(f"[DB] Event saved with Gemini validation: {event_type} - {gemini_reason}")
    except Exception as e:
        print(f"[DB] Error saving event: {e}")
