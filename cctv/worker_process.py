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

# Import shared utilities
from cctv.utils import (
    create_rtsp_capture,
    save_clip,
    save_validation_clip,
    upload_validation_clip_to_s3,
    save_event,
    validate_detection,
    DEFAULT_EVENT_COOLDOWN,
)

# Import detectors
try:
    from detectors import UnifiedDetector
    DETECTOR_AVAILABLE = True
except ImportError as e:
    DETECTOR_AVAILABLE = False
    print(f"Warning: Detectors not available - {e}")

# Import ML client for microservice mode
try:
    from cctv.ml_client import MLDetectorProxy, MLServiceClient
    ML_CLIENT_AVAILABLE = True
except ImportError as e:
    ML_CLIENT_AVAILABLE = False
    print(f"Warning: ML client not available - {e}")

# Import Gemini validator
try:
    from detectors.gemini_validator import GeminiValidator
    GEMINI_AVAILABLE = True
except ImportError as e:
    GEMINI_AVAILABLE = False
    print(f"Warning: Gemini validator not available - {e}")

# Global manager for all workers (create once, reuse)
_manager = None


def safe_db_operation(operation, max_retries=3):
    """Execute a database operation with retry logic for connection issues.
    
    Args:
        operation: Callable that performs the DB operation
        max_retries: Maximum number of retry attempts
        
    Returns:
        Result of the operation or None on failure
    """
    from django.db import connection
    from django.db.utils import OperationalError
    import time
    
    for attempt in range(max_retries):
        try:
            return operation()
        except OperationalError as e:
            if 'SSL connection' in str(e) or 'closed unexpectedly' in str(e):
                print(f"[Worker] DB connection error (attempt {attempt + 1}/{max_retries}): {e}")
                # Close stale connection and let Django create a new one
                connection.close()
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
            raise
        except Exception as e:
            print(f"[Worker] Unexpected DB error: {e}")
            raise
    
    return None

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
    
    # Get polygon zones
    cashier_polygon = camera.get_cashier_zone_polygon_points()
    cash_drawer_polygon = camera.get_cash_drawer_zone_polygon_points() if hasattr(camera, 'get_cash_drawer_zone_polygon_points') else None
    
    # Debug: Show what polygon data was loaded from database
    import sys
    print(f"\n{'='*60}", flush=True)
    print(f"[Worker-{camera_id}] POLYGON DEBUG INFO", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Camera ID: {camera_id}", flush=True)
    print(f"Camera Name: {camera.name}", flush=True)
    print(f"cashier_zone_polygon from DB: {camera.cashier_zone_polygon}", flush=True)
    print(f"Parsed polygon points: {cashier_polygon}", flush=True)
    if cashier_polygon:
        print(f"Number of points: {len(cashier_polygon)}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    detector_config = {
        'models_dir': str(settings.BASE_DIR / 'models'),
        # GPU/CPU setting from environment
        'use_gpu': settings.DETECTION_CONFIG.get('USE_GPU', 'auto'),
        # Model selection - cash uses pose, violence/fire use nano
        'cash_pose_model': settings.DETECTION_CONFIG.get('CASH_POSE_MODEL', 'yolov8s-pose.pt'),
        'violence_pose_model': settings.DETECTION_CONFIG.get('VIOLENCE_POSE_MODEL', 'yolov8n-pose.pt'),
        'fire_yolo_model': settings.DETECTION_CONFIG.get('FIRE_YOLO_MODEL', 'yolov8n.pt'),
        'fire_model': settings.DETECTION_CONFIG.get('FIRE_MODEL', 'fire_smoke_yolov8.pt'),
        'cashier_zone_polygon': cashier_polygon,
        'cash_drawer_zone_polygon': cash_drawer_polygon,
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

    # Check if ML service mode is enabled
    use_ml_service = getattr(settings, 'USE_ML_SERVICE', False)

    if use_ml_service and ML_CLIENT_AVAILABLE:
        # MLDetectorProxy handles fallback internally:
        # - Tries ML service first
        # - Falls back to local detector if ML service is down
        # - Auto-reconnects to ML service when it comes back
        # - Prints [MODE: ML_SERVICE] or [MODE: LOCAL] on every mode change
        detector = MLDetectorProxy(config=detector_config, camera_id=camera_id)
        print(f"[Worker-{camera_id}] Detector initialized (ML service mode enabled, current: {detector._get_mode_label()})")
    else:
        detector = UnifiedDetector(detector_config)
        print(f"[Worker-{camera_id}] [MODE: LOCAL] Using local detector (ML service disabled)")
    
    # Connect to RTSP stream (using shared utility)
    shared_state['status'] = 'connecting'
    cap = create_rtsp_capture(camera.rtsp_url)
    
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
        cap = create_rtsp_capture(camera.rtsp_url)
    
    if not cap.isOpened():
        shared_state['error'] = 'Cannot connect to stream'
        shared_state['status'] = 'error'
        camera.status = 'offline'
        safe_db_operation(lambda: camera.save())
        return
    
    # Update camera status
    camera.status = 'online'
    safe_db_operation(lambda: camera.save())
    shared_state['status'] = 'running'
    
    # Frame buffer for clips
    frame_buffer = []
    buffer_size = 150  # 10 seconds at 15fps
    
    # Separate buffer for Gemini validation (3 seconds)
    gemini_buffer = []
    gemini_buffer_size = 45  # 3 seconds at 15fps
    
    # Separate buffer for Gemini validation (3 seconds)
    gemini_buffer = []
    gemini_buffer_size = 45  # 3 seconds at 15fps
    
    # Event cooldown tracking
    last_event_time = {}
    event_cooldown = 60  # seconds between same event type (1 minute)
    
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
                cap = create_rtsp_capture(camera.rtsp_url)
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
            
            # Also add to Gemini buffer (last 3 seconds)
            gemini_buffer.append(frame.copy())
            if len(gemini_buffer) > gemini_buffer_size:
                gemini_buffer.pop(0)
        
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
                                    from cctv.models import GeminiPrompts
                                    import os
                                    
                                    # Check if video validation is enabled (default: False, use image)
                                    use_video_validation = os.getenv('GEMINI_USE_VIDEO_VALIDATION', 'False').lower() == 'true'
                                    
                                    # Create validator with camera_id for logging and global prompts
                                    validator = GeminiValidator(api_key=gemini_api_key, camera_id=camera.id)
                                    
                                    # Get global prompts from GeminiPrompts model
                                    global_prompts = GeminiPrompts.get_prompts()
                                    if global_prompts.get('unified'):
                                        validator.set_custom_prompts({'unified': global_prompts['unified']})
                                    
                                    # Choose validation method based on environment variable
                                    if use_video_validation:
                                        # VIDEO MODE: Create 3-second validation video (local temp path)
                                        local_video_path = save_validation_clip(
                                            gemini_buffer[-45:] if len(gemini_buffer) >= 45 else gemini_buffer,
                                            camera, event_type
                                        )
                                        
                                        if local_video_path:
                                            # Validate with Gemini (reads local file)
                                            gemini_validated, gemini_confidence, gemini_reason, corrected_event_type = validator.validate_event_video(local_video_path, event_type)
                                            print(f"[Worker-{camera_id}] Gemini VIDEO validation: {event_type} = {gemini_validated}")
                                            
                                            # After Gemini reads it, upload to S3 and log to database
                                            final_video_url = upload_validation_clip_to_s3(local_video_path)
                                            if hasattr(validator, 'last_validation_log') and validator.last_validation_log:
                                                log_data = validator.last_validation_log
                                                from cctv.utils import _log_video_validation
                                                _log_video_validation(
                                                    camera_id=camera.id,
                                                    event_type=event_type,
                                                    is_valid=gemini_validated,
                                                    confidence=gemini_confidence,
                                                    reason=gemini_reason,
                                                    prompt=log_data.get('prompt', ''),
                                                    response_raw=log_data.get('response_raw', ''),
                                                    video_url=final_video_url,
                                                    processing_time_ms=log_data.get('processing_time_ms', 0)
                                                )
                                        else:
                                            # Fallback to image if video creation failed
                                            gemini_validated, gemini_confidence, gemini_reason, corrected_event_type = validator.validate_event(frame, event_type)
                                            print(f"[Worker-{camera_id}] Gemini IMAGE validation (fallback): {event_type} = {gemini_validated}")
                                    else:
                                        # IMAGE MODE: Use single frame validation (default)
                                        gemini_validated, gemini_confidence, gemini_reason, corrected_event_type = validator.validate_event(frame, event_type)
                                        print(f"[Worker-{camera_id}] Gemini IMAGE validation: {event_type} = {gemini_validated}")
                                    
                                    print(f"[Worker-{camera_id}] Result: corrected_type={corrected_event_type}, reason={gemini_reason}")
                                    
                                    # Use corrected event type if Gemini detected something different
                                    if corrected_event_type != event_type:
                                        print(f"[Worker-{camera_id}] Event type corrected: {event_type} → {corrected_event_type}")
                                        event_type = corrected_event_type
                                    
                                    if not gemini_validated:
                                        print(f"[Worker-{camera_id}] Event rejected by Gemini: {event_type} - {gemini_reason}")
                                        continue  # Skip saving this event
                            except Exception as e:
                                print(f"[Worker-{camera_id}] Gemini validation error: {e}")
                                import traceback
                                traceback.print_exc()
                                # On error, allow the event (don't block on validation errors)
                        
                        # Save event with clip using shared utilities
                        clip_path, thumb_path = save_clip(
                            frame_buffer[-150:] if len(frame_buffer) >= 150 else frame_buffer,
                            camera, event_type, fps=15
                        )
                        
                        if clip_path:
                            save_event(
                                camera, event_type, confidence, frame_count, bbox, 
                                clip_path, thumb_path,
                                gemini_validated=gemini_validated, 
                                gemini_confidence=gemini_confidence, 
                                gemini_reason=gemini_reason
                            )
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
    safe_db_operation(lambda: camera.save())
    print(f"[Worker-{camera_id}] Loop ended")


# ============================================================================
# DEPRECATED: These functions have been moved to cctv/utils.py
# The imports at the top of this file now use the shared utilities:
#   - create_rtsp_capture (was _create_rtsp_capture)
#   - save_clip (was _save_clip)  
#   - save_validation_clip (was _save_gemini_validation_clip)
#   - save_event (was _save_event)
# ============================================================================
