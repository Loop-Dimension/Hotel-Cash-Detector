#!/usr/bin/env python
"""
Test script for Worker and Streaming functionality
Tests RTSP connection, frame reading, detection, and JSON metadata output.

Run with: python test_worker_streaming.py
"""
import os
import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hotel_cctv.settings')
import django
django.setup()

from django.conf import settings
from cctv.models import Camera, Event


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_pass(msg):
    print(f"  {Colors.GREEN}✅ PASS:{Colors.RESET} {msg}")


def print_fail(msg):
    print(f"  {Colors.RED}❌ FAIL:{Colors.RESET} {msg}")


def print_warn(msg):
    print(f"  {Colors.YELLOW}⚠️ WARN:{Colors.RESET} {msg}")


def print_info(msg):
    print(f"  {Colors.CYAN}ℹ️ INFO:{Colors.RESET} {msg}")


def test_camera_model():
    """Test 1: Check Camera model exists and has RTSP URL"""
    print_header("Test 1: Camera Model")
    
    cameras = Camera.objects.all()
    if cameras.count() == 0:
        print_fail("No cameras found in database")
        return None
    
    print_pass(f"Found {cameras.count()} camera(s) in database")
    
    camera = cameras.first()
    print_info(f"Camera ID: {camera.camera_id}")
    print_info(f"Camera Name: {camera.name}")
    print_info(f"RTSP URL: {camera.rtsp_url[:50]}..." if len(camera.rtsp_url) > 50 else f"RTSP URL: {camera.rtsp_url}")
    print_info(f"Status: {camera.status}")
    print_info(f"Hand Touch Distance: {camera.hand_touch_distance}px")
    print_info(f"Cash Detection: {'Enabled' if camera.detect_cash else 'Disabled'}")
    print_info(f"Violence Detection: {'Enabled' if camera.detect_violence else 'Disabled'}")
    print_info(f"Fire Detection: {'Enabled' if camera.detect_fire else 'Disabled'}")
    
    zone = camera.get_cashier_zone()
    print_info(f"Cashier Zone: x={zone['x']}, y={zone['y']}, w={zone['width']}, h={zone['height']}")
    
    return camera


def test_rtsp_connection(camera):
    """Test 2: Test RTSP stream connection"""
    print_header("Test 2: RTSP Stream Connection")
    
    if not camera:
        print_fail("No camera to test")
        return False, None
    
    # Set FFmpeg options
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;60000000|max_delay;1000000|fflags;nobuffer+discardcorrupt|analyzeduration;2000000|probesize;2000000|buffer_size;4096000'
    
    print_info(f"Connecting to: {camera.rtsp_url}")
    
    start_time = time.time()
    cap = cv2.VideoCapture(camera.rtsp_url, cv2.CAP_FFMPEG)
    
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
    
    if not cap.isOpened():
        print_fail(f"Cannot open RTSP stream (took {time.time() - start_time:.1f}s)")
        return False, None
    
    connect_time = time.time() - start_time
    print_pass(f"Stream opened in {connect_time:.1f}s")
    
    # Get stream properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print_info(f"Stream FPS: {fps}")
    print_info(f"Resolution: {width}x{height}")
    
    return True, cap


def test_frame_reading(cap):
    """Test 3: Test reading frames from stream"""
    print_header("Test 3: Frame Reading")
    
    if cap is None:
        print_fail("No capture object")
        return False, None
    
    frames_read = 0
    frames_to_read = 30
    failed_reads = 0
    start_time = time.time()
    last_frame = None
    
    print_info(f"Reading {frames_to_read} frames...")
    
    for i in range(frames_to_read):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames_read += 1
            last_frame = frame
        else:
            failed_reads += 1
    
    elapsed = time.time() - start_time
    actual_fps = frames_read / elapsed if elapsed > 0 else 0
    
    if frames_read == frames_to_read:
        print_pass(f"Read all {frames_read}/{frames_to_read} frames successfully")
    elif frames_read > 0:
        print_warn(f"Read {frames_read}/{frames_to_read} frames ({failed_reads} failed)")
    else:
        print_fail("Could not read any frames")
        return False, None
    
    print_info(f"Actual FPS: {actual_fps:.1f}")
    print_info(f"Frame shape: {last_frame.shape if last_frame is not None else 'N/A'}")
    
    return True, last_frame


def test_detector_initialization():
    """Test 4: Test detector initialization"""
    print_header("Test 4: Detector Initialization")
    
    try:
        from detectors import UnifiedDetector
        print_pass("UnifiedDetector imported successfully")
    except ImportError as e:
        print_fail(f"Cannot import UnifiedDetector: {e}")
        return None
    
    models_dir = settings.BASE_DIR / 'models'
    print_info(f"Models directory: {models_dir}")
    
    # Check model files
    pose_model = models_dir / 'yolov8s-pose.pt'
    yolo_model = models_dir / 'yolov8s.pt'
    fire_model = models_dir / 'fire_smoke_yolov8.pt'
    
    if pose_model.exists():
        print_pass(f"Pose model found: {pose_model.name} ({pose_model.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print_fail(f"Pose model not found: {pose_model}")
    
    if yolo_model.exists():
        print_pass(f"YOLO model found: {yolo_model.name} ({yolo_model.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print_fail(f"YOLO model not found: {yolo_model}")
    
    if fire_model.exists():
        print_pass(f"Fire model found: {fire_model.name} ({fire_model.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print_warn(f"Fire model not found: {fire_model}")
    
    # Initialize detector
    config = {
        'models_dir': str(models_dir),
        'pose_model': 'yolov8s-pose.pt',
        'yolo_model': 'yolov8s.pt',
        'fire_model': 'fire_smoke_yolov8.pt',
        'cashier_zone': [0, 0, 640, 480],
        'hand_touch_distance': 100,
        'pose_confidence': 0.5,
        'detect_cash': True,
        'detect_violence': True,
        'detect_fire': True,
    }
    
    try:
        detector = UnifiedDetector(config)
        print_pass("UnifiedDetector initialized successfully")
        return detector
    except Exception as e:
        print_fail(f"Failed to initialize detector: {e}")
        return None


def test_detection(detector, frame):
    """Test 5: Test detection on a frame"""
    print_header("Test 5: Detection Processing")
    
    if detector is None:
        print_fail("No detector available")
        return None
    
    if frame is None:
        print_fail("No frame available")
        return None
    
    print_info(f"Processing frame of shape: {frame.shape}")
    
    start_time = time.time()
    try:
        result = detector.process_frame(frame.copy(), draw_overlay=True)
        elapsed = time.time() - start_time
        
        print_pass(f"Frame processed in {elapsed*1000:.0f}ms")
        
        if 'frame' in result:
            print_pass("Overlay frame generated")
        
        detections = result.get('detections', [])
        print_info(f"Detections found: {len(detections)}")
        
        for det in detections:
            print_info(f"  - {det.get('label')}: confidence={det.get('confidence', 0):.2f}")
            metadata = det.get('metadata', {})
            if metadata:
                print_info(f"    Metadata keys: {list(metadata.keys())}")
        
        return result
        
    except Exception as e:
        print_fail(f"Detection error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_cash_detector_metadata(detector, frame):
    """Test 6: Test cash detector metadata output"""
    print_header("Test 6: Cash Detector Metadata")
    
    if detector is None or not hasattr(detector, 'cash_detector'):
        print_fail("Cash detector not available")
        return
    
    cd = detector.cash_detector
    print_pass("Cash detector available")
    
    print_info(f"Hand touch distance threshold: {cd.hand_touch_distance}px")
    print_info(f"Pose confidence threshold: {cd.pose_confidence}")
    print_info(f"Cashier zone: {cd.cashier_zone}")
    print_info(f"Min transaction frames: {cd.min_transaction_frames}")
    
    # Check metadata structure expected in detections
    expected_metadata_keys = [
        'type', 'distance', 'distance_threshold', 'hand_confidence',
        'people_count', 'cashier_zone', 'cashier', 'customer', 'interaction_point'
    ]
    print_info(f"Expected metadata keys for cash detection:")
    for key in expected_metadata_keys:
        print_info(f"  - {key}")


def test_json_output():
    """Test 7: Test JSON file output"""
    print_header("Test 7: JSON Output Structure")
    
    json_dir = Path(settings.MEDIA_ROOT) / 'json'
    
    if not json_dir.exists():
        print_warn(f"JSON directory does not exist: {json_dir}")
        print_info("Creating JSON directory...")
        json_dir.mkdir(parents=True, exist_ok=True)
        print_pass("JSON directory created")
    else:
        print_pass(f"JSON directory exists: {json_dir}")
    
    # Check for existing JSON files
    json_files = list(json_dir.glob('*.json'))
    print_info(f"Found {len(json_files)} existing JSON files")
    
    if json_files:
        # Show latest JSON file content
        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        print_info(f"Latest JSON file: {latest.name}")
        
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print_pass("JSON file is valid")
            print_info("JSON structure:")
            for key in data.keys():
                value = data[key]
                if isinstance(value, dict):
                    print_info(f"  {key}: {{...}} ({len(value)} keys)")
                elif isinstance(value, list):
                    print_info(f"  {key}: [...] ({len(value)} items)")
                else:
                    val_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    print_info(f"  {key}: {val_str}")
            
            # Check for cashier/customer metadata
            if 'cashier' in data:
                print_pass("Cashier position included in metadata")
                print_info(f"  Cashier data: {json.dumps(data['cashier'], indent=4)}")
            else:
                print_warn("Cashier position not in metadata")
            
            if 'customer' in data:
                print_pass("Customer position included in metadata")
                print_info(f"  Customer data: {json.dumps(data['customer'], indent=4)}")
            else:
                print_warn("Customer position not in metadata")
            
            if 'measured_hand_distance' in data:
                print_pass(f"Hand distance included: {data['measured_hand_distance']}px")
            elif 'distance' in data:
                print_pass(f"Hand distance included: {data['distance']}px")
            else:
                print_warn("Hand distance not in metadata")
                
        except json.JSONDecodeError as e:
            print_fail(f"JSON parse error: {e}")
        except Exception as e:
            print_fail(f"Error reading JSON: {e}")
    else:
        print_info("No JSON files yet - they will be created when events are detected")
        
        # Create a sample JSON to show expected structure
        sample_json = {
            "timestamp": datetime.now().isoformat(),
            "frame_number": 1234,
            "confidence": 0.85,
            "bbox": [100, 100, 200, 200],
            "camera_id": "CAM-001",
            "camera_name": "Test Camera",
            "event_type": "cash",
            "clip_path": "/media/clips/cash_CAM-001_20251205_160000.mp4",
            "thumbnail_path": "/media/thumbnails/cash_CAM-001_20251205_160000.jpg",
            "cash_detection": {
                "hand_touch_distance_threshold": 100,
                "cashier_zone": [0, 0, 640, 480],
                "pose_confidence": 0.5
            },
            "cashier": {
                "center": [320, 400],
                "bbox": [250, 300, 400, 550],
                "hands": {
                    "left": [280, 380, 0.9],
                    "right": [350, 390, 0.85]
                },
                "in_zone": True,
                "hand_used": "right"
            },
            "customer": {
                "center": [500, 420],
                "bbox": [430, 320, 580, 560],
                "hands": {
                    "left": [450, 400, 0.88],
                    "right": [520, 410, 0.92]
                },
                "in_zone": False,
                "hand_used": "left"
            },
            "measured_hand_distance": 85.5,
            "interaction_point": [400, 395],
            "trigger_time": datetime.now().isoformat(),
            "frames_saved": 450,
            "duration_sec": 30.0
        }
        
        print_info("Expected JSON structure for cash events:")
        print(json.dumps(sample_json, indent=2))


def test_event_model():
    """Test 8: Test Event model and recent events"""
    print_header("Test 8: Event Model")
    
    total_events = Event.objects.count()
    print_info(f"Total events in database: {total_events}")
    
    # Recent events
    recent_events = Event.objects.order_by('-created_at')[:5]
    
    if recent_events:
        print_info("Recent events:")
        for event in recent_events:
            print_info(f"  - {event.event_type} @ {event.created_at.strftime('%Y-%m-%d %H:%M:%S')} (conf: {event.confidence:.2f})")
            if event.metadata:
                print_info(f"    Metadata path: {event.metadata}")
    else:
        print_info("No events recorded yet")
    
    # Event type counts
    from django.db.models import Count
    type_counts = Event.objects.values('event_type').annotate(count=Count('id'))
    if type_counts:
        print_info("Event counts by type:")
        for tc in type_counts:
            print_info(f"  - {tc['event_type']}: {tc['count']}")


def run_all_tests():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     WORKER & STREAMING TEST SUITE                         ║")
    print("║     Hotel Cash Detector System                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    
    start_time = time.time()
    results = {}
    
    # Test 1: Camera Model
    camera = test_camera_model()
    results['camera_model'] = camera is not None
    
    # Test 2: RTSP Connection
    connected, cap = test_rtsp_connection(camera)
    results['rtsp_connection'] = connected
    
    # Test 3: Frame Reading
    if cap:
        frame_ok, frame = test_frame_reading(cap)
        results['frame_reading'] = frame_ok
    else:
        results['frame_reading'] = False
        frame = None
    
    # Test 4: Detector Initialization
    detector = test_detector_initialization()
    results['detector_init'] = detector is not None
    
    # Test 5: Detection Processing
    if detector and frame is not None:
        detection_result = test_detection(detector, frame)
        results['detection'] = detection_result is not None
    else:
        results['detection'] = False
    
    # Test 6: Cash Detector Metadata
    test_cash_detector_metadata(detector, frame)
    results['cash_metadata'] = True  # Info only
    
    # Test 7: JSON Output
    test_json_output()
    results['json_output'] = True  # Info only
    
    # Test 8: Event Model
    test_event_model()
    results['event_model'] = True  # Info only
    
    # Cleanup
    if cap:
        cap.release()
    
    # Summary
    elapsed = time.time() - start_time
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        if passed_test:
            print_pass(f"{test_name}")
        else:
            print_fail(f"{test_name}")
    
    print(f"\n  {Colors.BOLD}Results: {passed}/{total} tests passed{Colors.RESET}")
    print(f"  {Colors.BOLD}Time: {elapsed:.1f}s{Colors.RESET}\n")
    
    if passed == total:
        print(f"  {Colors.GREEN}{Colors.BOLD}✅ ALL TESTS PASSED!{Colors.RESET}\n")
        return 0
    else:
        print(f"  {Colors.YELLOW}{Colors.BOLD}⚠️ SOME TESTS FAILED{Colors.RESET}\n")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
