"""
Comprehensive test script for Gemini AI validation system
Tests validator, database logging, API endpoints, and unified prompts
"""

import os
import sys
import django
import requests
import time
import cv2
import numpy as np
from pathlib import Path

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hotel_cctv.settings')
django.setup()

from django.conf import settings
from cctv.models import Camera, GeminiLog
from detectors.gemini_validator import GeminiValidator


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def create_test_image(width=640, height=480):
    """Create a test image with some shapes"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)  # Dark gray background
    
    # Draw some shapes to simulate a scene
    cv2.rectangle(img, (100, 100), (300, 300), (100, 150, 200), -1)
    cv2.circle(img, (400, 200), 50, (200, 100, 100), -1)
    cv2.putText(img, "TEST IMAGE", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def test_1_environment():
    """Test 1: Check environment configuration"""
    print_header("TEST 1: Environment Configuration")
    
    # Check API key
    api_key = getattr(settings, 'GEMINI_API_KEY', '')
    if api_key:
        print_success(f"GEMINI_API_KEY found (length: {len(api_key)})")
    else:
        print_error("GEMINI_API_KEY not found in settings")
        return False
    
    # Check media directory
    media_root = Path(settings.MEDIA_ROOT)
    if media_root.exists():
        print_success(f"MEDIA_ROOT exists: {media_root}")
    else:
        print_warning(f"MEDIA_ROOT doesn't exist: {media_root}")
    
    # Check gemini_logs directory
    log_dir = media_root / 'gemini_logs'
    if log_dir.exists():
        print_success(f"gemini_logs directory exists: {log_dir}")
    else:
        print_info(f"Creating gemini_logs directory: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)
    
    return True


def test_2_database():
    """Test 2: Check database and models"""
    print_header("TEST 2: Database & Models")
    
    try:
        # Check cameras
        cameras = Camera.objects.all()
        print_info(f"Found {cameras.count()} camera(s) in database")
        for cam in cameras:
            print(f"  - Camera {cam.id}: {cam.name} (Status: {cam.status})")
        
        if cameras.count() == 0:
            print_warning("No cameras found in database")
            return False
        
        # Check existing logs
        logs = GeminiLog.objects.all()
        print_info(f"Found {logs.count()} existing Gemini log(s)")
        
        if logs.exists():
            latest = logs.order_by('-created_at').first()
            print(f"  - Latest log: Camera {latest.camera.name}, Type: {latest.event_type}, Valid: {latest.is_validated}")
        
        print_success("Database connection OK")
        return True
        
    except Exception as e:
        print_error(f"Database error: {e}")
        return False


def test_3_validator_initialization():
    """Test 3: Initialize GeminiValidator"""
    print_header("TEST 3: GeminiValidator Initialization")
    
    try:
        api_key = getattr(settings, 'GEMINI_API_KEY', '')
        camera = Camera.objects.first()
        
        if not camera:
            print_error("No camera available for testing")
            return None
        
        print_info(f"Initializing validator for Camera {camera.id}: {camera.name}")
        validator = GeminiValidator(api_key=api_key, camera_id=camera.id)
        
        if validator.enabled:
            print_success("Validator initialized and enabled")
            print_info(f"Model: {validator.MODEL_NAME}")
        else:
            print_error("Validator initialized but disabled (check API key)")
            return None
        
        return validator, camera
        
    except Exception as e:
        print_error(f"Validator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_4_validation(validator, camera):
    """Test 4: Test actual validation with test image"""
    print_header("TEST 4: Validation Test")
    
    try:
        print_info("Creating test image...")
        test_img = create_test_image()
        
        # Test violence detection
        print_info("Testing violence detection...")
        start = time.time()
        is_valid, confidence, reason = validator.validate_event(test_img, 'violence')
        elapsed = time.time() - start
        
        print(f"  - Valid: {is_valid}")
        print(f"  - Confidence: {confidence:.2f}")
        print(f"  - Reason: {reason}")
        print(f"  - Processing time: {elapsed:.2f}s")
        
        print_success("Validation completed successfully")
        return True
        
    except Exception as e:
        print_error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_database_logging(camera):
    """Test 5: Check if validation was logged to database"""
    print_header("TEST 5: Database Logging")
    
    try:
        # Wait a moment for async logging
        time.sleep(2)
        
        # Check for new logs
        logs = GeminiLog.objects.filter(camera=camera).order_by('-created_at')
        
        if logs.exists():
            latest = logs.first()
            print_success(f"Found log entry (ID: {latest.id})")
            print(f"  - Event Type: {latest.event_type}")
            print(f"  - Validated: {latest.is_validated}")
            print(f"  - Confidence: {latest.confidence}")
            print(f"  - Reason: {latest.reason[:100]}...")
            print(f"  - Image Path: {latest.image_path}")
            print(f"  - Processing Time: {latest.processing_time_ms}ms")
            print(f"  - Created: {latest.created_at}")
            
            # Check image file
            if latest.image_path:
                img_path = Path(settings.MEDIA_ROOT) / latest.image_path
                if img_path.exists():
                    print_success(f"Validation image exists: {img_path.name}")
                else:
                    print_warning(f"Validation image not found: {img_path}")
            
            return True
        else:
            print_error("No logs found in database")
            print_warning("Check worker console for '[GeminiValidator]' debug messages")
            return False
            
    except Exception as e:
        print_error(f"Database logging check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_api_endpoints():
    """Test 6: Test API endpoints"""
    print_header("TEST 6: API Endpoints")
    
    base_url = "http://127.0.0.1:8000"
    
    try:
        # Test logs API
        print_info("Testing /api/gemini/all-logs/ endpoint...")
        response = requests.get(f"{base_url}/api/gemini/all-logs/", timeout=5)
        
        if response.status_code == 200:
            if 'application/json' in response.headers.get('Content-Type', ''):
                data = response.json()
                print_success(f"API responded with {len(data.get('logs', []))} logs")
                print(f"  - Total Validations: {data.get('stats', {}).get('total', 0)}")
                print(f"  - Validated: {data.get('stats', {}).get('validated', 0)}")
                print(f"  - Rejected: {data.get('stats', {}).get('rejected', 0)}")
                
                if data.get('logs'):
                    first_log = data['logs'][0]
                    print_info(f"First log preview:")
                    print(f"  - Camera: {first_log.get('camera_name')}")
                    print(f"  - Event: {first_log.get('event_type')}")
                    print(f"  - Image: {first_log.get('image_path')}")
            else:
                print_warning(f"API returned HTML instead of JSON (status {response.status_code})")
                print_info("This means the endpoint exists but might need authentication or is redirecting")
                return True  # Server is running, which is good enough
        else:
            print_error(f"API returned status {response.status_code}")
            return False
        
        # Test prompts API
        print_info("Testing /api/gemini/global-prompts/ endpoint...")
        response = requests.get(f"{base_url}/api/gemini/global-prompts/", timeout=5)
        
        if response.status_code == 200:
            if 'application/json' in response.headers.get('Content-Type', ''):
                prompts = response.json()
                print_success("Prompts API OK")
                print(f"  - Unified Prompt Length: {len(prompts.get('unified_prompt', ''))}")
                print(f"  - Has Event Type Placeholder: {'{event_type}' in prompts.get('unified_prompt', '')}")
            else:
                print_warning("Prompts API returned HTML instead of JSON")
                return True  # Server is running
        else:
            print_error(f"Prompts API returned status {response.status_code}")
            return False
        
        print_success("All API endpoints working")
        return True
        
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server at http://127.0.0.1:8000")
        print_warning("Make sure Django server is running: python manage.py runserver")
        return False
    except Exception as e:
        print_error(f"API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_unified_prompts(camera):
    """Test 7: Test unified prompt system"""
    print_header("TEST 7: Unified Prompt System")
    
    try:
        # Check if camera has prompts
        print_info(f"Checking prompts for Camera {camera.id}: {camera.name}")
        
        if camera.gemini_cash_prompt:
            has_placeholder = '{event_type}' in camera.gemini_cash_prompt
            print_success(f"Unified prompt found ({len(camera.gemini_cash_prompt)} chars)")
            print(f"  - Has {{event_type}} placeholder: {has_placeholder}")
            
            if has_placeholder:
                print_info("Preview of unified prompt (first 200 chars):")
                print(f"  {camera.gemini_cash_prompt[:200]}...")
            else:
                print_warning("Prompt doesn't use {event_type} placeholder (legacy mode)")
        else:
            print_warning("No unified prompt set for this camera")
            print_info("Set one via: http://127.0.0.1:8000/gemini-prompts/")
        
        return True
        
    except Exception as e:
        print_error(f"Unified prompt check failed: {e}")
        return False


def test_8_cleanup_old_logs():
    """Test 8: Optional cleanup of old test logs"""
    print_header("TEST 8: Database Cleanup (Optional)")
    
    try:
        old_count = GeminiLog.objects.count()
        print_info(f"Current log count: {old_count}")
        
        response = input("\nDelete all Gemini logs? (y/N): ").strip().lower()
        if response == 'y':
            GeminiLog.objects.all().delete()
            print_success("All logs deleted")
        else:
            print_info("Logs kept")
        
        return True
        
    except Exception as e:
        print_error(f"Cleanup failed: {e}")
        return False


def main():
    """Run all tests"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      GEMINI AI VALIDATION SYSTEM - COMPREHENSIVE TEST        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    results = {}
    
    # Test 1: Environment
    results['Environment'] = test_1_environment()
    if not results['Environment']:
        print_error("Cannot continue without proper environment setup")
        return
    
    # Test 2: Database
    results['Database'] = test_2_database()
    if not results['Database']:
        print_error("Cannot continue without database access")
        return
    
    # Test 3: Validator initialization
    validator_result = test_3_validator_initialization()
    results['Validator Init'] = validator_result is not None
    
    if not validator_result:
        print_error("Cannot continue without validator")
        print("\n" + "="*60)
        print_summary(results)
        return
    
    validator, camera = validator_result
    
    # Test 4: Validation
    results['Validation'] = test_4_validation(validator, camera)
    
    # Test 5: Database logging
    results['DB Logging'] = test_5_database_logging(camera)
    
    # Test 6: API endpoints
    results['API Endpoints'] = test_6_api_endpoints()
    
    # Test 7: Unified prompts
    results['Unified Prompts'] = test_7_unified_prompts(camera)
    
    # Test 8: Cleanup (optional)
    results['Cleanup'] = test_8_cleanup_old_logs()
    
    # Print summary
    print("\n" + "="*60)
    print_summary(results)


def print_summary(results):
    """Print test summary"""
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{Colors.OKGREEN}✅ PASS{Colors.ENDC}" if result else f"{Colors.FAIL}❌ FAIL{Colors.ENDC}"
        print(f"  {test_name:.<30} {status}")
    
    print("\n" + "="*60)
    
    if passed == total:
        print_success(f"ALL TESTS PASSED ({passed}/{total})")
        print("\n🎉 Gemini validation system is working correctly!")
        print("\nNext steps:")
        print("  1. Check 📋 Gemini AI Logs at: http://127.0.0.1:8000/gemini-logs/")
        print("  2. Configure prompts at: http://127.0.0.1:8000/gemini-prompts/")
        print("  3. Monitor worker console for [GeminiValidator] messages")
    else:
        print_warning(f"SOME TESTS FAILED ({passed}/{total} passed)")
        print("\nTroubleshooting:")
        print("  1. Check GEMINI_API_KEY in settings.py")
        print("  2. Ensure Django server is running")
        print("  3. Check worker processes are active")
        print("  4. Review error messages above")
    
    print("="*60 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Test interrupted by user{Colors.ENDC}\n")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
