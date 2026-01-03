#!/usr/bin/env python3
"""
Test script for Gemini video validation
Tests the GeminiValidator with a video file to check violence detection.

Usage:
    python test_gemini_video.py
"""

import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hotel_cctv.settings')
django.setup()

from detectors.gemini_validator import GeminiValidator
from cctv.models import Camera
import cv2


def test_video_validation():
    """Test Gemini validation on video file"""
    
    # Configuration
    video_path = "CAM-001_violence_20260103_231159.mp4"
    event_type = "violence"  # Can be: cash, violence, fire
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    print(f"{'='*70}")
    print(f"Testing Gemini Video Validation")
    print(f"{'='*70}")
    print(f"Video File: {video_path}")
    print(f"Event Type: {event_type}")
    print(f"{'='*70}\n")
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"📹 Video Information:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Frames: {frame_count}")
    print(f"   Duration: {duration:.2f}s")
    print(f"")
    
    # Get API key from environment or settings
    try:
        from django.conf import settings
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
    except:
        api_key = None
    
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY', '')
    
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in settings or environment")
        print("   Please set GEMINI_API_KEY in your environment or hotel_cctv/settings.py")
        return
    
    print(f"🔑 API Key: {api_key[:10]}...{api_key[-4:]}\n")
    
    # Try to get camera for logging (optional)
    camera_id = None
    try:
        camera = Camera.objects.first()
        if camera:
            camera_id = camera.id
            print(f"📷 Camera: {camera.name} (ID: {camera_id})")
        else:
            print(f"⚠️  No cameras found in database (validation won't be logged)")
    except Exception as e:
        print(f"⚠️  Cannot access database: {e}")
    
    print(f"")
    
    # Initialize validator
    print(f"🤖 Initializing Gemini Validator...")
    validator = GeminiValidator(api_key=api_key, camera_id=camera_id)
    
    if not validator.enabled:
        print("❌ Error: Validator not enabled")
        return
    
    print(f"✅ Validator initialized with model: {validator.MODEL_NAME}\n")
    
    # Run validation
    print(f"🔄 Sending video to Gemini API...")
    print(f"   This may take 10-30 seconds...\n")
    
    is_valid, confidence, reason, corrected_event_type = validator.validate_event_video(
        video_path=video_path,
        event_type=event_type
    )
    
    # Display results
    print(f"{'='*70}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"")
    print(f"Original Event Type:   {event_type}")
    print(f"Corrected Event Type:  {corrected_event_type}")
    print(f"")
    
    status_symbol = "✅" if is_valid else "❌"
    print(f"Is Valid:              {status_symbol} {is_valid}")
    print(f"Confidence:            {confidence:.2%}")
    print(f"Reason:                {reason}")
    print(f"")
    
    # Show last validation details
    if validator.last_validation_log:
        log = validator.last_validation_log
        print(f"Processing Time:       {log.get('processing_time_ms', 0)}ms")
        print(f"")
        print(f"Full Response:")
        print(f"{'-'*70}")
        import json
        print(json.dumps(log.get('response', {}), indent=2))
        print(f"{'-'*70}")
    
    print(f"")
    
    # Check if logged to database
    if camera_id:
        try:
            from cctv.models import GeminiLog
            latest_log = GeminiLog.objects.filter(
                camera_id=camera_id,
                validation_type='video'
            ).order_by('-timestamp').first()
            
            if latest_log:
                print(f"✅ Validation logged to database:")
                print(f"   Log ID: {latest_log.id}")
                print(f"   Camera: {latest_log.camera.name}")
                print(f"   Event: {latest_log.event_type}")
                print(f"   Valid: {latest_log.is_valid}")
                print(f"   Type: {latest_log.validation_type}")
                print(f"   Video: {latest_log.video_path}")
        except Exception as e:
            print(f"⚠️  Could not check database log: {e}")
    else:
        print(f"ℹ️  Validation not logged to database (no camera_id)")
    
    print(f"")
    print(f"{'='*70}")
    print(f"Test completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    try:
        test_video_validation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
