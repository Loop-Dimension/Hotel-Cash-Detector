#!/usr/bin/env python3
"""
Test script for Gemini validation with new soft scoring prompt.
Tests both IMAGE and VIDEO validation with the new response format.

Usage:
    python test_gemini_new_prompt.py
    python test_gemini_new_prompt.py --image-only
    python test_gemini_new_prompt.py --video-only
"""

import os
import sys
import json
import argparse
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hotel_cctv.settings')
django.setup()

import cv2
from detectors.gemini_validator import GeminiValidator, DEFAULT_UNIFIED_PROMPT
from cctv.models import Camera


def print_separator(title=""):
    """Print a visual separator"""
    print(f"\n{'='*80}")
    if title:
        print(f"  {title}")
        print(f"{'='*80}")


def print_result(result: dict, validation_type: str):
    """Pretty print validation result"""
    print(f"\n{'─'*60}")
    print(f"📊 {validation_type.upper()} VALIDATION RESULT")
    print(f"{'─'*60}")
    
    is_valid, confidence, reason, corrected_type = result['parsed']
    
    print(f"✅ Is Valid:           {is_valid}")
    print(f"🎯 Confidence:         {confidence:.2%}")
    print(f"🔄 Corrected Type:     {corrected_type}")
    print(f"📝 Reason:             {reason[:200]}...")
    
    if result.get('response'):
        print(f"\n📋 Full Response:")
        print(json.dumps(result['response'], indent=2, ensure_ascii=False))
    
    if result.get('processing_time_ms'):
        print(f"\n⏱️  Processing Time:    {result['processing_time_ms']}ms")


def test_image_validation(image_path: str, event_type: str, validator: GeminiValidator):
    """Test image validation"""
    print_separator(f"Testing IMAGE: {os.path.basename(image_path)}")
    print(f"Event Type: {event_type}")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Error: Cannot read image file")
        return None
    
    print(f"📷 Image size: {frame.shape[1]}x{frame.shape[0]}")
    print(f"\n🔄 Sending image to Gemini API...")
    
    # Run validation
    is_valid, confidence, reason, corrected_type = validator.validate_event(
        frame=frame,
        event_type=event_type,
        save_image=False  # Don't save during test
    )
    
    result = {
        'parsed': (is_valid, confidence, reason, corrected_type),
        'response': validator.last_validation_log.get('response', {}),
        'processing_time_ms': validator.last_validation_log.get('processing_time_ms', 0)
    }
    
    print_result(result, "IMAGE")
    return result


def test_video_validation(video_path: str, event_type: str, validator: GeminiValidator):
    """Test video validation"""
    print_separator(f"Testing VIDEO: {os.path.basename(video_path)}")
    print(f"Event Type: {event_type}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return None
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video file")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"📹 Video: {width}x{height}, {fps:.1f}fps, {duration:.2f}s ({frame_count} frames)")
    print(f"\n🔄 Sending video to Gemini API (this may take 10-30 seconds)...")
    
    # Run validation
    is_valid, confidence, reason, corrected_type = validator.validate_event_video(
        video_path=video_path,
        event_type=event_type
    )
    
    result = {
        'parsed': (is_valid, confidence, reason, corrected_type),
        'response': validator.last_validation_log.get('response', {}),
        'processing_time_ms': validator.last_validation_log.get('processing_time_ms', 0)
    }
    
    print_result(result, "VIDEO")
    return result


def main():
    parser = argparse.ArgumentParser(description='Test Gemini validation with new prompt')
    parser.add_argument('--image-only', action='store_true', help='Test only image validation')
    parser.add_argument('--video-only', action='store_true', help='Test only video validation')
    parser.add_argument('--image', type=str, help='Custom image path')
    parser.add_argument('--video', type=str, help='Custom video path')
    parser.add_argument('--event-type', type=str, default=None, help='Event type to test')
    args = parser.parse_args()
    
    # Default test files
    default_image = "cash_23_20251229_010042_225504.jpg"
    default_video = "CAM-001_violence_20260103_231159.mp4"
    
    image_path = args.image or default_image
    video_path = args.video or default_video
    
    print_separator("GEMINI VALIDATION TEST - NEW SOFT SCORING PROMPT")
    
    # Show prompt preview
    print(f"\n📄 Prompt Preview (first 500 chars):")
    print(f"{'─'*60}")
    print(DEFAULT_UNIFIED_PROMPT[:500] + "...")
    
    # Get API key
    try:
        from django.conf import settings
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
    except:
        api_key = None
    
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY', '')
    
    if not api_key:
        print("\n❌ Error: GEMINI_API_KEY not found!")
        print("   Set it in .env file or as environment variable")
        return
    
    print(f"\n🔑 API Key: {api_key[:10]}...{api_key[-4:]}")
    
    # Get camera for logging (optional)
    camera_id = None
    try:
        camera = Camera.objects.first()
        if camera:
            camera_id = camera.id
            print(f"📷 Using Camera: {camera.name} (ID: {camera_id})")
    except Exception as e:
        print(f"⚠️  No camera found: {e}")
    
    # Initialize validator
    print(f"\n🤖 Initializing GeminiValidator...")
    validator = GeminiValidator(api_key=api_key, camera_id=camera_id)
    
    if not validator.enabled:
        print("❌ Error: Validator not enabled")
        return
    
    print(f"✅ Validator initialized with model: {validator.MODEL_NAME}")
    
    results = {}
    
    # Test IMAGE validation
    if not args.video_only:
        # Determine event type from filename or use provided
        if args.event_type:
            image_event_type = args.event_type
        elif 'cash' in image_path.lower():
            image_event_type = 'cash'
        elif 'violence' in image_path.lower():
            image_event_type = 'violence'
        elif 'fire' in image_path.lower():
            image_event_type = 'fire'
        else:
            image_event_type = 'cash'  # default
        
        results['image'] = test_image_validation(image_path, image_event_type, validator)
    
    # Test VIDEO validation
    if not args.image_only:
        # Determine event type from filename or use provided
        if args.event_type:
            video_event_type = args.event_type
        elif 'violence' in video_path.lower():
            video_event_type = 'violence'
        elif 'cash' in video_path.lower():
            video_event_type = 'cash'
        elif 'fire' in video_path.lower():
            video_event_type = 'fire'
        else:
            video_event_type = 'violence'  # default
        
        results['video'] = test_video_validation(video_path, video_event_type, validator)
    
    # Summary
    print_separator("TEST SUMMARY")
    
    for test_type, result in results.items():
        if result:
            is_valid, confidence, reason, corrected_type = result['parsed']
            status = "✅ PASS" if is_valid else "❌ FAIL"
            print(f"\n{test_type.upper()}: {status}")
            print(f"  - Valid: {is_valid}, Confidence: {confidence:.2%}")
            print(f"  - Corrected Type: {corrected_type}")
            print(f"  - Processing Time: {result.get('processing_time_ms', 0)}ms")
        else:
            print(f"\n{test_type.upper()}: ⚠️  SKIPPED (file not found)")
    
    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
