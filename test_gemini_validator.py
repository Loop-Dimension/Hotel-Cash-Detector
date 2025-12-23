"""
Test script for Gemini AI Validator

This script tests the Gemini validator with sample images.
Usage:
    python test_gemini_validator.py
    python test_gemini_validator.py --image path/to/image.jpg --event-type cash
"""

import os
import sys
import cv2
import argparse
from pathlib import Path

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hotel_cctv.settings')
import django
django.setup()

from django.conf import settings
from detectors.gemini_validator import GeminiValidator


def create_test_frame():
    """Create a simple test frame if no image provided"""
    import numpy as np
    
    # Create a blank 640x480 frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some text
    cv2.putText(frame, "TEST FRAME", (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def test_validator_with_image(image_path, event_type='cash'):
    """Test validator with a specific image"""
    print(f"\n{'='*60}")
    print(f"Testing Gemini Validator")
    print(f"{'='*60}")
    
    # Check API key
    api_key = getattr(settings, 'GEMINI_API_KEY', '')
    if not api_key:
        print("❌ ERROR: No GEMINI_API_KEY found in settings")
        print("Please add GEMINI_API_KEY to your .env file")
        return False
    
    print(f"✓ API Key found: {api_key[:20]}...")
    
    # Load image
    if image_path:
        print(f"\n📷 Loading image: {image_path}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"❌ ERROR: Could not load image: {image_path}")
            return False
        print(f"✓ Image loaded: {frame.shape}")
    else:
        print("\n📷 Creating test frame...")
        frame = create_test_frame()
        print(f"✓ Test frame created: {frame.shape}")
    
    # Create validator
    print(f"\n🤖 Initializing Gemini Validator...")
    validator = GeminiValidator(api_key=api_key)
    
    if not validator.enabled:
        print("❌ ERROR: Validator not enabled (check API key)")
        return False
    
    print(f"✓ Validator initialized")
    
    # Test validation
    print(f"\n🔍 Testing validation for event type: {event_type}")
    print(f"{'='*60}")
    
    try:
        is_valid, confidence, reason = validator.validate_event(frame, event_type)
        
        print(f"\nResults:")
        print(f"  Valid: {'✅ YES' if is_valid else '❌ NO'}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Reason: {reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_event_types(image_path=None):
    """Test all event types with the same image"""
    print(f"\n{'='*60}")
    print(f"Testing All Event Types")
    print(f"{'='*60}")
    
    event_types = ['cash', 'violence', 'fire']
    
    for event_type in event_types:
        print(f"\n\n{'='*60}")
        print(f"Event Type: {event_type.upper()}")
        print(f"{'='*60}")
        
        success = test_validator_with_image(image_path, event_type)
        
        if not success:
            print(f"❌ Failed to test {event_type}")


def find_sample_images():
    """Find sample images in the workspace"""
    # Check common locations
    locations = [
        Path('media/thumbnails'),
        Path('testing'),
        Path('uploads'),
        Path('input'),
    ]
    
    images = []
    for loc in locations:
        if loc.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images.extend(list(loc.glob(ext)))
    
    return images[:5]  # Return first 5


def main():
    parser = argparse.ArgumentParser(description='Test Gemini AI Validator')
    parser.add_argument('--image', '-i', type=str, help='Path to test image')
    parser.add_argument('--event-type', '-e', type=str, 
                       choices=['cash', 'violence', 'fire', 'all'],
                       default='all',
                       help='Event type to test (default: all)')
    parser.add_argument('--list-samples', '-l', action='store_true',
                       help='List available sample images')
    
    args = parser.parse_args()
    
    # List samples
    if args.list_samples:
        print("\n📁 Searching for sample images...")
        samples = find_sample_images()
        if samples:
            print(f"\nFound {len(samples)} sample images:")
            for i, img in enumerate(samples, 1):
                print(f"  {i}. {img}")
            print(f"\nUse: python test_gemini_validator.py --image \"{samples[0]}\"")
        else:
            print("No sample images found")
        return
    
    # Run tests
    if args.event_type == 'all':
        test_all_event_types(args.image)
    else:
        test_validator_with_image(args.image, args.event_type)
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
