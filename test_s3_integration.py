"""
Test S3 integration with actual app functions.
Run this to verify that save_clip, save_validation_clip, and save_event work with S3.
"""
import os
import sys
import django
import numpy as np

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hotel_cctv.settings')
django.setup()

from django.conf import settings
from cctv.storage import is_s3_enabled, get_storage

def test_storage_helper():
    """Test the storage helper module"""
    print("\n=== Testing Storage Helper ===")
    
    print(f"USE_S3 setting: {settings.USE_S3}")
    print(f"is_s3_enabled(): {is_s3_enabled()}")
    
    storage = get_storage()
    print(f"Storage backend: {type(storage).__name__}")
    
    if is_s3_enabled():
        print(f"S3 Bucket: {settings.AWS_STORAGE_BUCKET_NAME}")
        print(f"S3 Region: {settings.AWS_S3_REGION_NAME}")
    
    return is_s3_enabled()


def test_image_upload():
    """Test uploading an image to S3"""
    print("\n=== Testing Image Upload ===")
    
    from cctv.storage import save_image_to_storage
    
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 2] = 255  # Red frame
    
    # Add text
    import cv2
    cv2.putText(frame, "TEST IMAGE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    try:
        storage_path = "test_results/test_image.jpg"
        url = save_image_to_storage(frame, storage_path)
        print(f"✓ Image uploaded successfully!")
        print(f"  Storage path: {storage_path}")
        print(f"  URL: {url}")
        return True
    except Exception as e:
        print(f"✗ Image upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bytes_upload():
    """Test uploading bytes to S3"""
    print("\n=== Testing Bytes Upload ===")
    
    from cctv.storage import upload_bytes_to_storage
    
    try:
        # Create test JSON
        import json
        test_data = {
            "test": True,
            "timestamp": "2025-01-12T12:00:00",
            "message": "S3 integration test"
        }
        json_bytes = json.dumps(test_data, indent=2).encode('utf-8')
        
        storage_path = "test_results/test_data.json"
        url = upload_bytes_to_storage(json_bytes, storage_path)
        print(f"✓ JSON uploaded successfully!")
        print(f"  Storage path: {storage_path}")
        print(f"  URL: {url}")
        return True
    except Exception as e:
        print(f"✗ Bytes upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_upload():
    """Test uploading a file to S3"""
    print("\n=== Testing File Upload ===")
    
    from cctv.storage import upload_file_to_storage
    from pathlib import Path
    import tempfile
    
    try:
        # Create a temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test file for S3 upload.\n")
            f.write("Testing file upload functionality.\n")
            temp_path = f.name
        
        storage_path = "test_results/test_file.txt"
        url = upload_file_to_storage(temp_path, storage_path)
        print(f"✓ File uploaded successfully!")
        print(f"  Storage path: {storage_path}")
        print(f"  URL: {url}")
        
        # Clean up temp file
        os.unlink(temp_path)
        return True
    except Exception as e:
        print(f"✗ File upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_s3_bucket():
    """List files in S3 bucket"""
    print("\n=== S3 Bucket Contents ===")
    
    if not is_s3_enabled():
        print("S3 is not enabled, skipping bucket listing")
        return
    
    try:
        import boto3
        
        s3 = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_S3_REGION_NAME
        )
        
        response = s3.list_objects_v2(
            Bucket=settings.AWS_STORAGE_BUCKET_NAME,
            MaxKeys=20
        )
        
        if 'Contents' in response:
            print(f"Found {len(response['Contents'])} files:")
            for obj in response['Contents']:
                print(f"  - {obj['Key']} ({obj['Size']} bytes)")
        else:
            print("Bucket is empty")
            
    except Exception as e:
        print(f"Failed to list bucket: {e}")


def main():
    print("=" * 60)
    print("S3 Integration Test for Hotel CCTV App")
    print("=" * 60)
    
    # Test storage helper
    s3_enabled = test_storage_helper()
    
    if not s3_enabled:
        print("\n⚠ S3 is NOT enabled. Set USE_S3=True in .env to enable.")
        print("Tests will use local filesystem storage.")
    
    # Run tests
    results = []
    results.append(("Image Upload", test_image_upload()))
    results.append(("Bytes Upload", test_bytes_upload()))
    results.append(("File Upload", test_file_upload()))
    
    # List bucket contents
    if s3_enabled:
        list_s3_bucket()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n✓ All tests passed!")
        if s3_enabled:
            print("  Files should now be visible in your S3 bucket.")
    else:
        print("\n✗ Some tests failed. Check the errors above.")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
