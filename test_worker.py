"""
Test script to manually start a worker and check if it's working
"""
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hotel_cctv.settings')
django.setup()

from cctv.models import Camera
from cctv.worker_process import CameraWorkerProcess
import time

def main():
    print("=" * 60)
    print("WORKER TEST SCRIPT")
    print("=" * 60)

    # Get first camera
    cameras = Camera.objects.all()
    if not cameras.exists():
        print("❌ No cameras found in database")
        print("\nRun this first:")
        print("  python manage.py seed_data")
        sys.exit(1)

    camera = cameras.first()
    print(f"\n📹 Testing camera: {camera.camera_id} - {camera.name}")
    print(f"   RTSP: {camera.rtsp_url}")

    # Create worker
    print("\n🚀 Starting worker process...")
    try:
        worker = CameraWorkerProcess(camera.id)
        worker.start()
        
        print("✅ Worker started!")
        print("\n⏱️  Waiting 15 seconds for connection...")
        
        for i in range(15):
            time.sleep(1)
            status = worker.get_status()
            print(f"   [{i+1}/15] Status: {status['status']} | "
                  f"Running: {status['running']} | "
                  f"Alive: {status['alive']} | "
                  f"Frames: {status['frames_processed']}")
            
            if status.get('error'):
                print(f"   ⚠️  Error: {status['error']}")
        
        print("\n📊 Final status:")
        final_status = worker.get_status()
        for key, value in final_status.items():
            print(f"   {key}: {value}")
        
        # Try to get a frame
        print("\n🖼️  Testing frame retrieval...")
        frame = worker.get_current_frame()
        if frame is not None:
            print(f"   ✅ Got frame: {frame.shape}")
        else:
            print("   ❌ No frame available")
            print("   Note: This is normal if RTSP stream is not accessible")
        
        # Stop worker
        print("\n🛑 Stopping worker...")
        worker.stop()
        print("✅ Worker stopped")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
