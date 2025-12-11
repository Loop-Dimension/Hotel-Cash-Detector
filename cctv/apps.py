from django.apps import AppConfig
import os
import threading


class CctvConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'cctv'
    
    def ready(self):
        """Called when Django starts - auto-start background workers"""
        # Only run in the main process (not in migrations, shell, etc.)
        # Check if we're running the server (not migrations or other commands)
        import sys
        if 'runserver' not in sys.argv and 'gunicorn' not in ' '.join(sys.argv):
            return
        
        # For Gunicorn: Only run in ONE worker process
        # Avoid running in every worker by checking worker ID
        worker_id = os.environ.get('GUNICORN_WORKER_ID', os.getpid())
        if os.environ.get('CCTV_WORKERS_STARTED'):
            return
        os.environ['CCTV_WORKERS_STARTED'] = str(worker_id)
        os.environ['GUNICORN_WORKER_ID'] = str(worker_id)
        
        # Start workers in a separate thread after a short delay
        # This ensures Django is fully loaded
        def start_workers_delayed():
            import time
            time.sleep(5)  # Wait for Django to fully start
            try:
                from .views import start_all_background_workers_internal
                print("\n" + "=" * 60)
                print(f"  🚀 AUTO-STARTING WORKERS (PID: {os.getpid()})")
                print("=" * 60)
                started = start_all_background_workers_internal()
                print("=" * 60)
                print(f"  ✅ STARTED {len(started)} DETECTION WORKERS")
                print("=" * 60)
            except Exception as e:
                print(f"[WARNING] Could not auto-start workers: {e}")
                import traceback
                traceback.print_exc()
        
        thread = threading.Thread(target=start_workers_delayed, daemon=True)
        thread.start()
