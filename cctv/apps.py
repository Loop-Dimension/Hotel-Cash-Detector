from django.apps import AppConfig
import os
import threading


class CctvConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'cctv'
    
    def ready(self):
        """Called when Django starts - auto-start background workers"""
        import sys
        
        # Only run for actual server processes
        running_server = False
        if 'runserver' in sys.argv:
            running_server = True
        elif any('gunicorn' in arg for arg in sys.argv):
            running_server = True
        
        if not running_server:
            return
        
        # For Gunicorn with preload_app: only start once (main process)
        # For regular runserver: use environment variable check
        if os.environ.get('CCTV_WORKERS_STARTED') == 'true':
            return
        os.environ['CCTV_WORKERS_STARTED'] = 'true'
        
        # Start workers in a separate thread after delay
        def start_workers_delayed():
            import time
            time.sleep(8)  # Wait for Django + models to be fully loaded
            try:
                from .views import start_all_background_workers_internal
                print("\n" + "=" * 60)
                print(f"  🚀 AUTO-STARTING DETECTION WORKERS")
                print("=" * 60)
                started = start_all_background_workers_internal()
                print("=" * 60)
                if started:
                    print(f"  ✅ STARTED {len(started)} DETECTION WORKERS: {started}")
                else:
                    print("  ⚠️  No cameras found or all failed to start")
                print("=" * 60 + "\n")
            except Exception as e:
                print(f"[WARNING] Could not auto-start workers: {e}")
                import traceback
                traceback.print_exc()
        
        thread = threading.Thread(target=start_workers_delayed, daemon=True)
        thread.start()
