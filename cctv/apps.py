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
        if 'runserver' not in sys.argv and 'gunicorn' not in sys.argv[0] if sys.argv else True:
            return
        
        # Avoid running twice (Django calls ready() twice in dev with auto-reload)
        if os.environ.get('CCTV_WORKERS_STARTED'):
            return
        os.environ['CCTV_WORKERS_STARTED'] = '1'
        
        # Start workers in a separate thread after a short delay
        # This ensures Django is fully loaded
        def start_workers_delayed():
            import time
            time.sleep(5)  # Wait for Django to fully start
            try:
                from .views import start_all_background_workers_internal
                start_all_background_workers_internal()
                print("=" * 60)
                print("  ✅ BACKGROUND DETECTION SERVICE AUTO-STARTED")
                print("=" * 60)
            except Exception as e:
                print(f"[WARNING] Could not auto-start workers: {e}")
        
        thread = threading.Thread(target=start_workers_delayed, daemon=True)
        thread.start()
