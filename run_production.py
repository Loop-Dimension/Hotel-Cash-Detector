"""
Production server for Flask app with large file upload support
Uses waitress (production WSGI server for Windows)
"""

from waitress import serve
from app import app
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

max_size_mb = config.get('MAX_FILE_SIZE_MB', 2048)

print("\n" + "="*70)
print("🚀 Starting Production Server (Waitress)")
print("="*70)
print("Open your browser: http://localhost:5000")
print(f"📏 Max file size: {max_size_mb}MB")
print(f"⚡ Threaded: Yes")
print(f"🔧 Server: Waitress (Production)")
print("="*70 + "\n")

# Serve with waitress - better for large uploads
serve(
    app,
    host='0.0.0.0',
    port=5000,
    threads=6,  # Handle multiple uploads simultaneously
    channel_timeout=300,  # 5 minutes timeout for large uploads
    max_request_body_size=max_size_mb * 1024 * 1024,  # Set max body size
    asyncore_use_poll=True  # Better for Windows
)

