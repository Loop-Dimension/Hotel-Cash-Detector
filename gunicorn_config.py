import multiprocessing

bind = "127.0.0.1:8000"

# Use 1 worker for detection app (detection threads handle parallelism)
# More workers = faster web requests, but each runs separate detection
workers = 1

# Use sync worker (not gevent) for better threading compatibility
worker_class = "sync"

# Use threads for web request parallelism (doesn't conflict with detection threads)
threads = 4

timeout = 300  # 5 min timeout for long operations
keepalive = 5

# Logging
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"

# Process naming
proc_name = "hotel-cctv"

# Server mechanics
daemon = False
pidfile = "/home/ubuntu/Hotel-Cash-Detector/gunicorn.pid"
user = "ubuntu"
group = "ubuntu"

# Preload app to save memory and ensure single detection instance
preload_app = True
