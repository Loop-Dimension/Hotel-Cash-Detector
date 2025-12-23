import multiprocessing

bind = "127.0.0.1:8000"
workers = 1
threads = 4
worker_class = "gthread"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"

# Process naming
proc_name = "hotel-cctv"

# Server mechanics
daemon = False
# pidfile removed - systemd manages the process (DELETE any pidfile line)
user = "ubuntu"
group = "ubuntu"