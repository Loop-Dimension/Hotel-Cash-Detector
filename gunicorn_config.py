import multiprocessing

bind = "127.0.0.1:8000"
workers = 4
worker_class = "gevent"
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
pidfile = "/home/ubuntu/Hotel-Cash-Detector/gunicorn.pid"
user = "ubuntu"
group = "ubuntu"
