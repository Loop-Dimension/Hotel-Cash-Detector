# Option 1: Clear all logs for hotel-cctv service (recommended)
sudo journalctl --vacuum-time=1s -u hotel-cctv

# Option 2: Rotate then vacuum
sudo journalctl --rotate
sudo journalctl --vacuum-time=1s -u hotel-cctv

# Option 3: Clear ALL journalctl logs (entire system)
sudo journalctl --vacuum-time=1s

journalctl -u hotel-cctv
# Should show minimal or no logs