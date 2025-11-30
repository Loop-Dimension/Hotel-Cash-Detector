"""
Configuration settings for Hotel Cash Detector
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Input/Output directories
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"

# Create directories if they don't exist
for dir_path in [INPUT_DIR, OUTPUT_DIR, MODELS_DIR, UPLOAD_DIR]:
    dir_path.mkdir(exist_ok=True)

# Flask settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'hotel-cash-detector-secret-key-2024')
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max upload

# Detection settings
class DetectionConfig:
    # Model paths
    YOLO_MODEL = str(MODELS_DIR / "yolov8n.pt")
    POSE_MODEL = str(MODELS_DIR / "yolov8n-pose.pt")
    FIRE_MODEL = str(MODELS_DIR / "fire_smoke_yolov8.pt")  # Trained fire/smoke model
    
    # Detection thresholds
    CONFIDENCE_THRESHOLD = 0.5
    POSE_CONFIDENCE = 0.5
    
    # Cash transaction detection
    HAND_TOUCH_DISTANCE = 80  # pixels - distance to consider hands touching
    MIN_TRANSACTION_FRAMES = 3  # minimum frames to confirm transaction
    
    # Violence detection
    VIOLENCE_CONFIDENCE = 0.6
    VIOLENCE_DURATION = 5  # frames to confirm violence
    
    # Fire detection - STRICT settings to reduce false positives
    FIRE_CONFIDENCE = 0.3  # Higher threshold
    MIN_FIRE_FRAMES = 4   # Need more consecutive frames
    MIN_FIRE_AREA = 50   # Larger minimum area
    
    # Cashier zone (will be loaded from camera config)
    DEFAULT_CASHIER_ZONE = [0, 280, 900, 400]  # x, y, width, height
    
    # Video processing
    FRAME_SKIP = 2  # Process every Nth frame
    
    # Alert settings
    ALERT_COOLDOWN = 30  # seconds between same type of alerts

# Detection labels
DETECTION_LABELS = {
    "CASH": {
        "name": "Cash Transaction",
        "color": (0, 255, 0),  # Green
        "priority": 1,
        "icon": "💵"
    },
    "VIOLENCE": {
        "name": "Violence Detected",
        "color": (0, 0, 255),  # Red
        "priority": 3,
        "icon": "⚠️"
    },
    "FIRE": {
        "name": "Fire/Smoke Detected",
        "color": (0, 69, 255),  # Orange-Red
        "priority": 3,
        "icon": "🔥"
    }
}
