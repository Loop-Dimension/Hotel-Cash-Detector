"""
Hotel CCTV Monitoring System - Flask Application

Main application for:
- Video upload and processing
- Live CCTV monitoring
- Real-time detection alerts (Cash, Violence, Fire)
"""
import os
import json
import cv2
import time
import threading
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

from config import (
    SECRET_KEY, MAX_CONTENT_LENGTH, 
    INPUT_DIR, OUTPUT_DIR, UPLOAD_DIR, MODELS_DIR,
    DetectionConfig
)
from detectors import UnifiedDetector

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)

# Initialize SocketIO for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
detector = None
current_video_path = None
is_processing = False
processing_thread = None
camera_configs = {}

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_camera_configs():
    """Load all camera configurations"""
    global camera_configs
    camera_configs = {}
    
    # Check input directory for camera configs
    if INPUT_DIR.exists():
        for item in INPUT_DIR.iterdir():
            if item.is_dir() and item.name.startswith('camera'):
                config_file = item / 'config.json'
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        camera_configs[item.name] = json.load(f)
    
    return camera_configs


def get_detector(cashier_zone=None):
    """Get or create the unified detector"""
    global detector
    
    if detector is None:
        config = {
            'models_dir': str(MODELS_DIR),
            'cashier_zone': cashier_zone or DetectionConfig.DEFAULT_CASHIER_ZONE,
            'hand_touch_distance': DetectionConfig.HAND_TOUCH_DISTANCE,
            'pose_confidence': DetectionConfig.POSE_CONFIDENCE,
            'min_transaction_frames': DetectionConfig.MIN_TRANSACTION_FRAMES,
            # Fire detection settings
            'fire_confidence': DetectionConfig.FIRE_CONFIDENCE,
            'min_fire_frames': DetectionConfig.MIN_FIRE_FRAMES,
            'min_fire_area': DetectionConfig.MIN_FIRE_AREA,
            # Violence detection settings
            'violence_confidence': DetectionConfig.VIOLENCE_CONFIDENCE,
            'min_violence_frames': DetectionConfig.VIOLENCE_DURATION,
            # Toggles
            'detect_cash': True,
            'detect_violence': True,
            'detect_fire': True
        }
        detector = UnifiedDetector(config)
    
    return detector


def generate_frames(video_path):
    """Generator for video frames with detection"""
    global is_processing
    
    det = get_detector()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield b''
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = 1.0 / fps
    
    is_processing = True
    frame_count = 0
    
    while is_processing:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        
        # Skip frames for performance
        if frame_count % DetectionConfig.FRAME_SKIP != 0:
            continue
        
        # Process frame
        result = det.process_frame(frame, draw_overlay=True)
        
        # Send alerts via WebSocket
        if result['detections']:
            socketio.emit('detections', {
                'frame': frame_count,
                'detections': result['detections'],
                'timestamp': datetime.now().isoformat()
            })
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', result['frame'], [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(frame_delay / 2)  # Maintain rough fps
    
    cap.release()


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main dashboard"""
    # Get available videos
    videos = []
    
    # Check input folder
    if INPUT_DIR.exists():
        for video in INPUT_DIR.glob('*.mp4'):
            videos.append({
                'name': video.name,
                'path': str(video),
                'source': 'input'
            })
    
    # Check uploads folder
    if UPLOAD_DIR.exists():
        for video in UPLOAD_DIR.glob('*'):
            if video.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                videos.append({
                    'name': video.name,
                    'path': str(video),
                    'source': 'upload'
                })
    
    # Load camera configs
    configs = load_camera_configs()
    
    return render_template('index.html', 
                         videos=videos, 
                         camera_configs=configs)


@app.route('/upload', methods=['POST'])
def upload_video():
    """Upload a video file"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        
        filepath = UPLOAD_DIR / filename
        file.save(str(filepath))
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': str(filepath)
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/video_feed')
def video_feed():
    """Video streaming endpoint"""
    global current_video_path
    
    video_path = request.args.get('path')
    if not video_path:
        # Use default video from input folder
        videos = list(INPUT_DIR.glob('*.mp4'))
        if videos:
            video_path = str(videos[0])
        else:
            return "No video available", 404
    
    current_video_path = video_path
    
    return Response(
        generate_frames(video_path),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stop_processing')
def stop_processing():
    """Stop video processing"""
    global is_processing
    is_processing = False
    return jsonify({'success': True})


@app.route('/set_cashier_zone', methods=['POST'])
def set_cashier_zone():
    """Update the cashier zone"""
    data = request.json
    zone = data.get('zone')  # [x, y, width, height]
    
    if not zone or len(zone) != 4:
        return jsonify({'error': 'Invalid zone format'}), 400
    
    det = get_detector()
    det.set_cashier_zone(zone)
    
    # Also save to config file for persistence
    config_file = INPUT_DIR / 'cashier_zone.json'
    with open(config_file, 'w') as f:
        json.dump({'cashier_zone': zone}, f)
    
    return jsonify({'success': True, 'zone': zone})


@app.route('/get_cashier_zone')
def get_cashier_zone():
    """Get the current cashier zone"""
    det = get_detector()
    zone = det.cash_detector.cashier_zone if det.cash_detector else DetectionConfig.DEFAULT_CASHIER_ZONE
    return jsonify({'zone': zone})


@app.route('/get_video_info')
def get_video_info():
    """Get video dimensions for zone drawing"""
    video_path = request.args.get('path')
    if not video_path:
        return jsonify({'error': 'No video path provided'}), 400
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({'error': 'Cannot open video'}), 400
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return jsonify({
        'width': width,
        'height': height,
        'path': video_path
    })


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    """Toggle detection types"""
    data = request.json
    detection_type = data.get('type')
    enabled = data.get('enabled', True)
    
    det = get_detector()
    
    if detection_type == 'cash':
        det.detect_cash = enabled
    elif detection_type == 'violence':
        det.detect_violence = enabled
    elif detection_type == 'fire':
        det.detect_fire = enabled
    
    return jsonify({
        'success': True,
        'detect_cash': det.detect_cash,
        'detect_violence': det.detect_violence,
        'detect_fire': det.detect_fire
    })


@app.route('/toggle_debug', methods=['POST'])
def toggle_debug():
    """Toggle debug mode"""
    data = request.json
    enabled = data.get('enabled', False)
    
    det = get_detector()
    result = det.toggle_debug(enabled)
    
    return jsonify({
        'success': True,
        'debug_mode': result
    })


@app.route('/get_summary')
def get_summary():
    """Get detection summary"""
    det = get_detector()
    summary = det.get_detection_summary()
    return jsonify(summary)


@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts"""
    det = get_detector()
    return jsonify({
        'alerts': det.alerts_history[-20:]
    })


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


# ==================== WEBSOCKET EVENTS ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'status': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    pass


@socketio.on('start_processing')
def handle_start_processing(data):
    """Start video processing"""
    video_path = data.get('video_path')
    if video_path:
        emit('processing_started', {'path': video_path})


@socketio.on('get_status')
def handle_get_status():
    """Get current processing status"""
    det = get_detector()
    emit('status', {
        'is_processing': is_processing,
        'frame_count': det.frame_count if det else 0,
        'detect_cash': det.detect_cash if det else True,
        'detect_violence': det.detect_violence if det else True,
        'detect_fire': det.detect_fire if det else True
    })


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🏨 HOTEL CCTV MONITORING SYSTEM")
    print("=" * 60)
    print()
    print("Detection Types:")
    print("  💵 Cash Transactions")
    print("  ⚠️  Violence")
    print("  🔥 Fire/Smoke")
    print()
    print("Starting server...")
    print()
    print("🌐 Open http://localhost:5000 in your browser")
    print("=" * 60)
    print()
    
    # Create necessary directories
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run the app
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
