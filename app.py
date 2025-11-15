"""
Flask App for Hotel Cash Transaction Detection
Upload videos and extract clips around detected transactions
Uses the EXACT same detection logic as main.py
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import json
import threading
import uuid
from datetime import datetime
import cv2
from ultralytics import YOLO
import numpy as np
import math
import time
import subprocess
import shutil

# Import the detection classes
from main import SimpleHandTouchConfig, SimpleHandTouchDetector
from multi_detector import MultiEventDetector

app = Flask(__name__)

# Check if ffmpeg is available for video conversion
def check_ffmpeg():
    """Check if ffmpeg is available on the system"""
    return shutil.which('ffmpeg') is not None

FFMPEG_AVAILABLE = check_ffmpeg()
if FFMPEG_AVAILABLE:
    print("✅ FFmpeg detected - will use for video conversion")
else:
    print("⚠️  FFmpeg not found - using OpenCV codecs only")

def convert_avi_to_mp4(avi_path):
    """Convert AVI file to browser-compatible MP4 using ffmpeg"""
    if not FFMPEG_AVAILABLE:
        return None
    
    mp4_path = avi_path.replace('.avi', '.mp4')
    
    try:
        # Use ffmpeg to convert with H.264 codec (browser compatible)
        cmd = [
            'ffmpeg',
            '-i', avi_path,
            '-c:v', 'libx264',  # H.264 video codec
            '-preset', 'fast',   # Encoding speed
            '-crf', '23',        # Quality (lower = better, 23 is good)
            '-y',                # Overwrite output file
            mp4_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(mp4_path):
            # Delete original AVI file
            try:
                os.remove(avi_path)
            except:
                pass
            return mp4_path
        else:
            print(f"⚠️  FFmpeg conversion failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"⚠️  FFmpeg error: {e}")
        return None

# Load configuration from config.json
CONFIG_FILE = 'config.json'
try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        APP_CONFIG = json.load(f)
    print(f"✅ Loaded configuration from {CONFIG_FILE}")
except FileNotFoundError:
    print(f"⚠️  Config file not found, using defaults")
    APP_CONFIG = {
        'MAX_FILE_SIZE_MB': 500,
        'ALLOWED_EXTENSIONS': ['mp4', 'avi', 'mov', 'mkv'],
        'MAX_VIDEOS_PER_UPLOAD': 5,
        'SECONDS_BEFORE_TRANSACTION': 2,
        'SECONDS_AFTER_TRANSACTION': 2
    }

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = APP_CONFIG.get('MAX_FILE_SIZE_MB', 500) * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = set(APP_CONFIG.get('ALLOWED_EXTENSIONS', ['mp4', 'avi', 'mov', 'mkv']))
app.config['MAX_VIDEOS'] = APP_CONFIG.get('MAX_VIDEOS_PER_UPLOAD', 5)

# Create necessary folders
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['OUTPUT_FOLDER']).mkdir(exist_ok=True)
Path('static').mkdir(exist_ok=True)

# Store processing status
processing_status = {}

print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
print(f"📁 Output folder: {app.config['OUTPUT_FOLDER']}")
print(f"📏 Max file size: {APP_CONFIG.get('MAX_FILE_SIZE_MB', 500)}MB")
print(f"🎥 Max videos per upload: {app.config['MAX_VIDEOS']}")
print()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    max_size = APP_CONFIG.get('MAX_FILE_SIZE_MB', 500)
    return jsonify({
        'error': 'File too large',
        'message': f'파일 크기가 너무 큽니다. 최대 허용 크기: {max_size}MB',
        'max_size_mb': max_size
    }), 413


class TransactionClipExtractor:
    """Extract clips around detected events (Cash, Fire, Violence) using multi-detector"""
    
    def __init__(self, config_dict=None):
        # Initialize multi-event detector
        self.detector = MultiEventDetector(config_dict)
        self.config_dict = config_dict
        
        # Clip extraction settings from config
        self.SECONDS_BEFORE = config_dict.get('SECONDS_BEFORE_TRANSACTION', 2)
        self.SECONDS_AFTER = config_dict.get('SECONDS_AFTER_TRANSACTION', 2)
        self.MERGE_THRESHOLD = config_dict.get('MERGE_CLIPS_WITHIN_SECONDS', 0.5)
        
        self.detection_types = config_dict.get('DETECTION_TYPES', {})
        self.video_extension = '.avi'  # Will be set based on available codec
        
        print(f"⏱️  Clip padding: {self.SECONDS_BEFORE}s before, {self.SECONDS_AFTER}s after")
        print(f"🔗 Merge clips within: {self.MERGE_THRESHOLD}s")
        print()
    
    def detect_transactions(self, video_path, progress_callback=None):
        """Detect all events (cash, fire, violence) in video and save annotated version"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0, None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create temporary annotated video with browser-compatible codec
        temp_annotated_path = video_path.replace('.mp4', '_annotated_temp.avi')  # Use AVI for compatibility
        
        # Try codecs in order of reliability (MJPEG works everywhere)
        fourcc_options = [
            ('MJPG', 'Motion JPEG (universal browser support)', '.avi'),
            ('XVID', 'XVID (good compatibility)', '.avi'),
            ('VP80', 'VP8 WebM (if available)', '.webm'),
            ('VP90', 'VP9 WebM (if available)', '.webm'),
        ]
        
        annotated_writer = None
        actual_extension = '.avi'  # Default to AVI
        for codec, description, ext in fourcc_options:
            try:
                test_path = video_path.replace('.mp4', f'_annotated_temp{ext}')
                fourcc = cv2.VideoWriter_fourcc(*codec)
                annotated_writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                if annotated_writer.isOpened():
                    print(f"✅ Using codec: {codec} ({description})")
                    temp_annotated_path = test_path
                    actual_extension = ext
                    break
                else:
                    # Clean up failed attempt
                    if os.path.exists(test_path):
                        os.remove(test_path)
            except Exception as e:
                continue
        
        if annotated_writer is None or not annotated_writer.isOpened():
            # Last resort: raw AVI (very large but always works)
            print("⚠️  Warning: Using uncompressed AVI (large file)")
            temp_annotated_path = video_path.replace('.mp4', '_annotated_temp.avi')
            actual_extension = '.avi'
            fourcc = 0  # Uncompressed
            annotated_writer = cv2.VideoWriter(temp_annotated_path, fourcc, fps, (width, height))
        
        # Store the extension for use in extract_clips
        self.video_extension = actual_extension
        print(f"📹 Output format: {actual_extension}")
        
        # Reset cash detector state if it exists
        if 'CASH_EXCHANGE' in self.detector.detectors:
            cash_detector = self.detector.detectors['CASH_EXCHANGE']
            cash_detector.transaction_history = {}
            cash_detector.person_id_map = {}
            cash_detector.next_stable_id = 1
            cash_detector.cashier_persistence = {}
            cash_detector.stats = {
                'frames': 0, 
                'transactions': 0, 
                'confirmed_transactions': 0,
                'cash_detections': 0,
                'cash_types': {}
            }
        
        all_events = []  # Store all detected events
        current_events = {}  # Track ongoing events by type
        frame_num = 0
        
        print(f"📹 Analyzing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Progress update
            if progress_callback and frame_num % 30 == 0:
                progress_callback(int(frame_num / total_frames * 100))
            
            # Run ALL detectors on this frame (pass FPS for violence detector)
            output_frame, detections_dict = self.detector.detect_all(frame, fps=fps)
            
            # Save annotated frame
            annotated_writer.write(output_frame)
            
            # Process each detection type
            for det_type, detections in detections_dict.items():
                if det_type == 'CASH_EXCHANGE':
                    # Handle cash transactions
                    for trans in detections:
                        trans_key = f"{trans['p1_id']}-{trans['p2_id']}"
                        event_key = f"CASH_{trans_key}"
                        
                        if event_key not in current_events:
                            current_events[event_key] = {
                                'type': 'CASH_EXCHANGE',
                                'key': trans_key,
                                'start_frame': frame_num,
                                'end_frame': frame_num,
                                'start_time': frame_num / fps,
                                'end_time': frame_num / fps,
                                'p1_id': trans['p1_id'],
                                'p2_id': trans['p2_id'],
                                'hand_type': trans['hand_type']
                            }
                        else:
                            current_events[event_key]['end_frame'] = frame_num
                            current_events[event_key]['end_time'] = frame_num / fps
                
                else:
                    # Handle violence/fire detections
                    for detection in detections:
                        event_key = f"{det_type}_{frame_num}"
                        
                        if det_type not in current_events:
                            current_events[det_type] = {
                                'type': det_type,
                                'start_frame': frame_num,
                                'end_frame': frame_num,
                                'start_time': frame_num / fps,
                                'end_time': frame_num / fps,
                                'confidence': detection.get('confidence', 0.8),
                                'description': detection.get('description', det_type)
                            }
                        else:
                            current_events[det_type]['end_frame'] = frame_num
                            current_events[det_type]['end_time'] = frame_num / fps
            
            # Check for ended events (no detection in this frame)
            ended_events = []
            for event_key, event in list(current_events.items()):
                if event['end_frame'] < frame_num - 5:  # Event ended 5 frames ago
                    all_events.append(event)
                    ended_events.append(event_key)
            
            for event_key in ended_events:
                del current_events[event_key]
        
        # Add remaining active events
        for event in current_events.values():
            all_events.append(event)
        
        cap.release()
        annotated_writer.release()
        
        # Merge close events
        merged_events = self._merge_close_events(all_events, fps)
        
        print(f"✅ Detection complete: Found {len(merged_events)} event(s)")
        for event in merged_events:
            det_type = event.get('type', 'UNKNOWN')
            label = self.detection_types.get(det_type, {}).get('label', det_type)
            print(f"  • {label}: {event['start_time']:.1f}s - {event['end_time']:.1f}s")
        
        return merged_events, fps, temp_annotated_path
    
    def _merge_close_events(self, events, fps):
        """Merge events that are close together (same type)"""
        if not events:
            return []
        
        # Sort by type and start time
        events.sort(key=lambda x: (x.get('type', ''), x['start_time']))
        
        merged = []
        current = events[0].copy()
        
        for event in events[1:]:
            # If same type and within 1 second, merge them
            if (event.get('type') == current.get('type') and 
                event['start_time'] - current['end_time'] <= 1.0):
                current['end_frame'] = event['end_frame']
                current['end_time'] = event['end_time']
                # For cash, merge keys if different
                if 'key' in event and 'key' in current and event['key'] != current['key']:
                    current['key'] = f"{current['key']},{event['key']}"
            else:
                merged.append(current)
                current = event.copy()
        
        merged.append(current)
        return merged
    
    def _merge_overlapping_clips(self, events, fps):
        """Merge events that would create overlapping clips"""
        if not events:
            return []
        
        # Add margins and create clip ranges
        clip_ranges = []
        for event in events:
            start_frame = max(0, event['start_frame'] - int(self.SECONDS_BEFORE * fps))
            end_frame = event['end_frame'] + int(self.SECONDS_AFTER * fps)
            
            # Get priority (lower = higher priority)
            event_type = event.get('type', 'CASH_EXCHANGE')
            priority = self.detection_types.get(event_type, {}).get('priority', 99)
            
            clip_ranges.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_time': start_frame / fps,
                'end_time': end_frame / fps,
                'type': event_type,
                'priority': priority,
                'p1_id': event.get('p1_id'),
                'p2_id': event.get('p2_id'),
                'hand_type': event.get('hand_type', 'N/A'),
                'confidence': event.get('confidence'),
                'description': event.get('description'),
                'events': [event]  # Track which events are in this clip
            })
        
        # Sort by priority first, then start frame
        clip_ranges.sort(key=lambda x: (x['priority'], x['start_frame']))
        
        # Merge overlapping ranges
        merged = []
        current = clip_ranges[0]
        
        for clip in clip_ranges[1:]:
            # Check if clips overlap or are very close
            overlap_threshold = int(self.MERGE_THRESHOLD * fps)
            
            if clip['start_frame'] <= current['end_frame'] + overlap_threshold:
                # Merge: extend current clip and keep highest priority
                current['end_frame'] = max(current['end_frame'], clip['end_frame'])
                current['end_time'] = current['end_frame'] / fps
                current['events'].extend(clip['events'])
                # Keep highest priority (lower number)
                if clip['priority'] < current['priority']:
                    current['type'] = clip['type']
                    current['priority'] = clip['priority']
                print(f"  🔗 Merging overlapping clips: {current['start_time']:.1f}s - {current['end_time']:.1f}s")
            else:
                # No overlap: save current and start new
                merged.append(current)
                current = clip
        
        merged.append(current)
        
        print(f"  ✅ Merged {len(clip_ranges)} detections into {len(merged)} clips")
        return merged
    
    def extract_clips(self, annotated_video_path, events, fps, output_folder, progress_callback=None):
        """Extract video clips from pre-annotated video (FAST - no re-detection!)"""
        cap = cv2.VideoCapture(annotated_video_path)
        if not cap.isOpened():
            return []
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Merge overlapping clips first
        merged_clips = self._merge_overlapping_clips(events, fps)
        
        clips = []
        
        for idx, clip_info in enumerate(merged_clips):
            start_frame = clip_info['start_frame']
            end_frame = clip_info['end_frame']
            start_time = clip_info['start_time']
            end_time = clip_info['end_time']
            
            # Create output path
            video_name = Path(annotated_video_path).stem.replace('_annotated_temp', '')
            event_type = clip_info.get('type', 'CASH_EXCHANGE')
            label = self.detection_types.get(event_type, {}).get('label', event_type)
            
            # Generate filename based on event type and content
            # Use the same extension as the annotated temp video
            base_name = f"{video_name}_{event_type}"
            if event_type == 'CASH_EXCHANGE' and clip_info.get('p1_id'):
                base_name += f"_P{clip_info['p1_id']}_P{clip_info['p2_id']}"
            base_name += f"_{int(start_time)}s"
            clip_name = base_name + self.video_extension
            
            clip_path = os.path.join(output_folder, clip_name)
            
            # Use same codec as the temp annotated video
            # Extract from the pre-encoded annotated video (no re-encoding needed!)
            # We'll just copy frames directly, which is fast
            if self.video_extension == '.avi':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG
            elif self.video_extension == '.webm':
                fourcc = cv2.VideoWriter_fourcc(*'VP80')  # VP8
            else:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Default to MJPEG
            
            out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                # Fallback to MJPEG
                print(f"⚠️  Warning: Primary codec failed, trying MJPEG fallback...")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    # Last resort: uncompressed
                    fourcc = 0
                    out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_written = 0
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Frame is already annotated - just write it!
                out.write(frame)
                frames_written += 1
            
            out.release()
            
            # Convert AVI to browser-compatible MP4 if ffmpeg is available
            final_path = clip_path
            final_name = clip_name
            if FFMPEG_AVAILABLE and clip_path.endswith('.avi'):
                print(f"  🔄 Converting to MP4...")
                mp4_path = convert_avi_to_mp4(clip_path)
                if mp4_path:
                    final_path = mp4_path
                    final_name = clip_name.replace('.avi', '.mp4')
                    print(f"  ✅ Converted to MP4")
                else:
                    print(f"  ⚠️  MP4 conversion failed, keeping AVI")
            
            clips.append({
                'filename': final_name,
                'path': final_path,
                'start_time': round(start_time, 2),
                'end_time': round(end_time, 2),
                'duration': round(end_time - start_time, 2),
                'type': event_type,
                'label': label,
                'priority': clip_info.get('priority', 99),
                'p1_id': clip_info.get('p1_id'),
                'p2_id': clip_info.get('p2_id'),
                'hand_type': clip_info.get('hand_type', 'N/A'),
                'confidence': clip_info.get('confidence'),
                'description': clip_info.get('description'),
                'frames': frames_written,
                'merged_count': len(clip_info['events'])  # How many events merged
            })
            
            if progress_callback:
                progress_callback(int((idx + 1) / len(merged_clips) * 100))
            
            merge_info = f" (merged {len(clip_info['events'])} detections)" if len(clip_info['events']) > 1 else ""
            print(f"  📎 Clip {idx+1}: {label} - {final_name} ({frames_written} frames){merge_info}")
        
        cap.release()
        return clips


def process_video(job_id, video_path, video_filename):
    """Process video in background thread"""
    annotated_video_path = None
    try:
        processing_status[job_id] = {
            'status': 'processing',
            'progress': 0,
            'stage': 'Initializing detector...',
            'filename': video_filename,
            'started': datetime.now().isoformat()
        }
        
        # Create extractor with full config
        extractor = TransactionClipExtractor(APP_CONFIG)
        
        # Detect transactions (creates annotated video)
        processing_status[job_id]['stage'] = 'Detecting transactions...'
        
        def detection_progress(progress):
            processing_status[job_id]['progress'] = progress // 2  # First 50%
        
        transactions, fps, annotated_video_path = extractor.detect_transactions(video_path, detection_progress)
        
        processing_status[job_id]['stage'] = f'Found {len(transactions)} transaction(s). Extracting clips...'
        processing_status[job_id]['transactions_count'] = len(transactions)
        
        if len(transactions) == 0:
            processing_status[job_id]['status'] = 'completed'
            processing_status[job_id]['progress'] = 100
            processing_status[job_id]['clips'] = []
            processing_status[job_id]['message'] = 'No transactions detected'
            # Clean up annotated video
            if annotated_video_path and os.path.exists(annotated_video_path):
                os.remove(annotated_video_path)
            return
        
        # Extract clips from annotated video (FAST!)
        output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        Path(output_folder).mkdir(exist_ok=True)
        
        def extraction_progress(progress):
            processing_status[job_id]['progress'] = 50 + (progress // 2)  # Last 50%
        
        clips = extractor.extract_clips(annotated_video_path, transactions, fps, output_folder, extraction_progress)
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['progress'] = 100
        processing_status[job_id]['clips'] = clips
        processing_status[job_id]['completed'] = datetime.now().isoformat()
        
        print(f"✅ Job {job_id} completed: {len(clips)} clips extracted")
        
    except Exception as e:
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['error'] = str(e)
        processing_status[job_id]['progress'] = 0
        print(f"❌ Job {job_id} error: {e}")
    
    finally:
        # Always clean up temporary annotated video
        if annotated_video_path and os.path.exists(annotated_video_path):
            try:
                os.remove(annotated_video_path)
                print(f"🧹 Cleaned up temporary annotated video")
            except:
                pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle multiple video uploads"""
    if 'videos' not in request.files:
        return jsonify({'error': 'No videos provided'}), 400
    
    files = request.files.getlist('videos')
    
    max_videos = app.config.get('MAX_VIDEOS', 5)
    if len(files) > max_videos:
        return jsonify({'error': f'Maximum {max_videos} videos allowed'}), 400
    
    jobs = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            job_id = str(uuid.uuid4())
            
            # Save uploaded file
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
            file.save(upload_path)
            
            # Start processing in background
            thread = threading.Thread(target=process_video, args=(job_id, upload_path, filename))
            thread.daemon = True
            thread.start()
            
            jobs.append({
                'job_id': job_id,
                'filename': filename
            })
    
    return jsonify({'jobs': jobs})


@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(processing_status[job_id])


@app.route('/download/<job_id>/<filename>')
def download_clip(job_id, filename):
    """Download or stream extracted clip"""
    output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
    
    # Determine MIME type based on file extension
    if filename.endswith('.mp4'):
        mimetype = 'video/mp4'
    elif filename.endswith('.avi'):
        mimetype = 'video/x-msvideo'
    elif filename.endswith('.webm'):
        mimetype = 'video/webm'
    else:
        mimetype = 'video/mp4'  # Default
    
    # Check if request wants to stream (for video player) or download
    # Video players send Range requests for streaming
    if request.args.get('stream') or request.headers.get('Range'):
        # Stream for browser video player (no attachment)
        return send_from_directory(output_folder, filename, 
                                   as_attachment=False,
                                   mimetype=mimetype)
    else:
        # Force download
        return send_from_directory(output_folder, filename, as_attachment=True)


@app.route('/results/<job_id>')
def view_results(job_id):
    """View results page"""
    if job_id not in processing_status:
        return "Job not found", 404
    
    return render_template('results.html', job_id=job_id, status=processing_status[job_id])


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting Flask Cash Transaction Detector")
    print("="*70)
    print("Using EXACT same detection logic as main.py")
    print("Open your browser: http://localhost:5000")
    print(f"📏 Max file size: {APP_CONFIG.get('MAX_FILE_SIZE_MB', 2048)}MB")
    print("="*70 + "\n")
    
    # Run with threaded=True for better handling of large uploads
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
