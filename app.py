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
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

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
        # Get original video FPS to preserve it
        cap = cv2.VideoCapture(avi_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Use ffmpeg to convert with H.264 codec (browser compatible, smooth playback)
        cmd = [
            'ffmpeg',
            '-i', avi_path,
            '-c:v', 'libx264',    # H.264 video codec
            '-preset', 'medium',  # Better quality encoding
            '-crf', '20',         # Higher quality (lower = better, 20 is high quality)
            '-pix_fmt', 'yuv420p', # Ensure compatibility
            '-movflags', '+faststart',  # Enable streaming
            '-r', str(int(original_fps)),  # Preserve original FPS
            '-vsync', 'cfr',      # Constant frame rate (no drops)
            '-y',                 # Overwrite output file
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
    
    def __init__(self, config_dict=None, options=None):
        self.config_dict = config_dict
        self.options = options or {}
        
        # Debug options
        self.debug_velocity = self.options.get('debug_velocity', False)
        self.show_detection_count = self.options.get('show_detection_count', True)
        
        # Initialize multi-event detector
        self.detector = MultiEventDetector(config_dict)
        
        # Clip extraction settings from config
        self.SECONDS_BEFORE = config_dict.get('SECONDS_BEFORE_TRANSACTION', 2)
        self.SECONDS_AFTER = config_dict.get('SECONDS_AFTER_TRANSACTION', 2)
        self.MERGE_THRESHOLD = config_dict.get('MERGE_CLIPS_WITHIN_SECONDS', 0.5)
        
        self.detection_types = config_dict.get('DETECTION_TYPES', {})
        self.video_extension = '.avi'  # Will be set based on available codec
        
        print(f"⏱️  Clip padding: {self.SECONDS_BEFORE}s before, {self.SECONDS_AFTER}s after")
        print(f"🔗 Merge clips within: {self.MERGE_THRESHOLD}s")
        if self.debug_velocity:
            print(f"🐛 Velocity visualization: ENABLED")
        else:
            print(f"🐛 Velocity visualization: DISABLED")
        if self.show_detection_count:
            print(f"📊 Detection counting: ENABLED")
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
        
        # Handle time interval
        start_frame = 0
        end_frame = total_frames
        
        if not self.options.get('full_video', True):
            start_time = self.options.get('start_time', 0)
            end_time = self.options.get('end_time', None)
            
            start_frame = int(start_time * fps)
            if end_time:
                end_frame = min(int(end_time * fps), total_frames)
            else:
                end_frame = total_frames  # Process until end if no end_time specified
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Calculate actual frames to process
        frames_to_process = end_frame - start_frame
        duration_seconds = frames_to_process / fps
        
        # SKIP temporary video for large files - process on demand instead
        print(f"⚡ Using on-demand processing (no temp video)")
        print(f"   This is faster for large videos and uses less disk space")
        temp_annotated_path = None  # No temp video
        self.video_extension = '.mp4'  # Will convert later
        
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
                'cash_types': {},
                'possible_detections': 0
            }
            # Reset possible detection tracking
            cash_detector.possible_events = []
            cash_detector.current_possible_event = None
        
        all_events = []  # Store all detected events
        current_events = {}  # Track ongoing events by type
        frame_num = start_frame  # Start from interval start
        
        # Process EVERY frame for smooth detection (no skipping)
        frame_skip = 1  # Process all frames for smooth output
        effective_frames = frames_to_process
        
        # Print analysis info
        if not self.options.get('full_video', True):
            print(f"⏱️  TIME INTERVAL: {start_frame/fps:.1f}s - {end_frame/fps:.1f}s")
            print(f"📹 Analyzing INTERVAL: {frames_to_process} frames at {fps} FPS")
            print(f"⏱️  Interval duration: {duration_seconds/60:.1f} minutes")
        else:
            print(f"📹 Analyzing FULL video: {frames_to_process} frames at {fps} FPS")
            print(f"⏱️  Estimated duration: {duration_seconds/60:.1f} minutes")
        
        print(f"🎬 Processing ALL frames for smooth detection (no frame skipping)")
        
        # Debug counters
        hands_close_count = 0
        cash_seen_count = 0
        violence_count = 0
        fire_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame_pos < end_frame * 0.9:  # Less than 90% of target processed
                    print(f"\n⚠️  Video reading stopped early at frame {current_frame_pos}/{end_frame}")
                break
            
            frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Stop if we've reached end_frame
            if frame_num >= end_frame:
                break
            
            # Progress update every 150 frames (every 10 seconds at 15fps)
            if progress_callback and frame_num % 150 == 0:
                frames_in_range = end_frame - start_frame
                progress_pct = int((frame_num - start_frame) / frames_in_range * 100)
                progress_callback(progress_pct)
                print(f"  ⚡ Progress: {frame_num - start_frame}/{frames_in_range} frames ({progress_pct}%)")
            
            try:
                # Run detection (no need to save output_frame since we don't store temp video)
                _, detections_dict = self.detector.detect_all(frame, fps=fps)
            except Exception as e:
                print(f"⚠️  Error processing frame {frame_num}: {e}")
                continue
            
            # Process each detection type
            for det_type, detections in detections_dict.items():
                if det_type == 'CASH_EXCHANGE':
                    # Handle cash transactions
                    if detections:
                        hands_close_count += 1
                    for trans in detections:
                        if trans.get('cash_detected'):
                            cash_seen_count += 1
                        
                        # Check if this is actually violence (fast movement or long duration)
                        is_violence = trans.get('is_violence', False)
                        
                        trans_key = f"{trans['p1_id']}-{trans['p2_id']}"
                        
                        # If violence flagged, treat as violence event
                        if is_violence:
                            violence_count += 1  # Count violence detections
                            event_key = f"VIOLENCE_{trans_key}"
                            if event_key not in current_events:
                                current_events[event_key] = {
                                    'type': 'VIOLENCE',
                                    'key': trans_key,
                                    'start_frame': frame_num,
                                    'end_frame': frame_num,
                                    'start_time': frame_num / fps,
                                    'end_time': frame_num / fps,
                                    'p1_id': trans['p1_id'],
                                    'p2_id': trans['p2_id'],
                                    'description': trans.get('cash_type', 'Fast Movement'),
                                    'velocities': trans.get('velocities', {})
                                }
                            else:
                                current_events[event_key]['end_frame'] = frame_num
                                current_events[event_key]['end_time'] = frame_num / fps
                        else:
                            # Normal cash transaction
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
                                
                                # Check if event is too long (>5 seconds) - reclassify as violence
                                duration = current_events[event_key]['end_time'] - current_events[event_key]['start_time']
                                max_cash_duration = self.config_dict.get('MAX_CASH_TRANSACTION_SECONDS', 5)
                                if duration > max_cash_duration:
                                    print(f"  ⚠️  🚨 RECLASSIFYING as VIOLENCE: Transaction too long ({duration:.1f}s > {max_cash_duration}s)")
                                    # Change type to violence
                                    current_events[event_key]['type'] = 'VIOLENCE'
                                    current_events[event_key]['description'] = f'Extended interaction ({duration:.1f}s)'
                
                else:
                    # Handle violence/fire detections
                    for detection in detections:
                        # Count detections
                        if det_type == 'FIRE':
                            fire_count += 1
                        elif det_type == 'VIOLENCE':
                            violence_count += 1
                        
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
        
        # Merge close events
        merged_events = self._merge_close_events(all_events, fps)
        
        print(f"\n" + "="*70)
        print(f"📊 DETECTION SUMMARY REPORT")
        print(f"="*70)
        print(f"  Total frames analyzed: {frame_num - start_frame}")
        if self.show_detection_count:
            print(f"\n🔍 FRAME-BY-FRAME DETECTIONS:")
            print(f"  🚨 VIOLENCE detections: {violence_count} frames")
            print(f"  🔥 FIRE detections: {fire_count} frames")
            print(f"  💵 CASH detections: {cash_seen_count} frames")
            print(f"  🤝 Hands close: {hands_close_count} frames")
        print(f"\n✅ Detection complete: Found {len(merged_events)} event(s)")
        print(f"="*70)
        
        if len(merged_events) == 0:
            print(f"\n⚠️  NO EVENTS DETECTED - Check 'Possible' folder for hand-close events")
            if hands_close_count == 0:
                print(f"     ❌ No hands detected close enough (<{cash_detector.config.HAND_TOUCH_DISTANCE}px)")
                print(f"     💡 Try: Increase HAND_TOUCH_DISTANCE in config")
            elif cash_seen_count == 0:
                print(f"     ❌ Hands were close but NO CASH/MATERIAL detected")
                print(f"     💡 Try: Set DETECT_CASH_COLOR to false (currently: {cash_detector.config.DETECT_CASH_COLOR})")
                print(f"     💡 Or: Lower CASH_DETECTION_CONFIDENCE (currently: {cash_detector.config.CASH_DETECTION_CONFIDENCE})")
            print()
        else:
            for event in merged_events:
                det_type = event.get('type', 'UNKNOWN')
                label = self.detection_types.get(det_type, {}).get('label', det_type)
                print(f"  • {label}: {event['start_time']:.1f}s - {event['end_time']:.1f}s")
        
        # Note: Possible detections will be saved later when we know the output folder location
        # (We can't save here because we don't know the job output folder yet)
        
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
    
    def extract_clips(self, source_video_path, events, fps, output_folder, progress_callback=None):
        """Extract video clips by re-processing only needed frames"""
        # Open original source video
        cap = cv2.VideoCapture(source_video_path)
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
            video_name = Path(source_video_path).stem
            event_type = clip_info.get('type', 'CASH_EXCHANGE')
            label = self.detection_types.get(event_type, {}).get('label', event_type)
            
            # Generate filename
            base_name = f"{video_name}_{event_type}"
            if event_type == 'CASH_EXCHANGE' and clip_info.get('p1_id'):
                base_name += f"_P{clip_info['p1_id']}_P{clip_info['p2_id']}"
            base_name += f"_{int(start_time)}s"
            clip_name_avi = base_name + '.avi'
            clip_path_avi = os.path.join(output_folder, clip_name_avi)
            
            # Use MJPEG AVI with high quality for smooth playback
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(clip_path_avi, fourcc, fps, (width, height))
            
            # Verify video writer was created successfully
            if not out.isOpened():
                # Try alternative codec
                print(f"  ⚠️  MJPG codec failed, trying XVID...")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(clip_path_avi, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"⚠️  Warning: Could not create video writer for clip {idx+1}")
                print(f"  Tried codecs: MJPG, XVID")
                print(f"  Target: {width}x{height} @ {fps}fps")
                continue
            
            print(f"  ✅ Video writer ready: {width}x{height} @ {fps}fps")
            
            # Reset detector state
            if 'CASH_EXCHANGE' in self.detector.detectors:
                cash_detector = self.detector.detectors['CASH_EXCHANGE']
                cash_detector.transaction_history = {}
                cash_detector.person_id_map = {}
                cash_detector.next_stable_id = 1
                cash_detector.cashier_persistence = {}
            
            # Seek to start frame and process
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_written = 0
            clip_debug_data = {
                "clip_info": {
                    "filename": base_name,
                    "start_time": f"{start_time:.2f}s",
                    "end_time": f"{end_time:.2f}s",
                    "duration": f"{end_time - start_time:.2f}s",
                    "fps": fps,
                    "event_type": event_type,
                    "label": label
                },
                "frames": []
            }
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    print(f"  ⚠️  Frame read failed at {frame_idx}, stopping clip")
                    break
                
                # Verify frame is valid
                if frame is None or frame.size == 0:
                    print(f"  ⚠️  Invalid frame at {frame_idx}, skipping")
                    continue
                
                # Re-run detector on this frame to get annotations
                try:
                    annotated_frame, detections_dict = self.detector.detect_all(frame, fps=fps)
                except Exception as e:
                    print(f"  ⚠️  Detection error at frame {frame_idx}: {e}")
                    annotated_frame = frame  # Use original frame if detection fails
                
                # Collect debug data from this frame (if available)
                if self.config_dict.get('DEBUG_MODE', False):
                    frame_data = {
                        "frame_number": frame_idx,
                        "timestamp": f"{frame_idx/fps:.2f}s",
                        "detections": {}
                    }
                    
                    # Extract debug data from detections
                    for det_type, detections in detections_dict.items():
                        if det_type == 'CASH_EXCHANGE' and detections:
                            for det in detections:
                                # Check if this detection has debug data
                                if 'analysis_scores' in det and 'debug_data' in det['analysis_scores']:
                                    frame_data['detections'][det_type] = det['analysis_scores']['debug_data']
                                    break  # Only store first detection per frame
                    
                    if frame_data['detections']:  # Only add frames with detections
                        clip_debug_data['frames'].append(frame_data)
                
                # Ensure frame is in correct format and size before writing
                if annotated_frame.shape[:2] != (height, width):
                    annotated_frame = cv2.resize(annotated_frame, (width, height))
                
                # Write annotated frame
                out.write(annotated_frame)
                frames_written += 1
            
            out.release()
            
            # Save debug data as JSON file
            if self.config_dict.get('DEBUG_MODE', False) and clip_debug_data['frames']:
                json_filename = base_name + '_debug.json'
                json_path = os.path.join(output_folder, json_filename)
                try:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(clip_debug_data, f, indent=2, ensure_ascii=False)
                    print(f"  📊 Saved debug data: {json_filename}")
                except Exception as e:
                    print(f"  ⚠️  Failed to save debug JSON: {e}")
            
            # Convert AVI to browser-compatible MP4 if ffmpeg is available
            final_path = clip_path_avi
            final_name = clip_name_avi
            if FFMPEG_AVAILABLE:
                print(f"  🔄 Converting to MP4...")
                mp4_path = convert_avi_to_mp4(clip_path_avi)
                if mp4_path:
                    final_path = mp4_path
                    final_name = clip_name_avi.replace('.avi', '.mp4')
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


def process_video(job_id, video_path, video_filename, options=None):
    """Process video in background thread"""
    annotated_video_path = None
    if options is None:
        options = {}
    
    try:
        processing_status[job_id] = {
            'status': 'processing',
            'progress': 0,
            'stage': 'Initializing detector...',
            'filename': video_filename,
            'started': datetime.now().isoformat(),
            'options': options
        }
        
        # Create extractor with full config
        extractor = TransactionClipExtractor(APP_CONFIG, options)
        
        # Detect transactions (creates annotated video)
        time_range = ""
        if not options.get('full_video', True):
            start = options.get('start_time', 0)
            end = options.get('end_time', 'end')
            time_range = f" ({start}s - {end}s)"
        
        processing_status[job_id]['stage'] = f'Detecting transactions{time_range}...'
        
        def detection_progress(progress):
            # Detection is now the main task (90% of time), extraction is fast (10%)
            processing_status[job_id]['progress'] = int(progress * 0.9)  # 0-90%
        
        transactions, fps, annotated_video_path = extractor.detect_transactions(video_path, detection_progress)
        
        processing_status[job_id]['stage'] = f'Found {len(transactions)} transaction(s). Extracting clips...'
        processing_status[job_id]['transactions_count'] = len(transactions)
        
        # Create output folder (needed for possible detections even if no clips)
        output_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id)
        Path(output_folder).mkdir(exist_ok=True)
        
        # ALWAYS save possible detections (even if no confirmed transactions)
        if 'CASH_EXCHANGE' in extractor.detector.detectors:
            cash_detector = extractor.detector.detectors['CASH_EXCHANGE']
            if hasattr(cash_detector, 'possible_events') and (cash_detector.possible_events or cash_detector.current_possible_event):
                try:
                    print(f"\n💾 Saving possible detections (JSON + Video clips) to output folder...")
                    cash_detector._save_possible_detections(video_path, output_folder, fps)
                    processing_status[job_id]['has_possible_detections'] = True
                    print(f"  ✅ Possible detections saved with video clips")
                except Exception as e:
                    print(f"  ⚠️  Failed to save possible detections: {e}")
                    import traceback
                    traceback.print_exc()
        
        if len(transactions) == 0:
            processing_status[job_id]['status'] = 'completed'
            processing_status[job_id]['progress'] = 100
            processing_status[job_id]['clips'] = []
            processing_status[job_id]['message'] = 'No transactions detected'
            return
        
        # Extract clips from original video (re-process frames on-demand)
        def extraction_progress(progress):
            processing_status[job_id]['progress'] = 90 + int(progress * 0.1)  # Last 10%
        
        # Use original video path instead of non-existent annotated path
        clips = extractor.extract_clips(video_path, transactions, fps, output_folder, extraction_progress)
        
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
    
    # Get time interval and debug options
    full_video_str = request.form.get('full_video', 'true')
    full_video = full_video_str.lower() == 'true'
    
    # Handle time inputs (only if not full video)
    start_time = None
    end_time = None
    if not full_video:
        start_time_str = request.form.get('start_time', '')
        end_time_str = request.form.get('end_time', '')
        
        if start_time_str:
            start_time = float(start_time_str)
        else:
            start_time = 0  # Default to start of video
        
        if end_time_str:
            end_time = float(end_time_str)
        else:
            end_time = None  # Default to end of video
    
    debug_velocity = request.form.get('debug_velocity', 'false').lower() == 'true'
    show_detection_count = request.form.get('show_detection_count', 'true').lower() == 'true'
    
    # Debug: Print received options
    print(f"\n📋 UPLOAD OPTIONS:")
    print(f"   full_video: {full_video_str} → {full_video}")
    if not full_video:
        print(f"   start_time: {start_time}s")
        print(f"   end_time: {end_time}s (None = until end)")
    print(f"   debug_velocity: {debug_velocity}")
    print(f"   show_detection_count: {show_detection_count}\n")
    
    jobs = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            job_id = str(uuid.uuid4())
            
            # Save uploaded file
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
            file.save(upload_path)
            
            # Processing options
            process_options = {
                'full_video': full_video,
                'start_time': start_time,
                'end_time': end_time,
                'debug_velocity': debug_velocity,
                'show_detection_count': show_detection_count
            }
            
            # Start processing in background
            thread = threading.Thread(target=process_video, args=(job_id, upload_path, filename, process_options))
            thread.daemon = True
            thread.start()
            
            jobs.append({
                'job_id': job_id,
                'filename': filename,
                'options': process_options
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
    elif filename.endswith('.json'):
        mimetype = 'application/json'
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


@app.route('/possible/<job_id>')
def list_possible_detections(job_id):
    """List possible detection files (JSON + video) for a job"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    possible_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id, 'Possible')
    
    if not os.path.exists(possible_folder):
        return jsonify({'possible_detections': []})
    
    # Group JSON and video files together
    detections = []
    for filename in os.listdir(possible_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(possible_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Find matching video file (same name but .mp4 or .avi)
                base_name = filename.replace('.json', '')
                video_file = None
                for ext in ['.mp4', '.avi']:
                    video_path = os.path.join(possible_folder, base_name + ext)
                    if os.path.exists(video_path):
                        video_file = base_name + ext
                        break
                
                detections.append({
                    'filename': filename,
                    'video_filename': video_file,
                    'group_id': data.get('group_id'),
                    'event_count': data.get('event_count'),
                    'time_range': data.get('time_range'),
                    'json_url': f'/download/{job_id}/Possible/{filename}',
                    'video_url': f'/download/{job_id}/Possible/{video_file}?stream=true' if video_file else None
                })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    # Sort by group_id
    detections.sort(key=lambda x: x.get('group_id', 0))
    
    return jsonify({'possible_detections': detections, 'total': len(detections)})


@app.route('/download/<job_id>/Possible/<filename>')
def download_possible_file(job_id, filename):
    """Download or stream a possible detection file (JSON or video)"""
    possible_folder = os.path.join(app.config['OUTPUT_FOLDER'], job_id, 'Possible')
    
    if not os.path.exists(os.path.join(possible_folder, filename)):
        return "File not found", 404
    
    # Determine MIME type
    if filename.endswith('.mp4'):
        mimetype = 'video/mp4'
    elif filename.endswith('.avi'):
        mimetype = 'video/x-msvideo'
    elif filename.endswith('.json'):
        mimetype = 'application/json'
    else:
        mimetype = 'application/octet-stream'
    
    # Stream video files, download JSON files
    if request.args.get('stream') or request.headers.get('Range') or filename.endswith('.mp4') or filename.endswith('.avi'):
        return send_from_directory(possible_folder, filename, 
                                   as_attachment=False,
                                   mimetype=mimetype)
    else:
        return send_from_directory(possible_folder, filename, as_attachment=True)


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
