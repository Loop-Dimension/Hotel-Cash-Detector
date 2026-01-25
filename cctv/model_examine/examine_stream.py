"""
CCTV Model Examine Stream

Real-time CCTV streaming with YOLO detection overlay.
Split view: Left = Original, Right = Detection overlay

Features:
- Zone configuration (Cashier Zone, Cash Drawer Zone)
- 2-track cash detection (potential_cash + cash)
- Violence detection
- Fire detection
- Auto clip recording on detection (rolling buffer, 12s)
- Dual clip save: original + overlay

Usage:
    python -m cctv.model_examine.examine_stream --rtsp "rtsp://..." --zone-config zones.json
    python -m cctv.model_examine.examine_stream --video test.mp4 --setup-zones
"""

import argparse
import cv2
import numpy as np
import time
import json
import threading
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import deque

# Import detectors
try:
    from .detectors import UnifiedDetector, Detection
except ImportError:  # Support direct execution without -m
    from detectors import UnifiedDetector, Detection
try:
    from .detectors.gemini_validator import GeminiValidator
except Exception:
    GeminiValidator = None


class ClipRecorder:
    """Records detection clips using a rolling buffer.

    Captures pre_buffer_sec BEFORE event and post_buffer_sec AFTER event.
    Total clip duration = pre_buffer_sec + post_buffer_sec (default: 6+6=12s)

    Uses a queue-based approach to avoid blocking the main thread.
    """

    def __init__(
        self,
        output_dir: str,
        stream_fps: float,
        buffer_stride: int = 4,  # 4프레임마다 1개 저장 (30fps → 7.5fps)
        pre_buffer_sec: float = 6.0,
        post_buffer_sec: float = 6.0,
        gemini_validator=None,
        gemini_source: str = "original",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stream_fps = stream_fps if stream_fps > 0 else 30
        self.buffer_stride = max(1, int(buffer_stride))
        self.buffer_fps = max(1.0, self.stream_fps / self.buffer_stride)
        self.pre_buffer_sec = pre_buffer_sec
        self.post_buffer_sec = post_buffer_sec
        self.gemini_validator = gemini_validator
        self.gemini_source = gemini_source

        # Frame queue for background processing (main thread just enqueues)
        self._frame_queue = deque(maxlen=300)  # ~40s at 7.5fps (stride=4)
        self._frame_queue_lock = threading.Lock()

        # Buffer built by background thread
        pre_buffer_frames = int(round(self.pre_buffer_sec * self.buffer_fps))
        # Store overlay frames only to reduce memory usage
        self._buffer_overlay = deque(maxlen=max(1, pre_buffer_frames))
        self._buffer_lock = threading.Lock()

        # Pending events
        self._pending_events = deque()
        self._pending_lock = threading.Lock()

        # Event triggers queue (main thread enqueues, worker processes)
        self._trigger_queue = deque()
        self._trigger_lock = threading.Lock()

        self._stop_threads = False

        # Background thread for frame processing (copies frames from queue to buffer)
        self._frame_thread = threading.Thread(target=self._frame_worker, daemon=True)
        self._frame_thread.start()

        # Background thread for saving clips
        self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self._save_thread.start()

    def stop(self):
        """Stop background threads"""
        self._stop_threads = True
        if self._frame_thread and self._frame_thread.is_alive():
            self._frame_thread.join(timeout=2)
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join(timeout=5)

    def add_frame(self, frame: np.ndarray, frame_count: int):
        """Enqueue frame for background processing (non-blocking).

        IMPORTANT: Makes an immediate copy to prevent race condition.
        The main thread may overwrite the original numpy array before
        the worker thread processes it.
        """
        if frame_count % self.buffer_stride != 0:
            return

        # CRITICAL: Copy immediately to prevent race condition
        # Without this, the main thread may overwrite the frame before worker copies it
        frame_copy = frame.copy()

        with self._frame_queue_lock:
            self._frame_queue.append((frame_copy, frame_count))

    def trigger_event(self, detection: Detection):
        """Queue event trigger for background processing (non-blocking)."""
        with self._trigger_lock:
            self._trigger_queue.append((detection, time.time()))
        print(f"[Clip] Event queued: {detection.label} ({detection.event_type or detection.label.lower()})")

    def _frame_worker(self):
        """Background worker that processes frames and handles triggers."""
        while not self._stop_threads:
            # Process pending triggers first
            self._process_triggers()

            # Process frames from queue
            frame_data = None
            with self._frame_queue_lock:
                if self._frame_queue:
                    frame_data = self._frame_queue.popleft()

            if frame_data is None:
                time.sleep(0.005)  # 5ms sleep if no frames
                continue

            # Frame is already copied in add_frame() - no need to copy again
            frame, _ = frame_data

            # Add to pre-buffer (frame already copied, safe to use directly)
            with self._buffer_lock:
                self._buffer_overlay.append(frame)

            # Add to post-buffers of pending events
            with self._pending_lock:
                for event in self._pending_events:
                    if event.get('collecting_post', False):
                        event['post_overlay'].append(frame)

    def _process_triggers(self):
        """Process queued event triggers."""
        triggers_to_process = []
        with self._trigger_lock:
            while self._trigger_queue:
                triggers_to_process.append(self._trigger_queue.popleft())

        for detection, trigger_time in triggers_to_process:
            event_type = detection.event_type or detection.label.lower()

            with self._pending_lock:
                # Check if already processing this event type
                already_processing = False
                for event in self._pending_events:
                    existing_type = event['detection'].event_type or event['detection'].label.lower()
                    if existing_type == event_type and event.get('collecting_post', False):
                        already_processing = True
                        break

                if already_processing:
                    continue

                # Snapshot pre-buffer
                with self._buffer_lock:
                    pre_overlay = list(self._buffer_overlay)

                post_frames_needed = int(round(self.post_buffer_sec * self.buffer_fps))

                self._pending_events.append({
                    'detection': detection,
                    'pre_overlay': pre_overlay,
                    'post_overlay': [],
                    'post_frames_needed': post_frames_needed,
                    'trigger_time': trigger_time,
                    'collecting_post': True,
                })
                print(f"[Clip] Event triggered: {detection.label} ({event_type}) - collecting {self.post_buffer_sec}s post-buffer")

    def _save_worker(self):
        """Background worker that saves clips when post-buffer is collected."""
        while not self._stop_threads:
            ready_event = None

            with self._pending_lock:
                # Find an event that has finished collecting post-buffer
                for i, event in enumerate(self._pending_events):
                    if event.get('collecting_post', False):
                        post_collected = len(event['post_overlay'])
                        post_needed = event.get('post_frames_needed', 0)
                        if post_collected >= post_needed:
                            # Ready to save
                            ready_event = self._pending_events[i]
                            del self._pending_events[i]
                            break

            if ready_event is None:
                time.sleep(0.1)
                continue

            # Combine pre-buffer + post-buffer
            overlay_frames = ready_event['pre_overlay'] + ready_event['post_overlay'][:ready_event['post_frames_needed']]

            # Validate frames
            valid_frames = [f for f in overlay_frames if f is not None and f.size > 0]
            min_frames = int(round((self.pre_buffer_sec + self.post_buffer_sec) * self.buffer_fps * 0.5))  # At least 50% of expected
            if len(valid_frames) < min_frames:
                print(f"[Clip] Not enough valid frames ({len(valid_frames)}/{min_frames}), skipping")
                continue

            try:
                self._save_clip_sync(ready_event['detection'], overlay_frames)
            except Exception as e:
                print(f"[Clip] Save error: {e}")

    def _write_video(self, frames: List[np.ndarray], filepath: Path, fps: float,
                     detection: Detection):
        """Write frames to a video file with label overlay."""
        if not frames:
            return
        h, w = frames[0].shape[:2]
        temp_path = filepath.with_suffix('.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(temp_path), fourcc, fps, (w, h))

        colors = {
            'CASH': (0, 255, 255),
            'VIOLENCE': (0, 0, 255),
            'FIRE': (0, 165, 255),
            'SMOKE': (128, 128, 128)
        }
        color = colors.get(detection.label, (255, 255, 255))
        label = f"{detection.label} - {detection.event_type or 'detected'}"

        for frame in frames:
            overlay_frame = frame.copy()
            cv2.rectangle(overlay_frame, (10, 10), (350, 45), (0, 0, 0), -1)
            cv2.putText(overlay_frame, label, (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            out.write(overlay_frame)

        out.release()
        subprocess.run([
            'ffmpeg', '-y', '-i', str(temp_path),
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-r', str(int(round(fps))),
            str(filepath)
        ], capture_output=True, timeout=120)
        temp_path.unlink(missing_ok=True)

    def _save_clip_sync(self, detection: Detection,
                        overlay_frames: List[np.ndarray]):
        """Save clip to file (runs in background thread)"""
        if not overlay_frames:
            print(f"[Clip] No frames to save")
            return

        timestamp = detection.timestamp.strftime('%Y%m%d_%H%M%S')
        event_type = detection.event_type or detection.label.lower()
        base_name = f"{timestamp}_{event_type}"
        overlay_path = self.output_dir / f"{base_name}_overlay.mp4"

        try:
            self._write_video(overlay_frames, overlay_path, self.buffer_fps, detection)
            print(f"[Clip] Saved: {overlay_path.name} ({len(overlay_frames)} frames)")
        except subprocess.TimeoutExpired:
            print(f"[Clip] FFmpeg timeout, keeping AVI")
        except Exception as e:
            print(f"[Clip] FFmpeg failed: {e}")
            return

        if self.gemini_validator:
            try:
                video_path = overlay_path
                is_valid, confidence, reason, corrected = self.gemini_validator.validate_event_video(
                    str(video_path), event_type
                )

                # API 에러 시 파일명 변경 안함 (평가 실패)
                if "API error" in reason or "error" in reason.lower():
                    print(f"[Clip] Gemini API error, keeping original name: {reason}")
                elif not is_valid:
                    fp_overlay = overlay_path.with_name(f"FP_{overlay_path.name}")
                    overlay_path.rename(fp_overlay)
                    print(f"[Clip] Gemini rejected: renamed to FP_ (conf={confidence:.2f}) {reason}")
                else:
                    tp_overlay = overlay_path.with_name(f"TP_{overlay_path.name}")
                    overlay_path.rename(tp_overlay)
                    print(f"[Clip] Gemini accepted: renamed to TP_ (conf={confidence:.2f}) {reason}")
            except Exception as e:
                print(f"[Clip] Gemini validation error: {e}")


class ZoneConfigurator:
    """Interactive zone configuration tool"""

    def __init__(self, frame: np.ndarray):
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        self.cashier_points = []
        self.drawer_points = []
        self.current_zone = 'cashier'
        self.done = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_zone == 'cashier':
                self.cashier_points.append([x, y])
            else:
                self.drawer_points.append([x, y])
            self._redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_zone == 'cashier' and self.cashier_points:
                self.cashier_points.pop()
            elif self.current_zone == 'drawer' and self.drawer_points:
                self.drawer_points.pop()
            self._redraw()

    def _redraw(self):
        self.frame = self.original_frame.copy()
        h, w = self.frame.shape[:2]

        # Draw cashier zone
        if len(self.cashier_points) >= 2:
            pts = np.array(self.cashier_points, dtype=np.int32)
            if len(self.cashier_points) >= 3:
                overlay = self.frame.copy()
                cv2.fillPoly(overlay, [pts.reshape((-1, 1, 2))], (255, 200, 0))
                cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)
            cv2.polylines(self.frame, [pts.reshape((-1, 1, 2))], True, (255, 200, 0), 2)

        for i, pt in enumerate(self.cashier_points):
            cv2.circle(self.frame, tuple(pt), 6, (255, 200, 0), -1)
            cv2.putText(self.frame, f"C{i+1}", (pt[0]+8, pt[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw drawer zone
        if len(self.drawer_points) >= 2:
            pts = np.array(self.drawer_points, dtype=np.int32)
            if len(self.drawer_points) >= 3:
                overlay = self.frame.copy()
                cv2.fillPoly(overlay, [pts.reshape((-1, 1, 2))], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)
            cv2.polylines(self.frame, [pts.reshape((-1, 1, 2))], True, (0, 255, 0), 2)

        for i, pt in enumerate(self.drawer_points):
            cv2.circle(self.frame, tuple(pt), 6, (0, 255, 0), -1)
            cv2.putText(self.frame, f"D{i+1}", (pt[0]+8, pt[1]+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Instructions
        cv2.rectangle(self.frame, (0, 0), (w, 80), (0, 0, 0), -1)
        mode_color = (255, 200, 0) if self.current_zone == 'cashier' else (0, 255, 0)
        mode_text = "CASHIER ZONE" if self.current_zone == 'cashier' else "DRAWER ZONE"

        cv2.putText(self.frame, f"Mode: {mode_text}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
        cv2.putText(self.frame, "Left: Add | Right: Remove | TAB: Switch | S: Save | Q: Quit",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(self.frame, f"Cashier: {len(self.cashier_points)} pts | Drawer: {len(self.drawer_points)} pts",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def run(self) -> Tuple[List, List]:
        """Run zone configuration UI"""
        cv2.namedWindow("Zone Configuration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Zone Configuration", self.mouse_callback)
        self._redraw()

        while not self.done:
            cv2.imshow("Zone Configuration", self.frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('\t') or key == 9:
                self.current_zone = 'drawer' if self.current_zone == 'cashier' else 'cashier'
                self._redraw()
            elif key == ord('s'):
                self.done = True
            elif key == ord('r'):
                self.cashier_points = []
                self.drawer_points = []
                self._redraw()

        cv2.destroyWindow("Zone Configuration")
        return self.cashier_points, self.drawer_points


def run_stream(source: str, zone_config: str = None, setup_zones: bool = False,
               output_dir: str = None, detect_cash: bool = True,
               detect_violence: bool = True, detect_fire: bool = True,
               hand_distance: int = 100, global_cash_cooldown: float = None,
               cash_cooldown: float = None, cash_predict_cooldown: float = None,
               violence_cooldown: float = None, fire_cooldown: float = None,
               zone_profile: int = None, gemini_api_key: str = None,
               gemini_enabled: bool = False, gemini_source: str = "original"):
    """
    Main streaming function

    Args:
        source: RTSP URL or video file path
        zone_config: Path to zone configuration JSON
        setup_zones: Whether to run zone setup first
        output_dir: Directory for clip recordings
        detect_cash: Enable cash detection
        detect_violence: Enable violence detection
        detect_fire: Enable fire detection
        hand_distance: Hand touch distance threshold
    """
    print(f"\n{'='*60}")
    print("CCTV Model Examine Stream")
    print(f"{'='*60}")
    print(f"Source: {source}")
    print(f"{'='*60}\n")

    # Open video source
    is_live_stream = source.startswith('rtsp://') or source.startswith('http://')
    if is_live_stream:
        # RTSP options for low latency streaming:
        # - rtsp_transport;tcp: Use TCP for reliable delivery
        # - fflags;nobuffer: Reduce buffering
        # - flags;low_delay: Minimize latency
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay'
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        # Buffer size 2: balance between latency and stability
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open source: {source}")
        return

    # Get properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live_stream else 0
    duration_sec = (total_frames / fps) if (not is_live_stream and fps > 0) else 0
    print(f"Stream: {width}x{height} @ {fps}fps")

    # Resolve zone config file
    if zone_profile is not None:
        zone_config = f"zones_{zone_profile}.json"

    # Load or setup zones
    cashier_zone = None
    cash_drawer_zone = None

    if zone_config and Path(zone_config).exists():
        with open(zone_config, 'r') as f:
            config = json.load(f)
            cashier_zone = config.get('cashier_zone')
            cash_drawer_zone = config.get('cash_drawer_zone')
            print(f"Loaded zones from: {zone_config}")

    if setup_zones:
        ret, frame = cap.read()
        if ret:
            configurator = ZoneConfigurator(frame)
            cashier_zone, cash_drawer_zone = configurator.run()

            # Save zones
            save_path = zone_config or "zones.json"
            with open(save_path, 'w') as f:
                json.dump({
                    'cashier_zone': cashier_zone,
                    'cash_drawer_zone': cash_drawer_zone
                }, f, indent=2)
            print(f"Saved zones to: {save_path}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize detector
    models_dir = Path(__file__).parent / "models"
    detect_every = 4
    detect_fps = max(1.0, fps / detect_every)

    def seconds_to_frames(seconds: float) -> int:
        return max(1, int(round(seconds * detect_fps)))

    # Initialize Gemini validator (optional)
    gemini_validator = None
    if gemini_enabled:
        if GeminiValidator:
            try:
                gemini_validator = GeminiValidator(api_key=gemini_api_key)
            except Exception as e:
                print(f"[Gemini] Validator init failed: {e}")
        else:
            print("[Gemini] Validator unavailable (missing dependency)")

    detector_config = {
        'models_dir': str(models_dir),
        'device': 'auto',
        'detect_cash': detect_cash,
        'detect_violence': detect_violence,
        'detect_fire': detect_fire,
        'cashier_zone': cashier_zone,
        'cash_drawer_zone': cash_drawer_zone,
        'hand_touch_distance': hand_distance,
        'detection_mode': 'strict' if cashier_zone else 'simple',
        'enable_logs': True,
        'gemini_enabled': gemini_enabled,
        'gemini_api_key': gemini_api_key,
        'gemini_validator': gemini_validator
    }
    # Global cash cooldown (all tracks share this)
    if global_cash_cooldown is not None:
        detector_config['global_cash_cooldown'] = seconds_to_frames(global_cash_cooldown)
        # Also set individual cooldowns to the same value for consistency
        detector_config['potential_cash_cooldown'] = seconds_to_frames(global_cash_cooldown)
        detector_config['transaction_cooldown'] = seconds_to_frames(global_cash_cooldown)
        detector_config['cash_object_cooldown'] = seconds_to_frames(global_cash_cooldown)
    else:
        # Legacy: individual cooldowns (deprecated)
        if cash_predict_cooldown is not None:
            detector_config['potential_cash_cooldown'] = seconds_to_frames(cash_predict_cooldown)
        if cash_cooldown is not None:
            detector_config['transaction_cooldown'] = seconds_to_frames(cash_cooldown)
    if violence_cooldown is not None:
        detector_config['violence_cooldown'] = seconds_to_frames(violence_cooldown)
    if fire_cooldown is not None:
        detector_config['fire_cooldown'] = seconds_to_frames(fire_cooldown)

    detector = UnifiedDetector(detector_config)

    if not detector.initialize():
        print("Error: Failed to initialize detector")
        cap.release()
        return

    # Initialize clip recorder (live streams only)
    clip_recorder = None
    full_video_validation_enabled = (
        (not is_live_stream)
        and gemini_enabled
        and gemini_validator is not None
        and duration_sec > 0
        and duration_sec <= 20
    )
    validated_event_types = set()

    if is_live_stream:
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent / "model_examine_record")
        clip_recorder = ClipRecorder(
            output_dir=output_dir,
            stream_fps=fps,
            buffer_stride=2,
            pre_buffer_sec=6.0,   # 6 seconds before event
            post_buffer_sec=6.0,  # 6 seconds after event
            gemini_validator=gemini_validator,
            gemini_source=gemini_source,
        )
        print(f"Clip output directory: {output_dir}")
        print(f"Clip buffer fps: {clip_recorder.buffer_fps:.1f}")
        print(f"Clip duration: {clip_recorder.pre_buffer_sec}s before + {clip_recorder.post_buffer_sec}s after = {clip_recorder.pre_buffer_sec + clip_recorder.post_buffer_sec}s total")
    else:
        print("Clip recording disabled for video files.")
        if full_video_validation_enabled:
            print(f"[Gemini] Full video validation enabled ({duration_sec:.1f}s)")
    print("\nControls: Q=Quit, P=Pause, Z=Setup Zones, 1=Toggle Cash, 2=Toggle Violence, 3=Toggle Fire")
    print(f"{'='*60}\n")

    # Main loop
    frame_count = 0
    paused = False
    running = True
    last_frame = None
    reconnect_attempts = 0
    max_reconnect_attempts = 10

    while running:
        if not paused:
            # Flush RTSP buffer to get the latest frame (prevent lag/teleporting)
            # grab() discards 1 frame, keeping buffer at 1 (out of 2 max)
            if is_live_stream:
                cap.grab()  # Discard 1 buffered frame
            ret, frame = cap.read()
            if not ret:
                if is_live_stream:
                    reconnect_attempts += 1
                    if reconnect_attempts > max_reconnect_attempts:
                        print(f"Max reconnect attempts ({max_reconnect_attempts}) reached. Exiting.")
                        break
                    print(f"Stream lost, reconnecting... (attempt {reconnect_attempts}/{max_reconnect_attempts})")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    continue
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            # Reset reconnect counter on successful read
            reconnect_attempts = 0

            frame_count += 1

            # Run detection (every 4th frame)
            detections = []
            if frame_count % detect_every == 0:
                detections = detector.detect(frame)

                # Handle detections
                for det in detections:
                    print(f"[{det.timestamp.strftime('%H:%M:%S')}] {det.label} ({det.event_type}) conf={det.confidence:.2f}")

                    # Trigger clip save (live streams only)
                    if clip_recorder:
                        clip_recorder.trigger_event(det)
                    elif full_video_validation_enabled:
                        event_type = det.event_type or det.label.lower()
                        if event_type not in validated_event_types:
                            validated_event_types.add(event_type)
                            try:
                                is_valid, confidence, reason, corrected = gemini_validator.validate_event_video(
                                    source, event_type
                                )
                                status = "ACCEPT" if is_valid else "REJECT"
                                print(f"[Gemini] {status} {event_type} (conf={confidence:.2f}) {reason}")
                            except Exception as e:
                                print(f"[Gemini] Video validation error: {e}")

            # Create full overlay for display
            overlay_display = detector.draw_overlay(frame.copy(), detections)

            # Add to rolling buffer FIRST (before adding display labels)
            # Save overlay frames (zones + hands only) for clips/Gemini
            if clip_recorder:
                overlay_clip = detector.draw_overlay(
                    frame.copy(),
                    detections,
                    overlay_mode="minimal"
                )
                clip_recorder.add_frame(overlay_clip, frame_count)

            # Now add display labels (these won't be in the saved clips)
            cv2.putText(frame, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay_display, "Detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Status bar
            status = detector.get_status()
            status_text = f"Cash: {'ON' if status['cash']['enabled'] else 'OFF'} | "
            status_text += f"Violence: {'ON' if status['violence']['enabled'] else 'OFF'} | "
            status_text += f"Fire: {'ON' if status['fire']['enabled'] else 'OFF'}"
            cv2.putText(overlay_display, status_text, (10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Combine side by side for display
            combined = np.hstack([frame, overlay_display])

            # Resize if too large
            max_width = 1920
            if combined.shape[1] > max_width:
                scale = max_width / combined.shape[1]
                combined = cv2.resize(combined, None, fx=scale, fy=scale)

            last_frame = frame

        else:
            # Paused - show last frame
            if last_frame is not None:
                combined = np.hstack([last_frame, detector.draw_overlay(last_frame.copy(), [])])
                if combined.shape[1] > 1920:
                    scale = 1920 / combined.shape[1]
                    combined = cv2.resize(combined, None, fx=scale, fy=scale)

        # Display
        cv2.imshow("Model Examine - CCTV Stream", combined)

        # Handle keys
        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord('q'):
            running = False
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('z'):
            paused = True
            if last_frame is not None:
                configurator = ZoneConfigurator(last_frame)
                cashier_zone, cash_drawer_zone = configurator.run()
                detector.set_zones(cashier_zone, cash_drawer_zone)

                save_path = zone_config or "zones.json"
                with open(save_path, 'w') as f:
                    json.dump({
                        'cashier_zone': cashier_zone,
                        'cash_drawer_zone': cash_drawer_zone
                    }, f, indent=2)
                print(f"Saved zones to: {save_path}")
            paused = False
        elif key == ord('1'):
            result = detector.toggle_detection('cash')
            print(f"Cash detection: {'ON' if result else 'OFF'}")
        elif key == ord('2'):
            result = detector.toggle_detection('violence')
            print(f"Violence detection: {'ON' if result else 'OFF'}")
        elif key == ord('3'):
            result = detector.toggle_detection('fire')
            print(f"Fire detection: {'ON' if result else 'OFF'}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    if clip_recorder:
        print("\nSaving pending clips...")
        clip_recorder.stop()

    print("\nStream ended.")


def main():
    parser = argparse.ArgumentParser(description='CCTV Model Examine Stream')
    parser.add_argument('--rtsp', '-r', type=str, help='RTSP stream URL')
    parser.add_argument('--video', '-v', type=str, help='Video file path')
    parser.add_argument('--zone-config', '-z', type=str, default='zones.json',
                        help='Zone configuration JSON file')
    parser.add_argument('--zone-profile', type=int,
                        help='Load zones from zones_<N>.json (e.g., 1 -> zones_1.json)')
    parser.add_argument('--setup-zones', '-s', action='store_true',
                        help='Run zone setup before streaming')
    parser.add_argument('--output', '-o', type=str,
                        help='Output directory for clip recordings')
    parser.add_argument('--hand-distance', type=int, default=125,
                        help='Hand touch distance threshold (pixels)')
    parser.add_argument('--global-cash-cooldown', type=float,
                        help='Global cash cooldown seconds (all tracks share this)')
    parser.add_argument('--cash-cooldown', type=float,
                        help='(Deprecated) Use --global-cash-cooldown instead')
    parser.add_argument('--cash-predict-cooldown', type=float,
                        help='(Deprecated) Use --global-cash-cooldown instead')
    parser.add_argument('--violence-cooldown', type=float,
                        help='Violence cooldown seconds')
    parser.add_argument('--fire-cooldown', type=float,
                        help='Fire cooldown seconds')
    parser.add_argument('--no-cash', action='store_true', help='Disable cash detection')
    parser.add_argument('--no-violence', action='store_true', help='Disable violence detection')
    parser.add_argument('--no-fire', action='store_true', help='Disable fire detection')
    parser.add_argument('--gemini', action='store_true', help='Enable Gemini validation for clips')
    parser.add_argument('--gemini-key', type=str, help='Gemini API key (or use GEMINI_API_KEY env var)')
    parser.add_argument('--gemini-source', type=str, default='original',
                        choices=['original', 'overlay'],
                        help='Gemini input source for clip validation')

    args = parser.parse_args()

    # Determine source
    source = args.rtsp or args.video
    if not source:
        print("Select input source:")
        print("  1) RTSP stream")
        print("  2) Video file")
        choice = input("Enter 1 or 2: ").strip()
        if choice == '1':
            source = input("RTSP URL: ").strip()
        elif choice == '2':
            source = input("Video file path: ").strip()
        else:
            print("Error: Specify --rtsp or --video")
            return
        if not source:
            print("Error: Source is empty")
            return

    run_stream(
        source=source,
        zone_config=args.zone_config,
        setup_zones=args.setup_zones,
        output_dir=args.output,
        detect_cash=not args.no_cash,
        detect_violence=not args.no_violence,
        detect_fire=not args.no_fire,
        hand_distance=args.hand_distance,
        global_cash_cooldown=args.global_cash_cooldown,
        cash_cooldown=args.cash_cooldown,
        cash_predict_cooldown=args.cash_predict_cooldown,
        violence_cooldown=args.violence_cooldown,
        fire_cooldown=args.fire_cooldown,
        zone_profile=args.zone_profile,
        gemini_api_key=args.gemini_key or os.environ.get('GEMINI_API_KEY', ''),
        gemini_enabled=args.gemini or bool(args.gemini_key or os.environ.get('GEMINI_API_KEY', '')),
        gemini_source=args.gemini_source
    )


if __name__ == "__main__":
    main()
