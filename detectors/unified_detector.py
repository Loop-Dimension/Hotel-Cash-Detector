"""
Unified Detector

Combines all detection modules into a single interface.
Manages model loading, frame processing, and alert coordination.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from .base_detector import Detection
from .cash_detector import CashTransactionDetector
from .violence_detector import ViolenceDetector
from .fire_detector import FireDetector


class UnifiedDetector:
    """
    Unified detector that combines:
    - Cash Transaction Detection
    - Violence Detection
    - Fire/Smoke Detection
    
    Manages all detectors and provides a single interface for processing.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Get models directory
        models_dir = self.config.get('models_dir', 'models')
        
        # Initialize individual detectors
        self.cash_detector = CashTransactionDetector({
            'models_dir': models_dir,
            'cashier_zone': self.config.get('cashier_zone', [100, 100, 400, 300]),
            'hand_touch_distance': self.config.get('hand_touch_distance', 100),
            'pose_confidence': self.config.get('pose_confidence', 0.5),
            'min_transaction_frames': self.config.get('min_transaction_frames', 3)
        })
        
        self.violence_detector = ViolenceDetector({
            'models_dir': models_dir,
            'violence_confidence': self.config.get('violence_confidence', 0.6),
            'min_violence_frames': self.config.get('min_violence_frames', 5),
            'motion_threshold': self.config.get('motion_threshold', 50)
        })
        
        self.fire_detector = FireDetector({
            'models_dir': models_dir,
            'fire_confidence': self.config.get('fire_confidence', 0.7),
            'min_fire_frames': self.config.get('min_fire_frames', 10),
            'min_fire_area': self.config.get('min_fire_area', 2000)
        })
        
        # Detection state
        self.is_initialized = False
        self.frame_count = 0
        self.all_detections: List[Detection] = []
        
        # Alert tracking
        self.alerts_history: List[Dict] = []
        self.alert_callbacks = []
        
        # Detection toggles
        self.detect_cash = self.config.get('detect_cash', True)
        self.detect_violence = self.config.get('detect_violence', True)
        self.detect_fire = self.config.get('detect_fire', True)
        
        # Debug mode - shows detection details on frame
        self.debug_mode = self.config.get('debug_mode', False)
        
    def initialize(self) -> bool:
        """Initialize all detectors"""
        print("\n" + "=" * 50)
        print("Initializing Unified Detector")
        print("=" * 50)
        
        all_success = True
        
        if self.detect_cash:
            print("\nLoading Cash Transaction Detector...")
            if not self.cash_detector.initialize():
                print("[WARNING] Cash detector failed to initialize")
                all_success = False
            else:
                print("[OK] Cash detector ready")
        
        if self.detect_violence:
            print("\nLoading Violence Detector...")
            if not self.violence_detector.initialize():
                print("[WARNING] Violence detector failed to initialize")
                all_success = False
            else:
                print("[OK] Violence detector ready")
        
        if self.detect_fire:
            print("\nLoading Fire Detector...")
            if not self.fire_detector.initialize():
                print("[WARNING] Fire detector failed to initialize")
                all_success = False
            else:
                print("[OK] Fire detector ready")
        
        # Always set to initialized so frame processing can continue
        self.is_initialized = True
        
        if all_success:
            print("\n[OK] All detectors initialized successfully!")
        else:
            print("\n[WARNING] Some detectors failed - continuing with available detectors")
        
        print("=" * 50 + "\n")
        return True  # Always return True to allow processing
    
    def set_cashier_zone(self, zone: List[int]):
        """Update the cashier zone for cash detection and violence exclusion"""
        self.cash_detector.set_cashier_zone(zone)
        self.violence_detector.set_cashier_zone(zone)  # Exclude cashier zone from violence
    
    def toggle_debug(self, enabled: bool = None):
        """Toggle debug mode - shows extra detection info on frame"""
        if enabled is None:
            self.debug_mode = not self.debug_mode
        else:
            self.debug_mode = enabled
        return self.debug_mode
    
    def add_alert_callback(self, callback):
        """Add a callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alert(self, detection: Detection):
        """Trigger alert callbacks"""
        alert_info = {
            'detection': detection.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'frame_number': self.frame_count
        }
        
        self.alerts_history.append(alert_info)
        
        # Keep only last 100 alerts
        if len(self.alerts_history) > 100:
            self.alerts_history = self.alerts_history[-50:]
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_info)
            except Exception as e:
                print(f"⚠️ Alert callback error: {e}")
    
    def process_frame(self, frame: np.ndarray, draw_overlay: bool = True) -> Dict:
        """
        Process a single frame through all detectors
        
        Args:
            frame: BGR image as numpy array
            draw_overlay: Whether to draw detection overlays on frame
            
        Returns:
            Dictionary with:
            - frame: Processed frame with overlays
            - detections: List of all detections
            - alerts: List of new alerts triggered
        """
        # Always increment frame count first
        self.frame_count += 1
        
        if not self.is_initialized:
            if not self.initialize():
                # Still draw overlays even if initialization fails
                if draw_overlay:
                    frame = self.draw_status_bar(frame)
                return {
                    'frame': frame,
                    'detections': [],
                    'alerts': [],
                    'frame_number': self.frame_count
                }
        
        all_detections = []
        new_alerts = []
        
        # Run all enabled detectors
        if self.detect_cash:
            cash_detections = self.cash_detector.process_frame(frame)
            all_detections.extend(cash_detections)
        
        if self.detect_violence:
            violence_detections = self.violence_detector.process_frame(frame)
            all_detections.extend(violence_detections)
        
        if self.detect_fire:
            fire_detections = self.fire_detector.process_frame(frame)
            all_detections.extend(fire_detections)
        
        # Trigger alerts for new detections
        for det in all_detections:
            self._trigger_alert(det)
            new_alerts.append(det.to_dict())
        
        # Draw overlays if requested
        if draw_overlay:
            frame = self.draw_overlays(frame, all_detections)
        
        self.all_detections.extend(all_detections)
        
        return {
            'frame': frame,
            'detections': [d.to_dict() for d in all_detections],
            'alerts': new_alerts,
            'frame_number': self.frame_count
        }
    
    def draw_overlays(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw all detection overlays on frame"""
        # Draw cashier zone
        if self.detect_cash:
            frame = self.cash_detector.draw_cashier_zone(frame)
        
        # Draw detections
        for det in detections:
            if det.bbox:
                x1, y1, x2, y2 = det.bbox
                
                # Color coding
                colors = {
                    "CASH": (0, 255, 0),      # Green
                    "VIOLENCE": (0, 0, 255),   # Red
                    "FIRE": (0, 165, 255)      # Orange
                }
                color = colors.get(det.label, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw label (no emoji - causes display issues)
                label = f"{det.label}: {det.confidence:.2f}"
                
                # Label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw status bar
        frame = self.draw_status_bar(frame)
        
        # Draw debug info if enabled
        if self.debug_mode:
            frame = self.draw_debug_overlay(frame, detections)
        
        return frame
    
    def draw_debug_overlay(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw debug information on frame"""
        h, w = frame.shape[:2]
        
        # Debug panel background (top-left)
        cv2.rectangle(frame, (5, 5), (380, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (380, 180), (0, 255, 255), 2)
        
        y_offset = 25
        cv2.putText(frame, "DEBUG MODE", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Violence detector info - show actual threshold
        y_offset += 25
        violence_consec = getattr(self.violence_detector, 'consecutive_violence', 0)
        violence_thresh = getattr(self.violence_detector, 'min_violence_frames', 10)
        cv2.putText(frame, f"Violence frames: {violence_consec}/{violence_thresh}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Fire detector info - show actual threshold
        y_offset += 20
        fire_consec = getattr(self.fire_detector, 'consecutive_fire', 0)
        fire_thresh = getattr(self.fire_detector, 'min_fire_frames', 10)
        flicker = getattr(self.fire_detector, '_last_flicker', 0)
        cv2.putText(frame, f"Fire frames: {fire_consec}/{fire_thresh} | Conf: {flicker:.2f}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Cash detector info - show actual threshold
        y_offset += 20
        cash_consec = getattr(self.cash_detector, 'consecutive_detections', 0)
        cash_thresh = getattr(self.cash_detector, 'min_transaction_frames', 3)
        cv2.putText(frame, f"Cash frames: {cash_consec}/{cash_thresh}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Cashier zone
        y_offset += 20
        zone = self.cash_detector.cashier_zone
        cv2.putText(frame, f"Cashier zone: {zone}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection counts
        y_offset += 25
        cv2.putText(frame, f"Current detections: {len(detections)}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show detection types
        for det in detections:
            y_offset += 18
            if y_offset > 175:
                break
            cv2.putText(frame, f"  -> {det.label}: {det.confidence:.2f}", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        return frame
    
    def draw_status_bar(self, frame: np.ndarray) -> np.ndarray:
        """Draw status bar with detection info"""
        h, w = frame.shape[:2]
        
        # Status bar background
        cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)
        
        # Detection status (using plain text - emojis cause display issues)
        statuses = []
        if self.detect_cash:
            statuses.append("[CASH: ON]")
        if self.detect_violence:
            statuses.append("[VIOLENCE: ON]")
        if self.detect_fire:
            statuses.append("[FIRE: ON]")
        
        status_text = " | ".join(statuses)
        cv2.putText(frame, status_text, (10, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Frame counter
        frame_text = f"Frame: {self.frame_count}"
        if self.debug_mode:
            frame_text = "[DEBUG] " + frame_text
        cv2.putText(frame, frame_text, (w - 180, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if self.debug_mode else (255, 255, 255), 1)
        
        return frame
    
    def get_detection_summary(self) -> Dict:
        """Get summary of all detections"""
        summary = {
            'total_frames': self.frame_count,
            'total_detections': len(self.all_detections),
            'by_type': {
                'CASH': 0,
                'VIOLENCE': 0,
                'FIRE': 0
            },
            'recent_alerts': self.alerts_history[-10:]
        }
        
        for det in self.all_detections:
            if det.label in summary['by_type']:
                summary['by_type'][det.label] += 1
        
        return summary
    
    def reset(self):
        """Reset all detectors"""
        self.frame_count = 0
        self.all_detections = []
        self.alerts_history = []
        
        self.cash_detector.reset()
        self.violence_detector.reset()
        self.fire_detector.reset()
