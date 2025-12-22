"""
Fire and Smoke Detector

Detects fire and smoke using:
1. YOLO object detection (trained on fire/smoke if available)
2. Color-based detection as fallback (fire colors: red, orange, yellow)
3. Motion analysis (flickering patterns)
4. Smoke detection (gray/white regions with movement)
5. Skin color exclusion to reduce false positives
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from .base_detector import BaseDetector, Detection


class FireDetector(BaseDetector):
    """
    Detects fire and smoke using YOLO + computer vision techniques.
    
    Methods:
    1. YOLO-based fire/smoke detection (primary - most accurate)
    2. Color-based fire detection (fallback)
    3. Flickering detection (temporal variation)
    4. Skin color exclusion to avoid false positives
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        self.yolo_model = None
        self.use_yolo = True  # Try YOLO first
        
        # Detection parameters - STRICT thresholds
        self.fire_confidence = config.get('fire_confidence', 0.70)
        self.min_fire_frames = config.get('min_fire_frames', 10)
        self.min_fire_area = config.get('min_fire_area', 3000)  # Larger area required
        
        # Color ranges for fire detection (HSV) - STRICTER ranges
        # Fire is very bright orange/yellow, not just red
        self.fire_lower1 = np.array([5, 150, 200])     # Bright orange-yellow (stricter)
        self.fire_upper1 = np.array([25, 255, 255])
        
        self.fire_lower2 = np.array([0, 200, 220])     # Very bright red (stricter)
        self.fire_upper2 = np.array([5, 255, 255])
        
        # Skin color ranges to EXCLUDE (HSV) - prevents detecting people as fire
        self.skin_lower1 = np.array([0, 20, 70])
        self.skin_upper1 = np.array([20, 150, 255])
        self.skin_lower2 = np.array([0, 30, 100])
        self.skin_upper2 = np.array([25, 170, 200])
        
        # Smoke detection (gray/white with some transparency)
        self.smoke_lower = np.array([0, 0, 150])       # Brighter threshold
        self.smoke_upper = np.array([180, 30, 255])
        
        # Tracking state
        self.consecutive_fire = 0
        self.consecutive_smoke = 0
        self.last_fire_frame = -100
        self.fire_cooldown = 60  # frames between alerts
        
        # Frame history for flickering detection
        self.frame_history = deque(maxlen=10)
        self.fire_mask_history = deque(maxlen=10)
        self._last_flicker = 0.0  # For debug display
        
        # Background subtractor for motion/smoke detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize the detector - use trained fire/smoke YOLO model"""
        try:
            from ultralytics import YOLO
            from pathlib import Path
            from . import get_device
            
            # Get device based on USE_GPU setting
            use_gpu_setting = self.config.get('use_gpu', 'auto')
            device = get_device(use_gpu_setting)
            
            print(f"[GPU] Fire detector using: {device.upper()}")
            if device == 'cuda':
                import torch
                print(f"      GPU: {torch.cuda.get_device_name(0)}")
            
            models_dir = Path(self.config.get('models_dir', 'models'))
            
            # Get fire model name from config
            fire_model_name = self.config.get('fire_model', 'fire_smoke_yolov8.pt')
            
            # Try to load the trained fire/smoke detection model first
            fire_model_path = models_dir / fire_model_name
            if fire_model_path.exists():
                self.yolo_model = YOLO(str(fire_model_path))
                self.yolo_model.to(device)  # Move to GPU
                self.use_yolo = True
                print(f"[OK] Fire detector loaded: {fire_model_path} on {device}")
                
                # Get class names from the model
                self.fire_classes = self.yolo_model.names
                print(f"[INFO] Fire model classes: {self.fire_classes}")
            else:
                # Fallback to color-based detection
                print("[INFO] Fire YOLO model not found at:", fire_model_path)
                print("[INFO] Using color-based fire detection as fallback")
                self.use_yolo = False
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"[WARNING] YOLO fire model error: {e}")
            print("[INFO] Using color-based fire detection as fallback")
            self.use_yolo = False
            self.is_initialized = True
            return True
    
    def detect_fire_color(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect fire-colored regions in the frame
        Excludes skin-colored regions to avoid false positives
        
        Returns:
            fire_mask: Binary mask of fire-colored regions
            fire_regions: List of detected fire region info
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create fire color mask (combining orange and bright red ranges)
        mask1 = cv2.inRange(hsv, self.fire_lower1, self.fire_upper1)
        mask2 = cv2.inRange(hsv, self.fire_lower2, self.fire_upper2)
        fire_mask = cv2.bitwise_or(mask1, mask2)
        
        # Create skin color mask to EXCLUDE
        skin_mask1 = cv2.inRange(hsv, self.skin_lower1, self.skin_upper1)
        skin_mask2 = cv2.inRange(hsv, self.skin_lower2, self.skin_upper2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        # Dilate skin mask to be more aggressive in excluding skin
        kernel_skin = np.ones((15, 15), np.uint8)
        skin_mask = cv2.dilate(skin_mask, kernel_skin, iterations=2)
        
        # Remove skin regions from fire mask
        fire_mask = cv2.bitwise_and(fire_mask, cv2.bitwise_not(skin_mask))
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((7, 7), np.uint8)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_fire_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate region properties
                aspect_ratio = w / h if h > 0 else 0
                
                # Fire typically has irregular shape, not too flat or too tall
                # Exclude very horizontal regions (could be shelves, signs)
                if 0.3 < aspect_ratio < 4:
                    # Calculate mean color values in the region
                    roi = frame[y:y+h, x:x+w]
                    roi_hsv = hsv[y:y+h, x:x+w]
                    mean_brightness = np.mean(roi)
                    mean_saturation = np.mean(roi_hsv[:, :, 1])
                    
                    # Fire is typically VERY bright and saturated
                    # This helps exclude dim red objects
                    if mean_brightness > 150 and mean_saturation > 100:
                        fire_regions.append({
                            'bbox': (x, y, x + w, y + h),
                            'area': area,
                            'center': (x + w // 2, y + h // 2),
                            'brightness': mean_brightness,
                            'saturation': mean_saturation
                        })
        
        return fire_mask, fire_regions
    
    def detect_flickering(self, current_mask: np.ndarray) -> float:
        """
        Detect flickering patterns typical of fire
        
        Fire flickers - the bright regions change rapidly
        This is a KEY differentiator from static red/orange objects
        """
        if len(self.fire_mask_history) < 5:  # Need more history for reliable flickering
            return 0.0
        
        # Compare current mask with previous masks
        total_diff = 0
        count = 0
        current_h, current_w = current_mask.shape[:2]
        
        for prev_mask in list(self.fire_mask_history)[-5:]:
            # Ensure masks are the same size before comparing
            prev_h, prev_w = prev_mask.shape[:2]
            if prev_h != current_h or prev_w != current_w:
                # Resize previous mask to match current
                prev_mask = cv2.resize(prev_mask, (current_w, current_h))
            
            diff = cv2.absdiff(current_mask, prev_mask)
            total_diff += np.sum(diff) / (current_h * current_w)
            count += 1
        
        # Normalize flickering score
        flicker_score = total_diff / count / 255.0 if count > 0 else 0.0
        
        # Fire needs SIGNIFICANT flickering - be stricter
        # Low flickering = probably not fire
        return min(1.0, flicker_score * 3)  # Reduced multiplier for stricter detection
    
    def detect_smoke(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect smoke using background subtraction and color analysis
        """
        smoke_regions = []
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Look for gray/white moving regions
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        smoke_color_mask = cv2.inRange(hsv, self.smoke_lower, self.smoke_upper)
        
        # Combine motion and color
        smoke_mask = cv2.bitwise_and(fg_mask, smoke_color_mask)
        
        # Clean up
        kernel = np.ones((7, 7), np.uint8)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_fire_area * 2:  # Smoke regions tend to be larger
                x, y, w, h = cv2.boundingRect(contour)
                
                # Smoke tends to rise (appear in upper parts and move up)
                # For CCTV, we just detect large gray moving regions
                smoke_regions.append({
                    'bbox': (x, y, x + w, y + h),
                    'area': area,
                    'center': (x + w // 2, y + h // 2)
                })
        
        return smoke_regions
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect fire and smoke in the frame using YOLO or color-based fallback"""
        detections = []
        
        if not self.is_initialized:
            return detections
        
        try:
            # Use YOLO fire/smoke model if available (PRIMARY METHOD - most accurate)
            if self.use_yolo and self.yolo_model is not None:
                return self.detect_with_yolo(frame)
            
            # Fallback to color-based detection
            return self.detect_with_color(frame)
            
        except Exception as e:
            print(f"[WARNING] Fire detection error: {e}")
            return detections
    
    def detect_with_yolo(self, frame: np.ndarray) -> List[Detection]:
        """Detect fire and smoke using trained YOLO model"""
        detections = []
        
        try:
            # Run YOLO inference
            results = self.yolo_model(frame, verbose=False, conf=0.25)
            
            if not results or len(results) == 0:
                self.consecutive_fire = max(0, self.consecutive_fire - 1)
                self._last_flicker = 0.0
                return detections
            
            result = results[0]
            fire_detected = False
            best_detection = None
            best_confidence = 0.0
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    class_name = self.fire_classes.get(int(cls), "unknown").lower()
                    
                    # Check if it's fire or smoke
                    if "fire" in class_name or "smoke" in class_name:
                        if conf > best_confidence:
                            best_confidence = float(conf)
                            best_detection = {
                                'bbox': tuple(map(int, box)),
                                'confidence': float(conf),
                                'class': class_name
                            }
                            fire_detected = True
            
            # Store for debug display
            self._last_flicker = best_confidence if fire_detected else 0.0
            
            # Track fire detection
            if fire_detected and best_confidence >= self.fire_confidence:
                self.consecutive_fire += 1
            else:
                self.consecutive_fire = max(0, self.consecutive_fire - 1)
            
            # Generate fire alert after consecutive detections
            # IMPORTANT: Also check that current frame meets confidence threshold
            if (self.consecutive_fire >= self.min_fire_frames and
                self.frame_count - self.last_fire_frame > self.fire_cooldown and
                best_detection is not None and
                best_detection['confidence'] >= self.fire_confidence):
                
                detection = Detection(
                    label="FIRE",
                    confidence=best_detection['confidence'],
                    bbox=best_detection['bbox'],
                    metadata={
                        'type': best_detection['class'],
                        'method': 'yolo',
                        'model': 'fire_smoke_yolov8'
                    }
                )
                detections.append(detection)
                
                self.last_fire_frame = self.frame_count
                self.consecutive_fire = 0
                
        except Exception as e:
            print(f"[WARNING] YOLO fire detection error: {e}")
        
        return detections
    
    def detect_with_color(self, frame: np.ndarray) -> List[Detection]:
        """Fallback color-based fire detection"""
        detections = []
        
        try:
            # Store frame for temporal analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_history.append(gray)
            
            # Detect fire-colored regions
            fire_mask, fire_regions = self.detect_fire_color(frame)
            self.fire_mask_history.append(fire_mask)
            
            # Calculate flickering score
            flicker_score = self.detect_flickering(fire_mask)
            self._last_flicker = flicker_score  # Store for debug display
            
            # Detect smoke
            smoke_regions = self.detect_smoke(frame)
            
            # Analyze fire detections
            fire_detected = False
            best_fire_region = None
            fire_confidence = 0.0
            
            if fire_regions:
                # Sort by area (larger = more likely to be real fire)
                fire_regions.sort(key=lambda x: x['area'], reverse=True)
                best_fire_region = fire_regions[0]
                
                # Calculate confidence
                area_score = min(1.0, best_fire_region['area'] / 15000)
                brightness_score = min(1.0, best_fire_region['brightness'] / 255)
                saturation_score = min(1.0, best_fire_region.get('saturation', 100) / 255)
                
                # FLICKERING IS REQUIRED for color-based detection
                if flicker_score < 0.15:
                    fire_confidence = 0.0
                else:
                    fire_confidence = (
                        area_score * 0.15 +
                        flicker_score * 0.60 +
                        brightness_score * 0.15 +
                        saturation_score * 0.10
                    )
                
                if fire_confidence > self.fire_confidence:
                    fire_detected = True
            
            # Track fire detection
            if fire_detected:
                self.consecutive_fire += 1
            else:
                self.consecutive_fire = max(0, self.consecutive_fire - 1)
            
            # Generate fire alert
            # IMPORTANT: Also check that current frame meets confidence threshold
            if (self.consecutive_fire >= self.min_fire_frames and
                self.frame_count - self.last_fire_frame > self.fire_cooldown and
                best_fire_region is not None and
                fire_confidence > self.fire_confidence):
                
                detection = Detection(
                    label="FIRE",
                    confidence=fire_confidence,
                    bbox=best_fire_region['bbox'],
                    metadata={
                        'type': 'fire',
                        'area': best_fire_region['area'],
                        'flicker_score': flicker_score,
                        'regions_count': len(fire_regions)
                    }
                )
                detections.append(detection)
                
                self.last_fire_frame = self.frame_count
                self.consecutive_fire = 0
            
            # Also check for smoke (separate alert) - STRICTER
            if smoke_regions and len(smoke_regions) > 0:
                largest_smoke = max(smoke_regions, key=lambda x: x['area'])
                
                # Smoke confidence based on area - need LARGE area
                smoke_confidence = min(1.0, largest_smoke['area'] / 20000) * 0.7
                
                # Only alert for very confident smoke detection
                if smoke_confidence > 0.6 and self.frame_count - self.last_fire_frame > 60:
                    detection = Detection(
                        label="FIRE",  # Grouped with fire
                        confidence=smoke_confidence,
                        bbox=largest_smoke['bbox'],
                        metadata={
                            'type': 'smoke',
                            'area': largest_smoke['area'],
                            'regions_count': len(smoke_regions)
                        }
                    )
                    detections.append(detection)
        
        except Exception as e:
            print(f"⚠️ Fire detection error: {e}")
        
        return detections
    
    def draw_fire_overlay(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw fire/smoke detection overlays"""
        for det in detections:
            if det.label == "FIRE":
                x1, y1, x2, y2 = det.bbox
                
                # Fire = orange, Smoke = gray
                if det.metadata.get('type') == 'smoke':
                    color = (128, 128, 128)
                    label = "SMOKE"
                else:
                    color = (0, 100, 255)  # Orange
                    label = "FIRE"
                
                # Draw animated warning box
                thickness = 3 if self.frame_count % 10 < 5 else 2  # Flashing effect
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label with warning background
                label_text = f"🔥 {label}: {det.confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w + 10, y1), color, -1)
                cv2.putText(frame, label_text, (x1 + 5, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
