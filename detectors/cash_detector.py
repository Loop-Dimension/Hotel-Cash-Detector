"""
Cash Transaction Detector - Enhanced Two-Step Detection

Detection Flow:
1. STEP 1 - Hand Touch: Detect when Customer hand touches Cashier hand
   - Cashier = person inside Cashier Zone
   - Customer = person outside Cashier Zone
   - Distance between hands < hand_touch_distance threshold
   
2. STEP 2 - Cash Deposit: After touch, track Cashier's hand
   - Track for hand_tracking_duration frames
   - Detect when Cashier's hand enters Cash Drawer Zone
   - Only then trigger CASH detection

This ensures we only detect REAL cash transactions:
Customer pays → Cashier receives → Cashier deposits in drawer
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from .base_detector import BaseDetector, Detection


class CashTransactionDetector(BaseDetector):
    """
    Enhanced Cash Transaction Detector with Two-Step Verification.
    
    Detection only triggers when:
    1. Customer touches Cashier's hand (proximity check)
    2. Cashier's hand then moves to Cash Drawer zone
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.pose_model = None
        self.person_model = None
        
        # Cashier zone (where cashier stands)
        self.cashier_zone = config.get('cashier_zone', [100, 100, 400, 300])
        
        # Cash Drawer zone (where money is deposited)
        self.cash_drawer_zone = config.get('cash_drawer_zone', [50, 200, 150, 100])
        
        # Video dimensions
        self.video_width = config.get('video_width', 1920)
        self.video_height = config.get('video_height', 1080)
        
        # Detection parameters
        self.hand_touch_distance = config.get('hand_touch_distance', 100)
        self.pose_confidence = config.get('pose_confidence', 0.5)
        self.min_cash_confidence = config.get('min_cash_confidence', 0.70)
        
        # Hand tracking duration (frames to track after touch)
        self.hand_tracking_duration = config.get('hand_tracking_duration', 90)  # ~3 seconds at 30fps
        
        # Debug info storage
        self.last_detection_debug = {}
        self.show_pose_overlay = config.get('show_pose_overlay', False)
        
        # ==================== TWO-STEP TRACKING STATE ====================
        # Step 1: Hand touch detection
        self.pending_transaction = None  # Stores touch event waiting for drawer deposit
        self.touch_frame = -1  # Frame when touch occurred
        self.tracking_cashier_hands = False  # Whether we're tracking cashier's hands
        
        # Step 2: Cashier hand tracking  
        self.cashier_hand_history = deque(maxlen=30)  # Track hand positions
        self.frames_since_touch = 0
        
        # Cooldown to prevent duplicate detections
        self.last_transaction_frame = -100
        self.transaction_cooldown = 60  # frames between transactions (~2 seconds)
        
        # Frame counter
        self.frame_count = 0
        
        # Hand keypoint indices (COCO format)
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        
    def initialize(self) -> bool:
        """Load YOLO models for person and pose detection"""
        try:
            from ultralytics import YOLO
            from pathlib import Path
            
            # Get model paths from config or use defaults
            models_dir = Path(self.config.get('models_dir', 'models'))
            
            # Get model names from config (default to yolov8s for better accuracy)
            pose_model_name = self.config.get('pose_model', 'yolov8s-pose.pt')
            yolo_model_name = self.config.get('yolo_model', 'yolov8s.pt')
            
            # Check GPU availability
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"🎮 Using device: {device}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
            
            # Load pose model for hand detection
            pose_model_path = models_dir / pose_model_name
            if pose_model_path.exists():
                self.pose_model = YOLO(str(pose_model_path))
                self.pose_model.to(device)  # Move to GPU
                print(f"✅ Loaded pose model: {pose_model_path} on {device}")
            else:
                # Download if not exists
                self.pose_model = YOLO(pose_model_name)
                self.pose_model.to(device)  # Move to GPU
                print(f"✅ Downloaded and loaded pose model: {pose_model_name} on {device}")
            
            # Load person detection model as backup
            person_model_path = models_dir / yolo_model_name
            if person_model_path.exists():
                self.person_model = YOLO(str(person_model_path))
                self.person_model.to(device)  # Move to GPU
            else:
                self.person_model = YOLO(yolo_model_name)
                self.person_model.to(device)  # Move to GPU
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize CashTransactionDetector: {e}")
            return False
    
    def update_video_dimensions(self, width: int, height: int):
        """Update video dimensions"""
        self.video_width = width
        self.video_height = height
    
    def set_cashier_zone(self, zone: List[int]):
        """Update the cashier zone [x, y, w, h]"""
        self.cashier_zone = zone
    
    def set_cash_drawer_zone(self, zone: List[int]):
        """Update the cash drawer zone [x, y, w, h]"""
        self.cash_drawer_zone = zone
    
    def set_hand_touch_distance(self, distance: int):
        """Update hand touch distance threshold"""
        self.hand_touch_distance = max(10, min(500, distance))
    
    def set_hand_tracking_duration(self, frames: int):
        """Update hand tracking duration (frames after touch)"""
        self.hand_tracking_duration = max(15, min(300, frames))
    
    def is_in_cashier_zone(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the cashier zone"""
        x, y = point
        zx, zy, zw, zh = self.cashier_zone
        return zx <= x <= zx + zw and zy <= y <= zy + zh
    
    def is_in_cash_drawer_zone(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the cash drawer zone"""
        x, y = point
        zx, zy, zw, zh = self.cash_drawer_zone
        return zx <= x <= zx + zw and zy <= y <= zy + zh
    
    def get_person_center(self, keypoints: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get the center point of a person using hip keypoints (most stable).
        Falls back to bbox center if keypoints not available.
        
        COCO keypoint indices:
        - 11: left_hip, 12: right_hip
        - 5: left_shoulder, 6: right_shoulder
        """
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        
        # Try to use hip center (most stable for determining position)
        if keypoints is not None and len(keypoints) > RIGHT_HIP:
            left_hip = keypoints[LEFT_HIP]
            right_hip = keypoints[RIGHT_HIP]
            
            # Check if hips are detected with good confidence
            if (len(left_hip) >= 3 and left_hip[2] > 0.3 and 
                len(right_hip) >= 3 and right_hip[2] > 0.3):
                center_x = int((left_hip[0] + right_hip[0]) / 2)
                center_y = int((left_hip[1] + right_hip[1]) / 2)
                return (center_x, center_y)
            
            # Fallback to shoulder center
            if len(keypoints) > RIGHT_SHOULDER:
                left_shoulder = keypoints[LEFT_SHOULDER]
                right_shoulder = keypoints[RIGHT_SHOULDER]
                
                if (len(left_shoulder) >= 3 and left_shoulder[2] > 0.3 and 
                    len(right_shoulder) >= 3 and right_shoulder[2] > 0.3):
                    center_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
                    center_y = int((left_shoulder[1] + right_shoulder[1]) / 2)
                    return (center_x, center_y)
        
        # Fallback to bbox center
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def is_person_in_cashier_zone(self, keypoints: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if a person is in the cashier zone using MAJORITY BODY AREA.
        
        Uses the bounding box area overlap - if more than 50% of the body
        is inside the cashier zone, they are classified as cashier.
        This prevents one person being detected as both cashier and client
        when they're straddling the zone boundary.
        """
        x1, y1, x2, y2 = bbox
        zx, zy, zw, zh = self.cashier_zone
        
        # Calculate intersection area
        ix1 = max(x1, zx)
        iy1 = max(y1, zy)
        ix2 = min(x2, zx + zw)
        iy2 = min(y2, zy + zh)
        
        if ix2 <= ix1 or iy2 <= iy1:
            # No overlap - person is outside zone
            return False
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        box_area = (x2 - x1) * (y2 - y1)
        
        if box_area <= 0:
            return False
        
        overlap_ratio = intersection / box_area
        
        # If more than 50% of body is in zone, classify as inside
        return overlap_ratio > 0.5
    
    def is_box_in_cashier_zone(self, bbox: Tuple[int, int, int, int], threshold: float = 0.3) -> bool:
        """Check if a bounding box overlaps with cashier zone (legacy, kept for compatibility)"""
        x1, y1, x2, y2 = bbox
        zx, zy, zw, zh = self.cashier_zone
        
        # Calculate intersection
        ix1 = max(x1, zx)
        iy1 = max(y1, zy)
        ix2 = min(x2, zx + zw)
        iy2 = min(y2, zy + zh)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        box_area = (x2 - x1) * (y2 - y1)
        
        return (intersection / box_area) >= threshold if box_area > 0 else False
    
    def get_hand_positions(self, keypoints: np.ndarray, confidence_threshold: float = 0.3) -> Dict:
        """Extract hand (wrist) positions from pose keypoints"""
        hands = {}
        
        if keypoints is None or len(keypoints) < 11:
            return hands
        
        # Check left wrist
        if len(keypoints) > self.LEFT_WRIST:
            lw = keypoints[self.LEFT_WRIST]
            if len(lw) >= 3 and lw[2] >= confidence_threshold:
                hands['left'] = (int(lw[0]), int(lw[1]), float(lw[2]))
        
        # Check right wrist  
        if len(keypoints) > self.RIGHT_WRIST:
            rw = keypoints[self.RIGHT_WRIST]
            if len(rw) >= 3 and rw[2] >= confidence_threshold:
                hands['right'] = (int(rw[0]), int(rw[1]), float(rw[2]))
        
        return hands
    
    def calculate_hand_distance(self, hand1: Tuple, hand2: Tuple) -> float:
        """Calculate Euclidean distance between two hand positions"""
        return np.sqrt((hand1[0] - hand2[0])**2 + (hand1[1] - hand2[1])**2)
    
    def detect_hand_proximity(self, people_hands: List[Dict]) -> List[Dict]:
        """
        Detect when hands from CASHIER and CLIENT are close together.
        
        STRICT RULES:
        - Only triggers between cashier (in zone) and client (outside zone)
        - Two clients touching = NO detection
        - Two cashiers touching = NO detection
        - Distance must be within hand_touch_distance threshold
        """
        proximity_events = []
        
        for i, person1 in enumerate(people_hands):
            for j, person2 in enumerate(people_hands):
                if i >= j:
                    continue
                
                # STRICT: One must be IN zone (cashier), one must be OUTSIDE (customer)
                p1_in = person1.get('in_cashier_zone', False)
                p2_in = person2.get('in_cashier_zone', False)
                
                # XOR check: exactly one in zone, one outside
                is_cashier_customer_pair = (p1_in and not p2_in) or (not p1_in and p2_in)
                
                if not is_cashier_customer_pair:
                    # Skip: both are customers OR both are cashiers
                    continue
                
                # Check all hand combinations between cashier and customer
                for hand1_name, hand1_pos in person1.get('hands', {}).items():
                    for hand2_name, hand2_pos in person2.get('hands', {}).items():
                        distance = self.calculate_hand_distance(hand1_pos[:2], hand2_pos[:2])
                        hand_confidence = min(hand1_pos[2], hand2_pos[2])
                        
                        # STRICT: Only accept if distance is within threshold
                        if distance < self.hand_touch_distance:
                            # Calculate midpoint of the hand interaction
                            midpoint = (
                                (hand1_pos[0] + hand2_pos[0]) // 2,
                                (hand1_pos[1] + hand2_pos[1]) // 2
                            )
                            
                            # Calculate score based on distance (closer = higher score)
                            distance_score = max(0, 1 - (distance / self.hand_touch_distance))
                            
                            proximity_events.append({
                                'person1_idx': i,
                                'person2_idx': j,
                                'person1_role': 'cashier' if p1_in else 'client',
                                'person2_role': 'cashier' if p2_in else 'client',
                                'hand1': hand1_name,
                                'hand2': hand2_name,
                                'distance': distance,
                                'midpoint': midpoint,
                                'confidence': hand_confidence,
                                'distance_score': distance_score
                            })
        
        return proximity_events
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Enhanced Two-Step Cash Transaction Detection
        
        STEP 1: Detect Cashier-Customer Hand Touch
        - Cashier = person inside Cashier Zone
        - Customer = person outside Cashier Zone  
        - Hands must be within hand_touch_distance
        
        STEP 2: Track Cashier's Hand to Cash Drawer
        - After touch detected, track cashier's hands
        - If cashier's hand enters Cash Drawer Zone within tracking duration
        - THEN trigger CASH detection
        
        This ensures we only detect REAL cash transactions.
        """
        detections = []
        
        if not self.is_initialized:
            return detections
        
        try:
            self.frame_count += 1
            
            # Update video dimensions from frame
            h, w = frame.shape[:2]
            if w != self.video_width or h != self.video_height:
                self.update_video_dimensions(w, h)
            
            # Run pose estimation
            results = self.pose_model(frame, verbose=False, conf=self.pose_confidence)
            
            if not results or len(results) == 0:
                # No people detected - check if we should timeout pending transaction
                if self.tracking_cashier_hands:
                    self.frames_since_touch += 1
                    if self.frames_since_touch > self.hand_tracking_duration:
                        self._reset_tracking("No people detected - timeout")
                return detections
            
            result = results[0]
            
            # Extract people and their hand positions
            people_hands = []
            cashier_zone_people = []
            customer_zone_people = []
            debug_people = []
            
            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                for idx, (kpts, box) in enumerate(zip(keypoints_data, boxes)):
                    bbox = tuple(map(int, box))
                    hands = self.get_hand_positions(kpts)
                    center = self.get_person_center(kpts, bbox)
                    in_zone = self.is_person_in_cashier_zone(kpts, bbox)
                    
                    person_info = {
                        'idx': idx,
                        'bbox': bbox,
                        'hands': hands,
                        'keypoints': kpts,
                        'center': center,
                        'in_cashier_zone': in_zone
                    }
                    
                    people_hands.append(person_info)
                    
                    # Debug visualization data
                    hands_list = []
                    if 'left' in hands:
                        hands_list.append(hands['left'])
                    if 'right' in hands:
                        hands_list.append(hands['right'])
                    
                    debug_people.append({
                        'bbox': bbox,
                        'hands': hands_list,
                        'keypoints': kpts.tolist(),
                        'in_zone': in_zone,
                        'role': 'CASHIER' if in_zone else 'CLIENT'
                    })
                    
                    if in_zone:
                        cashier_zone_people.append(person_info)
                    else:
                        customer_zone_people.append(person_info)
            
            # ==================== STEP 2: CHECK CASH DRAWER DEPOSIT ====================
            # If we're tracking after a touch, check if cashier's hand goes to drawer
            if self.tracking_cashier_hands and self.pending_transaction:
                self.frames_since_touch += 1
                
                # Check timeout
                if self.frames_since_touch > self.hand_tracking_duration:
                    self._reset_tracking("Tracking timeout - no drawer deposit detected")
                else:
                    # Check if any cashier's hand is in the cash drawer zone
                    for cashier in cashier_zone_people:
                        for hand_name, hand_pos in cashier.get('hands', {}).items():
                            if self.is_in_cash_drawer_zone((hand_pos[0], hand_pos[1])):
                                # SUCCESS! Cashier deposited in drawer after customer touch
                                detection = self._create_detection(
                                    frame, 
                                    self.pending_transaction,
                                    cashier,
                                    hand_pos,
                                    "drawer_deposit"
                                )
                                if detection:
                                    detections.append(detection)
                                    print(f"[CashDetect] ✅ CASH DETECTED! Touch → Drawer deposit in {self.frames_since_touch} frames")
                                    self.last_transaction_frame = self.frame_count
                                
                                self._reset_tracking("Detection complete")
                                
                                # Update debug info with detection event
                                self.last_detection_debug = {
                                    'people': debug_people,
                                    'transaction_events': [self.pending_transaction] if self.pending_transaction else [],
                                    'num_people': len(people_hands),
                                    'num_cashier': len(cashier_zone_people),
                                    'num_client': len(customer_zone_people),
                                    'state': 'DETECTED',
                                    'tracking_frames': self.frames_since_touch
                                }
                                return detections
            
            # ==================== STEP 1: DETECT HAND TOUCH ====================
            # Only look for new touches if not already tracking
            if not self.tracking_cashier_hands:
                # Check cooldown
                if self.frame_count - self.last_transaction_frame <= self.transaction_cooldown:
                    self.last_detection_debug = {
                        'people': debug_people,
                        'transaction_events': [],
                        'num_people': len(people_hands),
                        'num_cashier': len(cashier_zone_people),
                        'num_client': len(customer_zone_people),
                        'state': 'COOLDOWN'
                    }
                    return detections
                
                # Look for hand touch between cashier and customer
                touch_events = self._detect_hand_touch(people_hands)
                
                if touch_events:
                    # Found a touch! Start tracking cashier's hand
                    best_event = min(touch_events, key=lambda x: x.get('distance', 999))
                    
                    self.pending_transaction = best_event
                    self.tracking_cashier_hands = True
                    self.frames_since_touch = 0
                    self.touch_frame = self.frame_count
                    
                    print(f"[CashDetect] 🤝 Hand touch detected! dist={best_event['distance']:.0f}px, tracking cashier for {self.hand_tracking_duration} frames")
                    
                    self.last_detection_debug = {
                        'people': debug_people,
                        'transaction_events': touch_events,
                        'num_people': len(people_hands),
                        'num_cashier': len(cashier_zone_people),
                        'num_client': len(customer_zone_people),
                        'state': 'TRACKING',
                        'touch_distance': best_event['distance']
                    }
                else:
                    self.last_detection_debug = {
                        'people': debug_people,
                        'transaction_events': [],
                        'num_people': len(people_hands),
                        'num_cashier': len(cashier_zone_people),
                        'num_client': len(customer_zone_people),
                        'state': 'WAITING'
                    }
            else:
                # Still tracking - update debug info
                self.last_detection_debug = {
                    'people': debug_people,
                    'transaction_events': [self.pending_transaction] if self.pending_transaction else [],
                    'num_people': len(people_hands),
                    'num_cashier': len(cashier_zone_people),
                    'num_client': len(customer_zone_people),
                    'state': 'TRACKING',
                    'tracking_frames': self.frames_since_touch,
                    'tracking_remaining': self.hand_tracking_duration - self.frames_since_touch
                }
        
        except Exception as e:
            print(f"⚠️ Cash detection error: {e}")
            import traceback
            traceback.print_exc()
        
        return detections
    
    def _detect_hand_touch(self, people_hands: List[Dict]) -> List[Dict]:
        """
        Detect when Customer hand touches Cashier hand.
        
        Rules:
        - Cashier = person inside Cashier Zone
        - Customer = person outside Cashier Zone
        - Distance between their hands < hand_touch_distance
        """
        touch_events = []
        
        for i, person1 in enumerate(people_hands):
            for j, person2 in enumerate(people_hands):
                if i >= j:
                    continue
                
                p1_in = person1.get('in_cashier_zone', False)
                p2_in = person2.get('in_cashier_zone', False)
                
                # Must be exactly one cashier + one customer
                if not ((p1_in and not p2_in) or (not p1_in and p2_in)):
                    continue
                
                # Check all hand combinations
                for hand1_name, hand1_pos in person1.get('hands', {}).items():
                    for hand2_name, hand2_pos in person2.get('hands', {}).items():
                        distance = self.calculate_hand_distance(hand1_pos[:2], hand2_pos[:2])
                        
                        if distance < self.hand_touch_distance:
                            midpoint = (
                                (hand1_pos[0] + hand2_pos[0]) // 2,
                                (hand1_pos[1] + hand2_pos[1]) // 2
                            )
                            
                            # Identify cashier and customer
                            if p1_in:
                                cashier_idx, customer_idx = i, j
                                cashier_hand, customer_hand = hand1_name, hand2_name
                            else:
                                cashier_idx, customer_idx = j, i
                                cashier_hand, customer_hand = hand2_name, hand1_name
                            
                            touch_events.append({
                                'cashier_idx': cashier_idx,
                                'customer_idx': customer_idx,
                                'person1_idx': i,
                                'person2_idx': j,
                                'person1_role': 'cashier' if p1_in else 'client',
                                'person2_role': 'cashier' if p2_in else 'client',
                                'cashier_hand': cashier_hand,
                                'customer_hand': customer_hand,
                                'hand1': hand1_name,
                                'hand2': hand2_name,
                                'distance': distance,
                                'midpoint': midpoint,
                                'confidence': min(hand1_pos[2], hand2_pos[2])
                            })
        
        return touch_events
    
    def _create_detection(self, frame, touch_event, cashier_info, drawer_hand_pos, detection_type) -> Optional[Detection]:
        """Create a CASH detection after successful two-step verification"""
        try:
            # Calculate confidence based on the quality of detection
            touch_distance = touch_event.get('distance', 100)
            distance_score = max(0, 1 - (touch_distance / self.hand_touch_distance))
            
            # Higher confidence if both steps completed quickly
            time_score = max(0, 1 - (self.frames_since_touch / self.hand_tracking_duration))
            
            confidence = (distance_score * 0.6 + time_score * 0.4)  # Weight touch more
            confidence = max(0.5, min(1.0, confidence))  # Clamp to 0.5-1.0
            
            # Create bounding box around the cash drawer area
            drawer_x, drawer_y = int(drawer_hand_pos[0]), int(drawer_hand_pos[1])
            tx_bbox = (
                max(0, drawer_x - 80),
                max(0, drawer_y - 80),
                min(frame.shape[1], drawer_x + 80),
                min(frame.shape[0], drawer_y + 80)
            )
            
            metadata = {
                'type': 'two_step_verification',
                'detection_type': detection_type,
                'touch_distance': round(touch_event['distance'], 1),
                'frames_to_drawer': self.frames_since_touch,
                'tracking_duration': self.hand_tracking_duration,
                'cashier_zone': self.cashier_zone,
                'cash_drawer_zone': self.cash_drawer_zone,
                'cashier_bbox': list(cashier_info['bbox']),
                'drawer_hand_position': [drawer_x, drawer_y]
            }
            
            return Detection(
                label="CASH",
                confidence=confidence,
                bbox=tx_bbox,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"⚠️ Error creating detection: {e}")
            return None
    
    def _reset_tracking(self, reason: str = ""):
        """Reset the tracking state"""
        if reason:
            print(f"[CashDetect] Tracking reset: {reason}")
        self.pending_transaction = None
        self.tracking_cashier_hands = False
        self.frames_since_touch = 0
        self.cashier_hand_history.clear()
    
    def draw_cashier_zone(self, frame: np.ndarray) -> np.ndarray:
        """Draw the cashier zone overlay on frame"""
        x, y, w, h = self.cashier_zone
        
        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # Draw label
        cv2.putText(frame, "CASHIER ZONE", (x + 5, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame
    
    def draw_pose_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw pose estimation overlay with hand positions and distances on frame.
        Shows hand positions, center point, and distance lines between people's hands.
        """
        if not self.is_initialized or self.pose_model is None:
            return frame
        
        try:
            # Run pose estimation
            results = self.pose_model(frame, verbose=False, conf=self.pose_confidence)
            
            if not results or len(results) == 0:
                return frame
            
            result = results[0]
            people_hands = []
            
            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                for idx, (kpts, box) in enumerate(zip(keypoints_data, boxes)):
                    bbox = tuple(map(int, box))
                    x1, y1, x2, y2 = bbox
                    hands = self.get_hand_positions(kpts)
                    
                    # Use CENTER POINT for zone determination
                    center = self.get_person_center(kpts, bbox)
                    in_zone = self.is_in_cashier_zone(center)
                    
                    # Color based on zone (green = cashier, orange = customer)
                    color = (0, 255, 0) if in_zone else (0, 165, 255)
                    role = "CASHIER" if in_zone else "CLIENT"
                    
                    # Draw person bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw CENTER POINT (larger, visible marker)
                    cv2.circle(frame, center, 12, color, -1)
                    cv2.circle(frame, center, 12, (255, 255, 255), 2)
                    cv2.putText(frame, "C", (center[0] - 5, center[1] + 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                    
                    # Draw role label
                    (text_w, text_h), _ = cv2.getTextSize(role, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - 22), (x1 + text_w + 6, y1), color, -1)
                    cv2.putText(frame, role, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    # Draw hand circles
                    for hand_name, hand_pos in hands.items():
                        cv2.circle(frame, (hand_pos[0], hand_pos[1]), 8, (255, 0, 255), -1)
                        cv2.circle(frame, (hand_pos[0], hand_pos[1]), 8, (255, 255, 255), 2)
                    
                    people_hands.append({
                        'idx': idx,
                        'hands': hands,
                        'in_zone': in_zone
                    })
            
            # Draw distance lines between hands of different people
            for i, p1 in enumerate(people_hands):
                for j, p2 in enumerate(people_hands):
                    if i >= j:
                        continue
                    
                    # Check if this is a valid cashier-client pair (XOR)
                    p1_in = p1.get('in_zone', False)
                    p2_in = p2.get('in_zone', False)
                    is_valid_pair = (p1_in and not p2_in) or (not p1_in and p2_in)
                    
                    for hand1_name, hand1_pos in p1.get('hands', {}).items():
                        for hand2_name, hand2_pos in p2.get('hands', {}).items():
                            # Calculate distance
                            dx = hand1_pos[0] - hand2_pos[0]
                            dy = hand1_pos[1] - hand2_pos[1]
                            distance = int(np.sqrt(dx*dx + dy*dy))
                            
                            # Color based on detection validity
                            is_close = distance < self.hand_touch_distance
                            
                            if is_close and is_valid_pair:
                                # VALID: Cashier-client close hands - GREEN
                                line_color = (0, 255, 0)
                            elif is_close and not is_valid_pair:
                                # IGNORED: Client-client or cashier-cashier - GRAY
                                line_color = (128, 128, 128)
                            else:
                                # Too far apart - RED
                                line_color = (0, 0, 255)
                            
                            # Draw line between hands
                            cv2.line(frame, (hand1_pos[0], hand1_pos[1]), 
                                    (hand2_pos[0], hand2_pos[1]), line_color, 2)
                            
                            # Draw distance label at midpoint
                            mid_x = (hand1_pos[0] + hand2_pos[0]) // 2
                            mid_y = (hand1_pos[1] + hand2_pos[1]) // 2
                            
                            # Add indicator for ignored pairs
                            if is_close and not is_valid_pair:
                                dist_text = f"{distance}px (IGNORED)"
                            else:
                                dist_text = f"{distance}px"
                            
                            (text_w, text_h), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(frame, (mid_x - 3, mid_y - text_h - 5), 
                                         (mid_x + text_w + 6, mid_y + 3), (0, 0, 0), -1)
                            cv2.putText(frame, dist_text, (mid_x, mid_y - 3), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
        
        except Exception as e:
            print(f"⚠️ Pose overlay error: {e}")
        
        return frame
