"""
Cash Transaction Detector

Detects cash transactions by analyzing:
1. Person presence in cashier zone
2. Hand positions and movements using pose estimation
3. Hand-to-hand proximity between people (cash handoff)
4. Extended hand gestures typical of payment
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from .base_detector import BaseDetector, Detection


class CashTransactionDetector(BaseDetector):
    """
    Detects cash transactions using pose estimation and spatial analysis.
    
    Since we can't detect Korean currency directly (no trained model),
    we detect the ACTION of cash exchange:
    - Two people near cashier zone
    - Hands coming together / close proximity
    - Hand-to-hand movement patterns
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.pose_model = None
        self.person_model = None
        
        # Cashier zone as percentage [x%, y%, width%, height%] - responsive to video size
        # Default zone covers center-bottom area (typical cashier position)
        self.cashier_zone_percent = config.get('cashier_zone_percent', [0.1, 0.3, 0.8, 0.6])
        
        # Legacy pixel-based zone (will be converted to percentage if video size known)
        self.cashier_zone = config.get('cashier_zone', [100, 100, 400, 300])
        
        # Video dimensions (updated when processing frames)
        self.video_width = config.get('video_width', 1920)
        self.video_height = config.get('video_height', 1080)
        
        # Detection parameters
        self.hand_touch_distance = config.get('hand_touch_distance', 100)
        self.pose_confidence = config.get('pose_confidence', 0.5)
        self.min_transaction_frames = config.get('min_transaction_frames', 1)  # Immediate detection
        self.min_cash_confidence = config.get('min_cash_confidence', 0.70)  # 70% minimum
        
        # Show pose overlay with hand positions and distances (disabled by default, debug only)
        self.show_pose_overlay = config.get('show_pose_overlay', False)
        
        # Tracking state
        self.potential_transactions = deque(maxlen=30)  # Last 30 frames
        self.consecutive_detections = 0
        self.last_transaction_frame = -100  # Cooldown tracking
        self.transaction_cooldown = 15  # frames between transactions (reduced from 45)
        
        # Hand keypoint indices for COCO format (used by YOLOv8-pose)
        # 9: left_wrist, 10: right_wrist
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
            
            # Load pose model for hand detection
            pose_model_path = models_dir / pose_model_name
            if pose_model_path.exists():
                self.pose_model = YOLO(str(pose_model_path))
                print(f"✅ Loaded pose model: {pose_model_path}")
            else:
                # Download if not exists
                self.pose_model = YOLO(pose_model_name)
                print(f"✅ Downloaded and loaded pose model: {pose_model_name}")
            
            # Load person detection model as backup
            person_model_path = models_dir / yolo_model_name
            if person_model_path.exists():
                self.person_model = YOLO(str(person_model_path))
            else:
                self.person_model = YOLO(yolo_model_name)
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize CashTransactionDetector: {e}")
            return False
    
    def update_video_dimensions(self, width: int, height: int):
        """Update video dimensions and recalculate pixel zone from percentage"""
        self.video_width = width
        self.video_height = height
        # Convert percentage zone to pixel zone for current video size
        self.cashier_zone = self.percent_to_pixels(self.cashier_zone_percent)
    
    def percent_to_pixels(self, zone_percent: List[float]) -> List[int]:
        """Convert percentage-based zone to pixel coordinates"""
        px, py, pw, ph = zone_percent
        return [
            int(px * self.video_width),
            int(py * self.video_height),
            int(pw * self.video_width),
            int(ph * self.video_height)
        ]
    
    def pixels_to_percent(self, zone_pixels: List[int]) -> List[float]:
        """Convert pixel-based zone to percentage coordinates"""
        x, y, w, h = zone_pixels
        return [
            x / self.video_width if self.video_width > 0 else 0,
            y / self.video_height if self.video_height > 0 else 0,
            w / self.video_width if self.video_width > 0 else 0,
            h / self.video_height if self.video_height > 0 else 0
        ]
    
    def set_cashier_zone(self, zone: List[int], as_percent: bool = False):
        """Update the cashier zone (can be pixels or percentage)"""
        if as_percent:
            self.cashier_zone_percent = zone
            self.cashier_zone = self.percent_to_pixels(zone)
        else:
            self.cashier_zone = zone
            self.cashier_zone_percent = self.pixels_to_percent(zone)
    
    def set_hand_touch_distance(self, distance: int):
        """Update hand touch distance threshold"""
        self.hand_touch_distance = max(10, min(500, distance))  # Clamp between 10-500px
    
    def is_in_cashier_zone(self, point: Tuple[int, int]) -> bool:
        """Check if a point is inside the cashier zone"""
        x, y = point
        zx, zy, zw, zh = self.cashier_zone
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
        Check if a person is in the cashier zone using their CENTER POINT.
        This ensures one person = one classification (not split between zones).
        """
        center = self.get_person_center(keypoints, bbox)
        return self.is_in_cashier_zone(center)
    
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
        Detect cash transactions in the frame
        
        Detection strategy:
        1. Find all people using pose estimation
        2. Identify people in/near cashier zone
        3. Track hand positions
        4. Detect hand-to-hand proximity events
        5. Confirm transaction after consecutive detections
        """
        detections = []
        
        if not self.is_initialized:
            return detections
        
        try:
            # Update video dimensions from frame (for responsive zone)
            h, w = frame.shape[:2]
            if w != self.video_width or h != self.video_height:
                self.update_video_dimensions(w, h)
            
            # Run pose estimation
            results = self.pose_model(frame, verbose=False, conf=self.pose_confidence)
            
            if not results or len(results) == 0:
                self.consecutive_detections = 0
                return detections
            
            result = results[0]
            
            # Extract people and their hand positions
            people_hands = []
            cashier_zone_people = []
            customer_zone_people = []
            
            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                for idx, (kpts, box) in enumerate(zip(keypoints_data, boxes)):
                    bbox = tuple(map(int, box))
                    hands = self.get_hand_positions(kpts)
                    
                    # Use CENTER POINT to determine if person is in cashier zone
                    # This ensures one person = one classification (not split)
                    center = self.get_person_center(kpts, bbox)
                    in_zone = self.is_in_cashier_zone(center)
                    
                    person_info = {
                        'idx': idx,
                        'bbox': bbox,
                        'hands': hands,
                        'keypoints': kpts,
                        'center': center,
                        'in_cashier_zone': in_zone
                    }
                    
                    people_hands.append(person_info)
                    
                    if person_info['in_cashier_zone']:
                        cashier_zone_people.append(person_info)
                    else:
                        customer_zone_people.append(person_info)
            
            # Look for hand proximity events between CASHIER and CLIENT only
            # detect_hand_proximity now enforces strict XOR (cashier+client)
            transaction_events = self.detect_hand_proximity(people_hands)
            
            # Track potential transactions
            self.potential_transactions.append(len(transaction_events) > 0)
            
            # Check for consistent detection - increment BEFORE checking threshold
            if transaction_events:
                self.consecutive_detections += 1
                
                # Debug: Log detection attempt
                best = min(transaction_events, key=lambda x: x.get('distance', 999))
                cooldown_active = self.frame_count - self.last_transaction_frame <= self.transaction_cooldown
                print(f"[CashDetect] Hand touch: dist={best['distance']:.0f} ({best['person1_role']}-{best['person2_role']}), consec={self.consecutive_detections}, cooldown={cooldown_active}")
            else:
                self.consecutive_detections = max(0, self.consecutive_detections - 1)
            
            # Confirm transaction after minimum frames
            # FIXED: Check consecutive_detections >= 1 (we just incremented it above)
            if (self.consecutive_detections >= self.min_transaction_frames and 
                self.frame_count - self.last_transaction_frame > self.transaction_cooldown and
                len(transaction_events) > 0):
                
                # Find the best transaction event using distance score
                best_event = max(transaction_events, key=lambda x: x.get('distance_score', 0))
                
                # STRICT: Only accept if distance is within threshold (already filtered above)
                # No OR logic - distance must be satisfied
                
                # Create bounding box around the transaction area
                mp = best_event['midpoint']
                tx_bbox = (
                    max(0, mp[0] - 60),
                    max(0, mp[1] - 60),
                    min(frame.shape[1], mp[0] + 60),
                    min(frame.shape[0], mp[1] + 60)
                )
                
                # Use distance score as confidence (closer hands = higher confidence)
                reported_confidence = best_event.get('distance_score', best_event['confidence'])
                
                detection = Detection(
                    label="CASH",
                    confidence=reported_confidence,
                    bbox=tx_bbox,
                    metadata={
                        'type': 'hand_exchange',
                        'distance': best_event['distance'],
                        'hand_confidence': best_event['confidence'],
                        'distance_threshold': self.hand_touch_distance,
                        'people_count': len(people_hands),
                        'cashier_zone': self.cashier_zone
                    }
                )
                detections.append(detection)
                print(f"[CashDetect] ✅ DETECTION TRIGGERED! dist={best_event['distance']:.0f}, conf={reported_confidence:.2f}")
                
                self.last_transaction_frame = self.frame_count
                self.consecutive_detections = 0
            
            # Also detect if only cashier is present with extended hands
            # (receiving money from someone off-screen or far away)
            if not transaction_events and len(cashier_zone_people) > 0:
                for person in cashier_zone_people:
                    hands = person['hands']
                    if hands:
                        # Check if hands are extended (far from body center)
                        bbox = person['bbox']
                        body_center_x = (bbox[0] + bbox[2]) // 2
                        
                        for hand_name, hand_pos in hands.items():
                            # If hand is extended outward significantly
                            if abs(hand_pos[0] - body_center_x) > 100:
                                # Potential receiving gesture - track but don't alert yet
                                pass
        
        except Exception as e:
            print(f"⚠️ Cash detection error: {e}")
        
        return detections
    
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
