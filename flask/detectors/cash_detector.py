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
        self.min_transaction_frames = config.get('min_transaction_frames', 3)
        self.min_cash_confidence = config.get('min_cash_confidence', 0.70)  # 70% minimum
        
        # Show pose overlay with hand positions and distances
        self.show_pose_overlay = config.get('show_pose_overlay', True)
        
        # Tracking state
        self.potential_transactions = deque(maxlen=30)  # Last 30 frames
        self.consecutive_detections = 0
        self.last_transaction_frame = -100  # Cooldown tracking
        self.transaction_cooldown = 45  # frames between transactions
        
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
            
            # Load pose model for hand detection
            pose_model_path = models_dir / "yolov8n-pose.pt"
            if pose_model_path.exists():
                self.pose_model = YOLO(str(pose_model_path))
                print(f"✅ Loaded pose model: {pose_model_path}")
            else:
                # Download if not exists
                self.pose_model = YOLO("yolov8n-pose.pt")
                print("✅ Downloaded and loaded pose model")
            
            # Load person detection model as backup
            person_model_path = models_dir / "yolov8n.pt"
            if person_model_path.exists():
                self.person_model = YOLO(str(person_model_path))
            else:
                self.person_model = YOLO("yolov8n.pt")
            
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
    
    def is_box_in_cashier_zone(self, bbox: Tuple[int, int, int, int], threshold: float = 0.3) -> bool:
        """Check if a bounding box overlaps with cashier zone"""
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
        Detect when hands from different people are close together
        (indicating potential cash exchange)
        
        Uses OR logic: triggers if EITHER:
        - Distance is within hand_touch_distance threshold, OR
        - Confidence is high enough (>= min_cash_confidence)
        
        This ensures detection works even if one metric is slightly off.
        """
        proximity_events = []
        
        # Use a more relaxed distance threshold for initial detection
        # Will use actual distance for scoring later
        max_detection_distance = self.hand_touch_distance * 1.5  # 50% more lenient
        
        for i, person1 in enumerate(people_hands):
            for j, person2 in enumerate(people_hands):
                if i >= j:
                    continue
                
                # Check all hand combinations between two people
                for hand1_name, hand1_pos in person1.get('hands', {}).items():
                    for hand2_name, hand2_pos in person2.get('hands', {}).items():
                        distance = self.calculate_hand_distance(hand1_pos[:2], hand2_pos[:2])
                        hand_confidence = min(hand1_pos[2], hand2_pos[2])
                        
                        # OR logic: accept if EITHER distance OR confidence is good
                        distance_ok = distance < self.hand_touch_distance
                        confidence_ok = hand_confidence >= self.min_cash_confidence
                        distance_close_enough = distance < max_detection_distance
                        
                        # Trigger if: (distance OK) OR (confidence high AND distance reasonably close)
                        if distance_ok or (confidence_ok and distance_close_enough):
                            # Calculate midpoint of the hand interaction
                            midpoint = (
                                (hand1_pos[0] + hand2_pos[0]) // 2,
                                (hand1_pos[1] + hand2_pos[1]) // 2
                            )
                            
                            # Calculate combined score (higher is better)
                            # Normalize distance score: 1.0 when touching, 0.0 at max_detection_distance
                            distance_score = max(0, 1 - (distance / max_detection_distance))
                            combined_score = (distance_score + hand_confidence) / 2
                            
                            proximity_events.append({
                                'person1_idx': i,
                                'person2_idx': j,
                                'hand1': hand1_name,
                                'hand2': hand2_name,
                                'distance': distance,
                                'midpoint': midpoint,
                                'confidence': hand_confidence,
                                'distance_score': distance_score,
                                'combined_score': combined_score,
                                'distance_ok': distance_ok,
                                'confidence_ok': confidence_ok
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
                    
                    person_info = {
                        'idx': idx,
                        'bbox': bbox,
                        'hands': hands,
                        'in_cashier_zone': self.is_box_in_cashier_zone(bbox)
                    }
                    
                    people_hands.append(person_info)
                    
                    if person_info['in_cashier_zone']:
                        cashier_zone_people.append(person_info)
                    else:
                        customer_zone_people.append(person_info)
            
            # Look for hand proximity events
            hand_events = self.detect_hand_proximity(people_hands)
            
            # Accept ANY hand proximity event between two different people
            # No longer require one person inside and one outside cashier zone
            # Just require that at least one person is near/in the cashier zone area
            transaction_events = []
            for event in hand_events:
                p1 = people_hands[event['person1_idx']]
                p2 = people_hands[event['person2_idx']]
                
                # Accept if at least one person is in or near cashier zone
                # OR if the midpoint of the hand interaction is in the zone
                either_in_zone = p1['in_cashier_zone'] or p2['in_cashier_zone']
                midpoint_in_zone = self.is_in_cashier_zone(event['midpoint'])
                
                if either_in_zone or midpoint_in_zone:
                    transaction_events.append(event)
            
            # Track potential transactions
            self.potential_transactions.append(len(transaction_events) > 0)
            
            # Check for consistent detection
            if transaction_events:
                self.consecutive_detections += 1
            else:
                self.consecutive_detections = max(0, self.consecutive_detections - 1)
            
            # Confirm transaction after minimum frames
            if (self.consecutive_detections >= self.min_transaction_frames and 
                self.frame_count - self.last_transaction_frame > self.transaction_cooldown and
                len(transaction_events) > 0):
                
                # Find the best transaction event using combined score
                best_event = max(transaction_events, key=lambda x: x.get('combined_score', x['confidence']))
                
                # OR logic for final confirmation:
                # Accept if EITHER distance is good OR confidence is good
                distance_ok = best_event.get('distance_ok', best_event['distance'] < self.hand_touch_distance)
                confidence_ok = best_event.get('confidence_ok', best_event['confidence'] >= self.min_cash_confidence)
                
                # Must pass at least one threshold
                if not (distance_ok or confidence_ok):
                    return detections
                
                # Create bounding box around the transaction area
                mp = best_event['midpoint']
                tx_bbox = (
                    max(0, mp[0] - 60),
                    max(0, mp[1] - 60),
                    min(frame.shape[1], mp[0] + 60),
                    min(frame.shape[0], mp[1] + 60)
                )
                
                # Use combined score as the reported confidence
                reported_confidence = best_event.get('combined_score', best_event['confidence'])
                
                detection = Detection(
                    label="CASH",
                    confidence=reported_confidence,
                    bbox=tx_bbox,
                    metadata={
                        'type': 'hand_exchange',
                        'distance': best_event['distance'],
                        'hand_confidence': best_event['confidence'],
                        'distance_ok': distance_ok,
                        'confidence_ok': confidence_ok,
                        'people_count': len(people_hands),
                        'cashier_zone': self.cashier_zone
                    }
                )
                detections.append(detection)
                
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
        Shows hand positions and distance lines between people's hands.
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
                    in_zone = self.is_box_in_cashier_zone(bbox)
                    
                    # Color based on zone (green = cashier, orange = customer)
                    color = (0, 255, 0) if in_zone else (0, 165, 255)
                    role = "CASHIER" if in_zone else "CLIENT"
                    
                    # Draw person bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
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
                    
                    for hand1_name, hand1_pos in p1.get('hands', {}).items():
                        for hand2_name, hand2_pos in p2.get('hands', {}).items():
                            # Calculate distance
                            dx = hand1_pos[0] - hand2_pos[0]
                            dy = hand1_pos[1] - hand2_pos[1]
                            distance = int(np.sqrt(dx*dx + dy*dy))
                            
                            # Color based on distance threshold
                            is_close = distance < self.hand_touch_distance
                            line_color = (0, 255, 0) if is_close else (0, 0, 255)
                            
                            # Draw line between hands
                            cv2.line(frame, (hand1_pos[0], hand1_pos[1]), 
                                    (hand2_pos[0], hand2_pos[1]), line_color, 2)
                            
                            # Draw distance label at midpoint
                            mid_x = (hand1_pos[0] + hand2_pos[0]) // 2
                            mid_y = (hand1_pos[1] + hand2_pos[1]) // 2
                            
                            dist_text = f"{distance}px"
                            (text_w, text_h), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                            cv2.rectangle(frame, (mid_x - 3, mid_y - text_h - 5), 
                                         (mid_x + text_w + 6, mid_y + 3), (0, 0, 0), -1)
                            cv2.putText(frame, dist_text, (mid_x, mid_y - 3), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
        
        except Exception as e:
            print(f"⚠️ Pose overlay error: {e}")
        
        return frame
