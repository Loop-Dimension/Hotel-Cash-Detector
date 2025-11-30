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
        
        # Cashier zone [x, y, width, height]
        self.cashier_zone = config.get('cashier_zone', [100, 100, 400, 300])
        
        # Detection parameters
        self.hand_touch_distance = config.get('hand_touch_distance', 100)
        self.pose_confidence = config.get('pose_confidence', 0.5)
        self.min_transaction_frames = config.get('min_transaction_frames', 3)
        self.min_cash_confidence = config.get('min_cash_confidence', 0.70)  # 70% minimum
        
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
    
    def set_cashier_zone(self, zone: List[int]):
        """Update the cashier zone"""
        self.cashier_zone = zone
    
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
        """
        proximity_events = []
        
        for i, person1 in enumerate(people_hands):
            for j, person2 in enumerate(people_hands):
                if i >= j:
                    continue
                
                # Check all hand combinations between two people
                for hand1_name, hand1_pos in person1.get('hands', {}).items():
                    for hand2_name, hand2_pos in person2.get('hands', {}).items():
                        distance = self.calculate_hand_distance(hand1_pos[:2], hand2_pos[:2])
                        
                        if distance < self.hand_touch_distance:
                            # Calculate midpoint of the hand interaction
                            midpoint = (
                                (hand1_pos[0] + hand2_pos[0]) // 2,
                                (hand1_pos[1] + hand2_pos[1]) // 2
                            )
                            
                            proximity_events.append({
                                'person1_idx': i,
                                'person2_idx': j,
                                'hand1': hand1_name,
                                'hand2': hand2_name,
                                'distance': distance,
                                'midpoint': midpoint,
                                'confidence': min(hand1_pos[2], hand2_pos[2])
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
            
            # Filter for events: one person INSIDE cashier zone, one OUTSIDE
            # This represents customer handing cash to cashier (or vice versa)
            transaction_events = []
            for event in hand_events:
                p1 = people_hands[event['person1_idx']]
                p2 = people_hands[event['person2_idx']]
                
                # One person must be IN zone, other must be OUTSIDE
                # XOR: exactly one of them in the cashier zone
                if p1['in_cashier_zone'] != p2['in_cashier_zone']:
                    # And the hand interaction should be near the zone boundary
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
                
                # Find the best transaction event
                best_event = max(transaction_events, key=lambda x: x['confidence'])
                
                # Only alert if confidence >= 70%
                if best_event['confidence'] < self.min_cash_confidence:
                    return detections
                
                # Create bounding box around the transaction area
                mp = best_event['midpoint']
                tx_bbox = (
                    max(0, mp[0] - 60),
                    max(0, mp[1] - 60),
                    min(frame.shape[1], mp[0] + 60),
                    min(frame.shape[0], mp[1] + 60)
                )
                
                detection = Detection(
                    label="CASH",
                    confidence=best_event['confidence'],
                    bbox=tx_bbox,
                    metadata={
                        'type': 'hand_exchange',
                        'distance': best_event['distance'],
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
