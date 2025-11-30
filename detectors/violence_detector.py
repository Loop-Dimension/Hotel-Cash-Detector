"""
Violence Detector

Detects violent actions using:
1. Pose-based action recognition (fighting poses, rapid movements)
2. Person proximity and interaction patterns
3. Sudden movement detection
4. Body posture analysis
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from .base_detector import BaseDetector, Detection


class ViolenceDetector(BaseDetector):
    """
    Detects violence using pose estimation and motion analysis.
    
    Detection methods:
    1. Two or more people in close physical contact with rapid motion
    2. Fall detection (person suddenly on ground)
    3. Sudden chaotic motion patterns
    
    IMPROVED: Requires MULTIPLE indicators to confirm violence
    - Single raised arm = NOT violence (could be waving, reaching)
    - Single person moving fast = NOT violence (could be running)
    - Two people close + aggressive poses + rapid motion = VIOLENCE
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.pose_model = None
        
        # Detection parameters - VERY strict to reduce false positives
        self.violence_confidence = config.get('violence_confidence', 0.80)
        self.min_violence_frames = config.get('min_violence_frames', 15)  # Need sustained detection
        self.motion_threshold = config.get('motion_threshold', 100)  # High motion threshold
        
        # Cashier zone exclusion (normal transactions shouldn't trigger violence)
        self.cashier_zone = config.get('cashier_zone', None)
        
        # Tracking state
        self.previous_keypoints = {}  # Track keypoints per person
        self.consecutive_violence = 0
        self.last_violence_frame = -100
        self.violence_cooldown = 90  # frames between alerts
        
        # Motion history
        self.motion_history = deque(maxlen=15)
        
        # Keypoint indices (COCO format)
        self.NOSE = 0
        self.LEFT_SHOULDER = 5
        self.RIGHT_SHOULDER = 6
        self.LEFT_ELBOW = 7
        self.RIGHT_ELBOW = 8
        self.LEFT_WRIST = 9
        self.RIGHT_WRIST = 10
        self.LEFT_HIP = 11
        self.RIGHT_HIP = 12
        
    def initialize(self) -> bool:
        """Load YOLO pose model"""
        try:
            from ultralytics import YOLO
            from pathlib import Path
            
            models_dir = Path(self.config.get('models_dir', 'models'))
            
            pose_model_path = models_dir / "yolov8n-pose.pt"
            if pose_model_path.exists():
                self.pose_model = YOLO(str(pose_model_path))
            else:
                self.pose_model = YOLO("yolov8n-pose.pt")
            
            print("[OK] Violence detector initialized")
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize ViolenceDetector: {e}")
            return False
    
    def set_cashier_zone(self, zone: List[int]):
        """Set cashier zone for exclusion from violence detection"""
        self.cashier_zone = zone
    
    def is_in_cashier_zone(self, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if bounding box is mostly inside cashier zone"""
        if self.cashier_zone is None:
            return False
        
        x1, y1, x2, y2 = bbox
        zx, zy, zw, zh = self.cashier_zone
        
        # Calculate center of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Check if center is in cashier zone
        return zx <= center_x <= zx + zw and zy <= center_y <= zy + zh
    
    def calculate_motion(self, current_kpts: np.ndarray, previous_kpts: np.ndarray) -> float:
        """Calculate average motion between keypoint sets"""
        if current_kpts is None or previous_kpts is None:
            return 0.0
        
        if len(current_kpts) != len(previous_kpts):
            return 0.0
        
        total_motion = 0.0
        valid_points = 0
        
        for i, (curr, prev) in enumerate(zip(current_kpts, previous_kpts)):
            if len(curr) >= 3 and len(prev) >= 3:
                if curr[2] > 0.3 and prev[2] > 0.3:  # Both points visible
                    motion = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                    total_motion += motion
                    valid_points += 1
        
        return total_motion / valid_points if valid_points > 0 else 0.0
    
    def detect_aggressive_pose(self, keypoints: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detect aggressive body poses
        
        Returns: (is_aggressive, confidence, pose_type)
        """
        if keypoints is None or len(keypoints) < 13:
            return False, 0.0, ""
        
        try:
            # Get key body parts
            nose = keypoints[self.NOSE] if len(keypoints) > self.NOSE else None
            l_shoulder = keypoints[self.LEFT_SHOULDER] if len(keypoints) > self.LEFT_SHOULDER else None
            r_shoulder = keypoints[self.RIGHT_SHOULDER] if len(keypoints) > self.RIGHT_SHOULDER else None
            l_elbow = keypoints[self.LEFT_ELBOW] if len(keypoints) > self.LEFT_ELBOW else None
            r_elbow = keypoints[self.RIGHT_ELBOW] if len(keypoints) > self.RIGHT_ELBOW else None
            l_wrist = keypoints[self.LEFT_WRIST] if len(keypoints) > self.LEFT_WRIST else None
            r_wrist = keypoints[self.RIGHT_WRIST] if len(keypoints) > self.RIGHT_WRIST else None
            l_hip = keypoints[self.LEFT_HIP] if len(keypoints) > self.LEFT_HIP else None
            r_hip = keypoints[self.RIGHT_HIP] if len(keypoints) > self.RIGHT_HIP else None
            
            poses_detected = []
            
            # Check for raised arms (fighting pose) - stricter thresholds
            # Require wrist to be significantly above shoulder (not just reaching for something)
            if l_wrist is not None and l_shoulder is not None:
                if l_wrist[2] > 0.5 and l_shoulder[2] > 0.5:  # Higher confidence required
                    if l_wrist[1] < l_shoulder[1] - 80:  # Wrist well above shoulder
                        poses_detected.append(("raised_arm_left", 0.6))
            
            if r_wrist is not None and r_shoulder is not None:
                if r_wrist[2] > 0.5 and r_shoulder[2] > 0.5:  # Higher confidence required
                    if r_wrist[1] < r_shoulder[1] - 80:  # Wrist well above shoulder
                        poses_detected.append(("raised_arm_right", 0.6))
            
            # Check for both arms raised (very aggressive)
            if len([p for p in poses_detected if "raised_arm" in p[0]]) >= 2:
                poses_detected.append(("both_arms_raised", 0.9))
            
            # Check for punching motion (elbow bent, wrist extended forward)
            if l_elbow is not None and l_wrist is not None and l_shoulder is not None:
                if all(p[2] > 0.3 for p in [l_elbow, l_wrist, l_shoulder]):
                    # Arm extended forward (elbow somewhat straight)
                    arm_length = np.sqrt((l_wrist[0] - l_shoulder[0])**2 + (l_wrist[1] - l_shoulder[1])**2)
                    if arm_length > 150:  # Extended arm
                        poses_detected.append(("punch_left", 0.8))
            
            if r_elbow is not None and r_wrist is not None and r_shoulder is not None:
                if all(p[2] > 0.3 for p in [r_elbow, r_wrist, r_shoulder]):
                    arm_length = np.sqrt((r_wrist[0] - r_shoulder[0])**2 + (r_wrist[1] - r_shoulder[1])**2)
                    if arm_length > 150:
                        poses_detected.append(("punch_right", 0.8))
            
            # Check for person on ground (fall detection)
            if nose is not None and l_hip is not None and r_hip is not None:
                if nose[2] > 0.3 and (l_hip[2] > 0.3 or r_hip[2] > 0.3):
                    hip_y = (l_hip[1] + r_hip[1]) / 2 if l_hip[2] > 0.3 and r_hip[2] > 0.3 else max(l_hip[1], r_hip[1])
                    # If head is close to hip level (person is horizontal/falling)
                    if abs(nose[1] - hip_y) < 50:
                        poses_detected.append(("person_down", 0.85))
            
            if poses_detected:
                best_pose = max(poses_detected, key=lambda x: x[1])
                return True, best_pose[1], best_pose[0]
            
        except Exception as e:
            pass
        
        return False, 0.0, ""
    
    def detect_close_combat(self, people: List[Dict]) -> List[Dict]:
        """
        Detect when two people are in physical altercation
        
        STRICT REQUIREMENTS for violence:
        - Two people VERY close (overlapping or nearly)
        - BOTH have high motion OR aggressive poses
        - Not in cashier zone
        """
        combat_events = []
        
        for i, person1 in enumerate(people):
            for j, person2 in enumerate(people):
                if i >= j:
                    continue
                
                # Skip if either person is in cashier zone
                if person1.get('in_cashier_zone') or person2.get('in_cashier_zone'):
                    continue
                
                # Calculate distance between people
                box1 = person1['bbox']
                box2 = person2['bbox']
                
                center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
                center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
                
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # People must be VERY close (practically touching)
                if distance < 100:
                    motion1 = person1.get('motion', 0)
                    motion2 = person2.get('motion', 0)
                    aggr1 = person1.get('aggressive', False)
                    aggr2 = person2.get('aggressive', False)
                    
                    # REQUIRE: Both people moving fast OR both aggressive
                    both_moving = motion1 > self.motion_threshold and motion2 > self.motion_threshold
                    both_aggressive = aggr1 and aggr2
                    mixed_indicators = (aggr1 or aggr2) and (motion1 > self.motion_threshold or motion2 > self.motion_threshold)
                    
                    if both_moving or both_aggressive or mixed_indicators:
                        confidence = 0.0
                        if both_aggressive:
                            confidence = 0.9
                        elif both_moving:
                            confidence = 0.85
                        elif mixed_indicators:
                            confidence = 0.75
                        
                        combat_events.append({
                            'person1': i,
                            'person2': j,
                            'distance': distance,
                            'center': ((center1[0] + center2[0]) // 2, 
                                      (center1[1] + center2[1]) // 2),
                            'confidence': confidence,
                            'type': 'close_combat'
                        })
        
        return combat_events
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect violence in the frame
        
        IMPROVED LOGIC:
        - Only trigger on CLOSE COMBAT between two people
        - Single person actions (raised arms, running) are NOT violence
        - Fall detection only when combined with other indicators
        """
        detections = []
        
        if not self.is_initialized:
            return detections
        
        try:
            # Run pose estimation
            results = self.pose_model(frame, verbose=False)
            
            if not results or len(results) == 0:
                self.consecutive_violence = max(0, self.consecutive_violence - 1)
                return detections
            
            result = results[0]
            
            people = []
            violence_indicators = []
            
            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                for idx, (kpts, box) in enumerate(zip(keypoints_data, boxes)):
                    bbox = tuple(map(int, box))
                    
                    # Calculate motion from previous frame
                    motion = 0.0
                    person_id = f"person_{idx}"
                    if person_id in self.previous_keypoints:
                        motion = self.calculate_motion(kpts, self.previous_keypoints[person_id])
                    self.previous_keypoints[person_id] = kpts.copy()
                    
                    # Check for aggressive poses
                    is_aggressive, aggr_conf, pose_type = self.detect_aggressive_pose(kpts)
                    
                    # Check if in cashier zone
                    in_cashier = self.is_in_cashier_zone(bbox)
                    
                    person_info = {
                        'idx': idx,
                        'bbox': bbox,
                        'keypoints': kpts,
                        'motion': motion,
                        'aggressive': is_aggressive,
                        'aggression_conf': aggr_conf,
                        'pose_type': pose_type,
                        'in_cashier_zone': in_cashier
                    }
                    people.append(person_info)
            
            # ONLY detect close combat (two people fighting)
            # Single person indicators are NOT enough for violence
            combat_events = self.detect_close_combat(people)
            
            for event in combat_events:
                p1 = people[event['person1']]
                p2 = people[event['person2']]
                
                # Create bounding box around both people
                combined_bbox = (
                    min(p1['bbox'][0], p2['bbox'][0]),
                    min(p1['bbox'][1], p2['bbox'][1]),
                    max(p1['bbox'][2], p2['bbox'][2]),
                    max(p1['bbox'][3], p2['bbox'][3])
                )
                
                violence_indicators.append({
                    'type': 'close_combat',
                    'confidence': event['confidence'],
                    'bbox': combined_bbox,
                    'center': event['center']
                })
            
            # Track violence indicators
            self.motion_history.append(len(violence_indicators) > 0)
            
            # Check for sustained violence indicators
            if violence_indicators:
                self.consecutive_violence += 1
            else:
                self.consecutive_violence = max(0, self.consecutive_violence - 1)
            
            # Generate detection after sustained indicators
            if (self.consecutive_violence >= self.min_violence_frames and
                self.frame_count - self.last_violence_frame > self.violence_cooldown and
                len(violence_indicators) > 0):
                
                # Find the highest confidence indicator
                best_indicator = max(violence_indicators, key=lambda x: x['confidence'])
                
                detection = Detection(
                    label="VIOLENCE",
                    confidence=best_indicator['confidence'],
                    bbox=best_indicator['bbox'],
                    metadata={
                        'type': best_indicator['type'],
                        'people_count': len(people),
                        'indicators_count': len(violence_indicators)
                    }
                )
                detections.append(detection)
                
                self.last_violence_frame = self.frame_count
                self.consecutive_violence = 0
        
        except Exception as e:
            print(f"⚠️ Violence detection error: {e}")
        
        return detections
