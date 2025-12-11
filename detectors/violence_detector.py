"""
Violence Detector

Detects violent actions using:
1. Physical contact between TWO or more people
2. Rapid aggressive movements while in close proximity
3. Fall detection after contact

STRICT RULES:
- Single person actions are NEVER violence
- Requires TWO people very close together
- Requires sustained aggressive motion between them
- Normal walking, waving, reaching = NOT violence
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from .base_detector import BaseDetector, Detection


class ViolenceDetector(BaseDetector):
    """
    Detects violence using pose estimation and motion analysis.
    
    STRICT DETECTION CRITERIA:
    1. Minimum 2 people required
    2. People must be in VERY close contact (overlapping bboxes)
    3. HIGH motion from BOTH people simultaneously  
    4. Sustained over many frames (min_violence_frames)
    
    WHAT IS NOT VIOLENCE:
    - One person raising arms (waving, stretching)
    - One person moving fast (running, exercising)
    - People standing close but not moving aggressively
    - Normal customer-cashier interactions
    """
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.pose_model = None
        
        # Detection parameters - EXTREMELY strict to reduce false positives
        self.violence_confidence = config.get('violence_confidence', 0.85)
        self.min_violence_frames = config.get('min_violence_frames', 20)  # Need 20+ frames
        self.motion_threshold = config.get('motion_threshold', 150)  # Very high motion
        
        # Cashier zone exclusion (normal transactions shouldn't trigger violence)
        self.cashier_zone = config.get('cashier_zone', None)
        
        # Tracking state
        self.previous_keypoints = {}
        self.previous_positions = {}  # Track bbox positions for motion
        self.consecutive_violence = 0
        self.last_violence_frame = -100
        self.violence_cooldown = 150  # Long cooldown between alerts
        
        # Motion history per person
        self.person_motion_history = {}  # person_id -> deque of motion values
        
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
            import torch
            
            # Check GPU availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[GPU] Violence detector using: {device}")
            if torch.cuda.is_available():
                print(f"      GPU: {torch.cuda.get_device_name(0)}")
            
            models_dir = Path(self.config.get('models_dir', 'models'))
            
            # Get pose model name from config (default to yolov8s-pose for better accuracy)
            pose_model_name = self.config.get('pose_model', 'yolov8s-pose.pt')
            
            pose_model_path = models_dir / pose_model_name
            if pose_model_path.exists():
                self.pose_model = YOLO(str(pose_model_path))
                self.pose_model.to(device)  # Move to GPU
                print(f"[OK] Violence detector loaded pose model: {pose_model_path} on {device}")
            else:
                self.pose_model = YOLO(pose_model_name)
                self.pose_model.to(device)  # Move to GPU
                print(f"[OK] Violence detector downloaded pose model: {pose_model_name} on {device}")
            
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
    
    def check_bbox_overlap(self, box1: Tuple, box2: Tuple) -> float:
        """Check how much two bounding boxes overlap (0-1 ratio)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        min_area = min(area1, area2)
        if min_area <= 0:
            return 0.0
        
        return intersection / min_area
    
    def detect_physical_altercation(self, people: List[Dict]) -> List[Dict]:
        """
        Detect physical fighting between two people
        
        STRICT REQUIREMENTS:
        1. Two people with OVERLAPPING bounding boxes (physically close)
        2. BOTH people have high motion (not just one person moving)
        3. Not in cashier zone (avoid false positives from transactions)
        """
        altercations = []
        
        if len(people) < 2:
            return altercations
        
        for i, person1 in enumerate(people):
            for j, person2 in enumerate(people):
                if i >= j:
                    continue
                
                # Skip if either person is in cashier zone
                if person1.get('in_cashier_zone') or person2.get('in_cashier_zone'):
                    continue
                
                box1 = person1['bbox']
                box2 = person2['bbox']
                
                # Check for significant overlap (people physically touching)
                overlap = self.check_bbox_overlap(box1, box2)
                
                # Require at least 20% overlap (people very close/touching)
                if overlap < 0.20:
                    continue
                
                # Get motion for both people
                motion1 = person1.get('avg_motion', 0)
                motion2 = person2.get('avg_motion', 0)
                
                # BOTH must be moving aggressively (not just one attacking)
                if motion1 < self.motion_threshold or motion2 < self.motion_threshold:
                    continue
                
                # Calculate violence confidence based on overlap and motion
                motion_score = min(1.0, (motion1 + motion2) / (self.motion_threshold * 4))
                overlap_score = min(1.0, overlap * 2)  # Scale overlap
                
                confidence = (motion_score * 0.6) + (overlap_score * 0.4)
                
                if confidence >= 0.7:  # High threshold
                    combined_bbox = (
                        min(box1[0], box2[0]),
                        min(box1[1], box2[1]),
                        max(box1[2], box2[2]),
                        max(box1[3], box2[3])
                    )
                    
                    altercations.append({
                        'person1': i,
                        'person2': j,
                        'overlap': overlap,
                        'motion1': motion1,
                        'motion2': motion2,
                        'confidence': confidence,
                        'bbox': combined_bbox
                    })
        
        return altercations
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect violence in the frame
        
        STRICT LOGIC - Only triggers on actual physical fighting:
        1. Must have 2+ people with overlapping bounding boxes
        2. BOTH people must have high motion (fighting involves both parties)
        3. Must be sustained over many consecutive frames
        4. Excludes cashier zone interactions
        """
        detections = []
        
        if not self.is_initialized:
            return detections
        
        try:
            # Run pose estimation
            results = self.pose_model(frame, verbose=False)
            
            if not results or len(results) == 0:
                self.consecutive_violence = max(0, self.consecutive_violence - 2)
                return detections
            
            result = results[0]
            people = []
            
            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                
                for idx, (kpts, box) in enumerate(zip(keypoints_data, boxes)):
                    bbox = tuple(map(int, box))
                    person_id = f"person_{idx}"
                    
                    # Calculate motion from previous frame
                    current_motion = 0.0
                    if person_id in self.previous_keypoints:
                        current_motion = self.calculate_motion(kpts, self.previous_keypoints[person_id])
                    self.previous_keypoints[person_id] = kpts.copy()
                    
                    # Track motion history for this person (average over last 5 frames)
                    if person_id not in self.person_motion_history:
                        self.person_motion_history[person_id] = deque(maxlen=5)
                    self.person_motion_history[person_id].append(current_motion)
                    
                    avg_motion = np.mean(self.person_motion_history[person_id]) if self.person_motion_history[person_id] else 0
                    
                    # Check if in cashier zone
                    in_cashier = self.is_in_cashier_zone(bbox)
                    
                    person_info = {
                        'idx': idx,
                        'bbox': bbox,
                        'keypoints': kpts,
                        'current_motion': current_motion,
                        'avg_motion': avg_motion,
                        'in_cashier_zone': in_cashier
                    }
                    people.append(person_info)
            
            # Clean up old person tracking (remove people not seen for a while)
            current_ids = {f"person_{p['idx']}" for p in people}
            old_ids = set(self.person_motion_history.keys()) - current_ids
            for old_id in old_ids:
                del self.person_motion_history[old_id]
                if old_id in self.previous_keypoints:
                    del self.previous_keypoints[old_id]
            
            # Detect physical altercations (two people fighting)
            altercations = self.detect_physical_altercation(people)
            
            # Update consecutive violence counter
            if altercations:
                self.consecutive_violence += 1
            else:
                # Decay faster when no violence detected
                self.consecutive_violence = max(0, self.consecutive_violence - 2)
            
            # Generate detection only after sustained violence over many frames
            if (self.consecutive_violence >= self.min_violence_frames and
                self.frame_count - self.last_violence_frame > self.violence_cooldown and
                len(altercations) > 0):
                
                # Find the highest confidence altercation
                best = max(altercations, key=lambda x: x['confidence'])
                
                # Only report if confidence meets threshold
                if best['confidence'] >= self.violence_confidence:
                    detection = Detection(
                        label="VIOLENCE",
                        confidence=best['confidence'],
                        bbox=best['bbox'],
                        metadata={
                            'type': 'physical_altercation',
                            'overlap': best['overlap'],
                            'motion1': best['motion1'],
                            'motion2': best['motion2'],
                            'people_count': len(people),
                            'consecutive_frames': self.consecutive_violence
                        }
                    )
                    detections.append(detection)
                    
                    self.last_violence_frame = self.frame_count
                    self.consecutive_violence = 0
        
        except Exception as e:
            print(f"[Violence] Detection error: {e}")
        
        return detections
