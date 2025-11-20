"""
SIMPLE HAND TOUCH DETECTOR - Step by Step
When hands of P1 touch P2 = Money Transaction
"""

from ultralytics import YOLO
import cv2
import numpy as np
import math
from pathlib import Path
import time
import json

class SimpleHandTouchConfig:
    """Simple configuration"""
    POSE_MODEL = 'models/yolov8s-pose.pt'
    
    # Simple rule: hands close = transaction
    HAND_TOUCH_DISTANCE = 80  # pixels - hands must be VERY close
    POSE_CONFIDENCE = 0.5
    
    # Cashier detection - Position-based (person at bottom of frame)
    # Person with highest Y value (bottom of frame) is identified as cashier (P1)
    # This works well for overhead cameras where cashier is in foreground
    
    # Temporal filtering - reduce false positives
    MIN_TRANSACTION_FRAMES = 3  # Must last at least 5 frames to be real
    
    # Visualization
    DRAW_HANDS = True
    DRAW_CONNECTIONS = True
    DEBUG_MODE = True  # Show all distances even when not detecting
    
    # Cash Color Detection (Korean Won Bills)
    DETECT_CASH_COLOR = True  # Enable color-based cash detection
    CASH_COLOR_THRESHOLD = 200  # Minimum pixels of cash color to confirm money
    # Color ranges for Korean currency (HSV format)
    CASH_COLORS = {
        '50000_won': {'lower': [15, 100, 100], 'upper': [30, 255, 255], 'name': '50,000원 (Yellow)'},
        '10000_won': {'lower': [40, 50, 50], 'upper': [80, 255, 255], 'name': '10,000원 (Green)'},
        '5000_won': {'lower': [0, 100, 100], 'upper': [15, 255, 255], 'name': '5,000원 (Red/Pink)'},
        '1000_won': {'lower': [100, 50, 50], 'upper': [130, 255, 255], 'name': '1,000원 (Blue)'}
    }
    
    # Camera calibration settings
    CAMERA_NAME = "default"
    CALIBRATION_SCALE = 1.0  # Pixel to real-world distance scale
    CAMERA_ANGLE = 0  # Camera angle in degrees (for future use)
    
    # Cashier Zone (Region of Interest) - Define area where cashier is located
    # Format: [x, y, width, height] or None for auto-detection
    CASHIER_ZONE = None  # Example: [100, 400, 500, 300] = rectangle zone
    
    # Cashier persistence - how long to keep cashier status after leaving zone
    CASHIER_PERSISTENCE_FRAMES = 20  # Frames (e.g., 20 frames = ~0.7 sec at 30fps)
    
    # Minimum overlap ratio - what % of person must be in zone to be cashier
    MIN_CASHIER_OVERLAP = 0.3  # 30% of person's body must be in zone (0.0 to 1.0)
    
    @classmethod
    def from_json(cls, json_path):
        """Load configuration from JSON file"""
        config = cls()
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Update configuration with JSON values
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            print(f"✅ Loaded configuration from: {json_path}")
            print(f"   - Camera: {config.CAMERA_NAME}")
            print(f"   - Hand Touch Distance: {config.HAND_TOUCH_DISTANCE}px")
            print(f"   - Calibration Scale: {config.CALIBRATION_SCALE}")
            if config.CASHIER_ZONE:
                print(f"   - Cashier Zone: {config.CASHIER_ZONE} (defined)")
            else:
                print(f"   - Cashier Zone: Auto-detect (bottom of frame)")
            
        except FileNotFoundError:
            print(f"⚠️  Config not found: {json_path}, using defaults")
        except json.JSONDecodeError as e:
            print(f"⚠️  Invalid JSON in {json_path}: {e}, using defaults")
        
        return config


class SimpleHandTouchDetector:
    """
    SIMPLE DETECTOR: When P1 hands touch P2 hands = Transaction
    No money detection, just hand proximity between people
    """
    
    def __init__(self, config=None):
        self.config = config or SimpleHandTouchConfig()
        
        print("=" * 70)
        print("🤝 CASHIER HAND TOUCH DETECTOR")
        print("=" * 70)
        print("Rule: Cashier hands touch Customer = Money Transaction")
        print(f"Touch distance threshold: {self.config.HAND_TOUCH_DISTANCE} pixels")
        if self.config.CASHIER_ZONE:
            print("Cashier Detection: ANYONE in CASHIER ZONE (multiple cashiers OK)")
        else:
            print("Cashier Detection: Person at BOTTOM of frame")
        print("Tracks transactions: EVERY cashier with EVERY customer")
        print("=" * 70)
        print()
        
        # Load model
        print(f"Loading pose model: {self.config.POSE_MODEL}...")
        self.pose_model = YOLO(self.config.POSE_MODEL)
        print("✅ Model loaded")
        print()
        
        # Stats
        self.stats = {
            'frames': 0,
            'transactions': 0,
            'confirmed_transactions': 0,
            'cash_detections': 0,
            'cash_types': {}  # Count each type of bill detected
        }
        
        # Temporal tracking - track transactions over time
        self.transaction_history = {}  # {person_pair: [frame_count]}
        
        # PERMANENT PERSON TRACKING - Map person positions to stable IDs
        self.person_id_map = {}  # {person_idx: {'center': (x,y), 'stable_id': int, 'frames_tracked': int}}
        self.next_stable_id = 1  # Counter for assigning new stable IDs
        
        # Cashier persistence - keep cashier status for N frames after leaving zone
        self.cashier_persistence = {}  # {stable_id: frames_remaining}
        self.persistence_frames = self.config.CASHIER_PERSISTENCE_FRAMES
        
        # Hand velocity tracking for violence detection
        self.hand_history = {}  # {person_id: {'left': [(x,y, frame)], 'right': [(x,y, frame)]}}
        self.max_history_frames = 5  # Track last 5 frames for velocity calculation
    
    def detect_hand_touches(self, frame):
        """
        STEP 1: Detect people and their hands
        STEP 2: Identify ALL cashiers (anyone in CASHIER_ZONE) and customers
        STEP 3: Check EVERY cashier with EVERY customer for hand touches
        STEP 4: Draw transactions
        """
        self.stats['frames'] += 1
        
        # STEP 1: Detect people and their poses
        people = []
        try:
            pose_results = self.pose_model(frame, conf=self.config.POSE_CONFIDENCE, verbose=False)
            
            if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                for person_idx, kpts in enumerate(pose_results[0].keypoints):
                    person = self._get_person_hands(kpts, person_idx)
                    if person:
                        people.append(person)
        except Exception as e:
            # If pose detection fails, still return processed frame
            output_frame = frame.copy()
            cv2.putText(output_frame, f"Pose detection error (continuing...)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(output_frame, f"Processing: Frame {self.stats['frames']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return output_frame, []
        
        # STEP 1.5: Assign stable IDs to people (tracking across frames)
        if people:
            people = self._assign_stable_ids(people)
        
        # STEP 2: Identify ALL cashiers (anyone in zone) and customers
        cashiers = []
        customers = []
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        if len(people) == 0:
            # Draw status message - waiting for people
            output_frame = frame.copy()
            cv2.putText(output_frame, "Waiting for people to detect...", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(output_frame, f"Processing: Frame {self.stats['frames']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Still draw cashier zone if defined
            if self.config.CASHIER_ZONE:
                zone_x, zone_y, zone_w, zone_h = self.config.CASHIER_ZONE
                cv2.rectangle(output_frame, (zone_x, zone_y), (zone_x + zone_w, zone_y + zone_h), 
                             (0, 255, 255), 3)
                cv2.putText(output_frame, "CASHIER ZONE", (zone_x + 10, zone_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return output_frame, []
        
        # Identify cashiers based on zone or position
        if self.config.CASHIER_ZONE:
            # Use defined zone: [x, y, width, height]
            zone_x, zone_y, zone_w, zone_h = self.config.CASHIER_ZONE
            
            # Track which stable_ids are currently in zone
            currently_in_zone = set()
            
            # Find ALL people whose bounding box overlaps cashier zone
            for idx, person in enumerate(people):
                # Check if person's bounding box overlaps with cashier zone (at least MIN_CASHIER_OVERLAP %)
                in_zone = self._bbox_overlaps_zone(person['bbox'], self.config.CASHIER_ZONE, self.config.MIN_CASHIER_OVERLAP)
                
                if in_zone:
                    # Person is in zone - definitely a cashier
                    person['id'] = idx + 1
                    person['role'] = 'cashier'
                    cashiers.append((idx, person))
                    currently_in_zone.add(person['stable_id'])
                    # Reset persistence for this person
                    self.cashier_persistence[person['stable_id']] = self.persistence_frames
                else:
                    # Not in zone - check if they have persistence from before
                    stable_id = person['stable_id']
                    if stable_id in self.cashier_persistence and self.cashier_persistence[stable_id] > 0:
                        # Still a cashier due to persistence
                        person['id'] = idx + 1
                        person['role'] = 'cashier'
                        cashiers.append((idx, person))
                        # Decrease persistence counter
                        self.cashier_persistence[stable_id] -= 1
                    else:
                        # Regular customer
                        person['id'] = idx + 1
                        person['role'] = 'customer'
                        customers.append((idx, person))
            
            # Clean up persistence for people who completely left
            to_remove = []
            for stable_id in self.cashier_persistence:
                if self.cashier_persistence[stable_id] <= 0:
                    to_remove.append(stable_id)
            for stable_id in to_remove:
                del self.cashier_persistence[stable_id]
        else:
            # Auto-detect: Person at bottom = cashier
            max_y_value = -1
            cashier_idx = None
            for idx, person in enumerate(people):
                x, y = person['center']
                if y > max_y_value:
                    max_y_value = y
                    cashier_idx = idx
            
            if cashier_idx is not None:
                people[cashier_idx]['id'] = 1
                people[cashier_idx]['role'] = 'cashier'
                cashiers.append((cashier_idx, people[cashier_idx]))
                
                for idx, person in enumerate(people):
                    if idx != cashier_idx:
                        person['id'] = idx + 1
                        person['role'] = 'customer'
                        customers.append((idx, person))
        
        # If no cashiers found, show message
        if len(cashiers) == 0:
            output_frame = frame.copy()
            if self.config.CASHIER_ZONE:
                cv2.putText(output_frame, "Waiting for cashier in zone...", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                zone_x, zone_y, zone_w, zone_h = self.config.CASHIER_ZONE
                cv2.rectangle(output_frame, (zone_x, zone_y), (zone_x + zone_w, zone_y + zone_h), 
                             (0, 255, 255), 3)
                cv2.putText(output_frame, "CASHIER ZONE", (zone_x + 10, zone_y + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(output_frame, "No cashier detected", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(output_frame, f"Processing: Frame {self.stats['frames']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            return output_frame, []
        
        # Need at least 1 cashier and 1 customer for transactions
        if len(customers) == 0:
            # Still draw cashier labels even if alone
            output_frame = self._draw_transactions(frame.copy(), people, [])
            return output_frame, []
        
        # STEP 3: Check hand touches between EVERY cashier and EVERY customer
        transactions = []
        
        # Check ALL cashier-customer pairs
        for cashier_idx, cashier in cashiers:
            for cust_idx, customer in customers:
                # Find the CLOSEST hand pair between this cashier and this customer
                closest_distance = float('inf')
                closest_transaction = None
                
                hand_pairs = [
                    (cashier.get('right_hand'), customer.get('right_hand'), 'R-R'),
                    (cashier.get('right_hand'), customer.get('left_hand'), 'R-L'),
                    (cashier.get('left_hand'), customer.get('right_hand'), 'L-R'),
                    (cashier.get('left_hand'), customer.get('left_hand'), 'L-L')
                ]
                
                for c_hand, n_hand, hand_type in hand_pairs:
                    if c_hand and n_hand:
                        dist = self._distance(c_hand, n_hand)
                        if dist <= self.config.HAND_TOUCH_DISTANCE and dist < closest_distance:
                            # Calculate hand velocities for violence detection
                            c_hand_type = hand_type.split('-')[0]  # 'L' or 'R'
                            n_hand_type = hand_type.split('-')[1]  # 'L' or 'R'
                            c_velocity, c_angle = self._calculate_hand_velocity(cashier['id'], c_hand, c_hand_type.lower(), self.stats['frames'])
                            n_velocity, n_angle = self._calculate_hand_velocity(customer['id'], n_hand, n_hand_type.lower(), self.stats['frames'])
                            
                            # Check for violence: Fast hand movement (>150 px/frame)
                            is_violent = False
                            violence_reason = None
                            if self.config.DETECT_HAND_VELOCITY:
                                violence_threshold = self.config.VIOLENCE_VELOCITY_THRESHOLD
                                if c_velocity > violence_threshold:
                                    is_violent = True
                                    violence_reason = f"Cashier hand moving fast: {c_velocity:.0f} px/frame"
                                elif n_velocity > violence_threshold:
                                    is_violent = True
                                    violence_reason = f"Customer hand moving fast: {n_velocity:.0f} px/frame"
                            
                            # If violent movement detected, mark as violence instead of cash
                            if is_violent:
                                print(f"  ⚠️  🚨 VIOLENCE DETECTED: {violence_reason}")
                                print(f"     Between: P{cashier['id']} ↔ P{customer['id']} ({hand_type})")
                                closest_distance = dist
                                closest_transaction = {
                                    'p1_id': cashier['id'],
                                    'p2_id': customer['id'],
                                    'p1_hand': c_hand,
                                    'p2_hand': n_hand,
                                    'hand_type': hand_type,
                                    'distance': dist,
                                    'cashier_customer_pair': f"C{cashier['id']}-P{customer['id']}",
                                    'cash_detected': True,  # Mark as detected
                                    'cash_type': '🚨 Violence (Fast Movement)',  # Violence label
                                    'cash_bbox': None,
                                    'cash_pixels': 0,
                                    'analysis_scores': {'velocity': max(c_velocity, n_velocity), 'reason': violence_reason},
                                    'is_violence': True  # NEW: Flag for violence
                                }
                                continue  # Skip material analysis
                            
                            # Check for cash/card with 3-Phase Handover Zone Analysis
                            material_detected, material_type, material_bbox, analysis_scores = self._analyze_handover_zone(frame, c_hand, n_hand, draw_debug=True)
                            
                            # Apply confidence threshold (default 0.65)
                            confidence_threshold = self.config.CASH_DETECTION_CONFIDENCE
                            if material_detected:
                                # Calculate overall confidence from analysis scores
                                geom = analysis_scores.get('geometric', 0)
                                glare = analysis_scores.get('photometric', 0)
                                color = analysis_scores.get('chromatic', 0)
                                
                                # Confidence is high if scores support the material type
                                if '💳' in material_type or 'Card' in material_type:
                                    # Card: high geometric + high glare + low color
                                    confidence = (geom + glare + (1 - color)) / 3
                                else:
                                    # Cash: low geometric + low glare + high color
                                    confidence = ((1 - geom) + (1 - glare) + color) / 3
                                
                                # Apply threshold
                                if confidence < confidence_threshold:
                                    material_detected = False
                                    if self.config.DEBUG_MODE:
                                        print(f"  ⚠️  Material rejected (confidence {confidence:.2f} < {confidence_threshold:.2f}) - P{cashier['id']}↔P{customer['id']}")
                            
                            # Debug: Print when hands are close but NO material detected
                            if self.config.DETECT_CASH_COLOR and not material_detected and self.config.DEBUG_MODE:
                                if dist < closest_distance:  # Only print for closest pair
                                    geom = analysis_scores.get('geometric', 0)
                                    glare = analysis_scores.get('photometric', 0)
                                    color = analysis_scores.get('chromatic', 0)
                                    print(f"  ⚠️  Hands close ({dist:.0f}px) but NO MATERIAL - P{cashier['id']}↔P{customer['id']} ({hand_type}) [G:{geom:.2f} L:{glare:.2f} C:{color:.2f}]")
                            
                            # Only confirm if BOTH hand proximity AND material detected
                            if self.config.DETECT_CASH_COLOR and not material_detected:
                                continue  # Skip this if no cash/card found
                            
                            closest_distance = dist
                            closest_transaction = {
                                'p1_id': cashier['id'],
                                'p2_id': customer['id'],
                                'p1_hand': c_hand,
                                'p2_hand': n_hand,
                                'hand_type': hand_type,
                                'distance': dist,
                                'cashier_customer_pair': f"C{cashier['id']}-P{customer['id']}",
                                'cash_detected': material_detected,
                                'cash_type': material_type,  # Now includes "💵 10,000원" or "💳 Card"
                                'cash_bbox': material_bbox,
                                'cash_pixels': 0,  # Deprecated, kept for compatibility
                                'analysis_scores': analysis_scores,  # NEW: Full analysis data
                                'velocities': {'cashier': c_velocity, 'customer': n_velocity},  # NEW: Velocity data
                                'is_violence': False  # Normal transaction
                            }
                
                # Add transaction if found for this pair
                if closest_transaction:
                    transactions.append(closest_transaction)
        
        # Track transactions over time (temporal filtering)
        confirmed_transactions = []
        current_pairs = set()
        
        for trans in transactions:
            pair_key = f"{trans['p1_id']}-{trans['p2_id']}"
            current_pairs.add(pair_key)
            
            # Initialize or update transaction history
            if pair_key not in self.transaction_history:
                self.transaction_history[pair_key] = 0
            
            self.transaction_history[pair_key] += 1
            
            # Confirm transaction if it lasts MIN_TRANSACTION_FRAMES or more
            if self.transaction_history[pair_key] >= self.config.MIN_TRANSACTION_FRAMES:
                trans['confirmed'] = True
                trans['duration'] = self.transaction_history[pair_key]
                confirmed_transactions.append(trans)
                
                # Count as confirmed transaction (only once when first confirmed)
                if self.transaction_history[pair_key] == self.config.MIN_TRANSACTION_FRAMES:
                    self.stats['confirmed_transactions'] += 1
                    
                    # Debug: Print material detection details (Cash or Card)
                    if self.config.DETECT_CASH_COLOR and trans.get('cash_detected'):
                        self.stats['cash_detections'] += 1
                        
                        # Track material type (Cash or Card)
                        material_type = trans.get('cash_type', 'Unknown')
                        if material_type not in self.stats['cash_types']:
                            self.stats['cash_types'][material_type] = 0
                        self.stats['cash_types'][material_type] += 1
                        
                        # Get analysis scores
                        scores = trans.get('analysis_scores', {})
                        
                        print(f"\n  🎯 MATERIAL DETECTED!")
                        print(f"     Type: {material_type}")
                        print(f"     📊 Analysis Scores:")
                        print(f"        - Geometric (shape): {scores.get('geometric', 0):.2f}")
                        print(f"        - Photometric (glare): {scores.get('photometric', 0):.2f}")
                        print(f"        - Chromatic (color): {scores.get('chromatic', 0):.2f}")
                        if trans.get('cash_bbox'):
                            cx1, cy1, cx2, cy2 = trans['cash_bbox']
                            print(f"     📍 Location: ({cx1}, {cy1}) to ({cx2}, {cy2})")
                            print(f"     📏 Size: {cx2-cx1}x{cy2-cy1} pixels")
                        print(f"     👥 Between: P{trans['p1_id']} ↔ P{trans['p2_id']} ({trans['hand_type']})")
                        print()
            else:
                trans['confirmed'] = False
                trans['duration'] = self.transaction_history[pair_key]
        
        # Decay history for pairs not detected in this frame
        pairs_to_remove = []
        for pair_key in self.transaction_history:
            if pair_key not in current_pairs:
                self.transaction_history[pair_key] = max(0, self.transaction_history[pair_key] - 2)
                if self.transaction_history[pair_key] == 0:
                    pairs_to_remove.append(pair_key)
        
        for pair_key in pairs_to_remove:
            del self.transaction_history[pair_key]
        
        # Count all detections
        if transactions:
            self.stats['transactions'] += len(transactions)
        
        # STEP 4: Draw hands and confirmed transactions
        output_frame = self._draw_transactions(frame.copy(), people, confirmed_transactions)
        
        return output_frame, confirmed_transactions
    
    def _get_person_hands(self, keypoints, person_idx):
        """Get hand positions, person center, and bounding box from keypoints"""
        kpts = keypoints.xy[0].cpu().numpy()
        conf = keypoints.conf[0].cpu().numpy()
        
        # Keypoint indices:
        # 0 = nose (for center calculation)
        # 9 = left wrist
        # 10 = right wrist
        
        right_hand = None
        left_hand = None
        center = None
        
        if conf[10] > 0.3:  # Right wrist detected
            right_hand = (int(kpts[10][0]), int(kpts[10][1]))
        
        if conf[9] > 0.3:  # Left wrist detected
            left_hand = (int(kpts[9][0]), int(kpts[9][1]))
        
        # Get person center (use nose if available, otherwise average of hands)
        if conf[0] > 0.3:  # Nose detected
            center = (int(kpts[0][0]), int(kpts[0][1]))
        elif right_hand and left_hand:
            center = ((right_hand[0] + left_hand[0]) // 2, (right_hand[1] + left_hand[1]) // 2)
        elif right_hand:
            center = right_hand
        elif left_hand:
            center = left_hand
        
        # Need at least one hand and center
        if (not right_hand and not left_hand) or not center:
            return None
        
        # Calculate bounding box from all visible keypoints
        visible_points = []
        for i in range(len(kpts)):
            if conf[i] > 0.3:
                visible_points.append(kpts[i])
        
        if visible_points:
            visible_points = np.array(visible_points)
            x_min = int(np.min(visible_points[:, 0]))
            y_min = int(np.min(visible_points[:, 1]))
            x_max = int(np.max(visible_points[:, 0]))
            y_max = int(np.max(visible_points[:, 1]))
            bbox = (x_min, y_min, x_max, y_max)
        else:
            # Fallback: create bbox around center
            bbox = (center[0]-50, center[1]-50, center[0]+50, center[1]+50)
        
        return {
            'id': person_idx + 1,
            'right_hand': right_hand,
            'left_hand': left_hand,
            'center': center,
            'bbox': bbox  # (x_min, y_min, x_max, y_max)
        }
    
    def _distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _analyze_handover_zone(self, frame, hand1, hand2, draw_debug=False):
        """
        ═══════════════════════════════════════════════════════════════
        🎯 3-PHASE HANDOVER ZONE ANALYSIS SYSTEM
        ═══════════════════════════════════════════════════════════════
        
        Phase 1: Zone Creation (WHERE?)
        - Create precise handover zone between two hands
        - Exclude wrists, sleeves, only fingertips + object
        
        Phase 2: Material Analysis (WHAT?)
        - Filter A: Geometric Logic (Shape)
        - Filter B: Photometric Logic (Glare)
        - Filter C: Chromatic Logic (Color)
        
        Returns: (detected, material_type, bbox, confidence_scores)
        ═══════════════════════════════════════════════════════════════
        """
        if not self.config.DETECT_CASH_COLOR:
            return False, None, None, {}
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: ZONE CREATION (Create Handover Zone)
        # ═══════════════════════════════════════════════════════════════
        
        x1, y1 = int(hand1[0]), int(hand1[1])
        x2, y2 = int(hand2[0]), int(hand2[1])
        
        # Calculate exact midpoint between hands
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        
        # Calculate hand distance
        hand_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Create TINY zone (40% of hand distance) - ONLY fingertips + object
        # This excludes wrists and sleeves mathematically
        zone_size = max(int(hand_distance * 0.4), 30)  # Min 30px, 40% of distance
        
        roi_x1 = max(0, mid_x - zone_size)
        roi_y1 = max(0, mid_y - zone_size)
        roi_x2 = min(frame.shape[1], mid_x + zone_size)
        roi_y2 = min(frame.shape[0], mid_y + zone_size)
        
        # Extract handover zone
        handover_zone = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if handover_zone.size == 0 or handover_zone.shape[0] < 10 or handover_zone.shape[1] < 10:
            return False, None, None, {}
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: MATERIAL ANALYSIS (Detect Cash vs Card vs Nothing)
        # ═══════════════════════════════════════════════════════════════
        
        # Convert zone to different color spaces
        hsv_zone = cv2.cvtColor(handover_zone, cv2.COLOR_BGR2HSV)
        gray_zone = cv2.cvtColor(handover_zone, cv2.COLOR_BGR2GRAY)
        
        # --- FILTER A: GEOMETRIC LOGIC (Shape Analysis) ---
        geometric_score = self._filter_geometric_shape(handover_zone, gray_zone)
        
        # --- FILTER B: PHOTOMETRIC LOGIC (Glare Detection) ---
        photometric_score = self._filter_photometric_glare(gray_zone)
        
        # --- FILTER C: CHROMATIC LOGIC (Color Saturation) ---
        chromatic_score, detected_bill = self._filter_chromatic_color(hsv_zone)
        
        # ═══════════════════════════════════════════════════════════════
        # DECISION LOGIC: Combine all filters
        # ═══════════════════════════════════════════════════════════════
        
        scores = {
            'geometric': geometric_score,      # > 0.7 = Card, < 0.3 = Cash
            'photometric': photometric_score,  # > 0.5 = Card (glare), < 0.5 = Cash (matte)
            'chromatic': chromatic_score,      # > 0.6 = Cash (colorful), < 0.4 = Card (gray)
            'bill_type': detected_bill
        }
        
        # Decision tree (prioritize strong signals)
        material_type = None
        confidence = 0
        
        # Strong Card signal: High glare + Low color + Rectangular shape
        if photometric_score > 0.5 and chromatic_score < 0.4 and geometric_score > 0.6:
            material_type = "💳 Card"
            confidence = (photometric_score + geometric_score + (1 - chromatic_score)) / 3
        
        # Strong Cash signal: High color + Low glare + Detected bill type
        elif chromatic_score > 0.6 and photometric_score < 0.5 and detected_bill:
            material_type = f"💵 {detected_bill}"
            confidence = chromatic_score
        
        # Weak Cash signal: Just high color
        elif chromatic_score > 0.5 and detected_bill:
            material_type = f"💵 {detected_bill}"
            confidence = chromatic_score * 0.8  # Lower confidence
        
        # Nothing detected or ambiguous
        else:
            material_type = None
            confidence = 0
        
        # Add confidence to scores dictionary
        scores['confidence'] = confidence
        
        # Debug visualization
        if draw_debug and self.config.DEBUG_MODE:
            self._draw_handover_debug(frame, roi_x1, roi_y1, roi_x2, roi_y2, 
                                     mid_x, mid_y, scores, material_type, confidence)
        
        # Only return detection if confidence is sufficient (internal threshold)
        if material_type and confidence > 0.5:
            bbox = (roi_x1, roi_y1, roi_x2, roi_y2)
            return True, material_type, bbox, scores
        
        return False, None, None, scores
    
    def _filter_geometric_shape(self, zone_bgr, zone_gray):
        """
        FILTER A: Geometric Logic (Shape Analysis)
        
        Logic:
        - Card: Rigid rectangle (fixed aspect ratio ~1.6:1)
        - Cash: Flexible, often folded/bent (irregular shape)
        
        Returns: geometric_score (0.0-1.0)
        - High score (>0.7) = Card-like (rectangular)
        - Low score (<0.3) = Cash-like (irregular)
        """
        # Find edges in zone
        edges = cv2.Canny(zone_gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get largest contour (assumed to be the object)
        largest = max(contours, key=cv2.contourArea)
        
        # Check if contour is too small (noise)
        if cv2.contourArea(largest) < 100:
            return 0.0
        
        # Fit minimum area rectangle
        rect = cv2.minAreaRect(largest)
        width, height = rect[1]
        
        if width == 0 or height == 0:
            return 0.0
        
        # Calculate aspect ratio
        aspect_ratio = max(width, height) / min(width, height)
        
        # Standard credit card aspect ratio is 1.586:1 (85.6mm × 53.98mm)
        card_aspect = 1.586
        aspect_diff = abs(aspect_ratio - card_aspect)
        
        # Card-like score: closer to card aspect = higher score
        # 0.0 diff = 1.0 score, 1.0 diff = 0.0 score
        geometric_score = max(0, 1.0 - aspect_diff)
        
        return geometric_score
    
    def _filter_photometric_glare(self, zone_gray):
        """
        FILTER B: Photometric Logic (Glare Detection)
        
        Logic:
        - Card: Plastic reflects light → bright glare spots
        - Cash: Paper absorbs light → matte/dull appearance
        
        Returns: photometric_score (0.0-1.0)
        - High score (>0.5) = Card-like (has glare)
        - Low score (<0.5) = Cash-like (matte)
        """
        # Find very bright pixels (potential glare)
        # Threshold at 240 (nearly white)
        _, bright_mask = cv2.threshold(zone_gray, 240, 255, cv2.THRESH_BINARY)
        
        # Count bright pixels
        total_pixels = zone_gray.shape[0] * zone_gray.shape[1]
        bright_pixels = np.count_nonzero(bright_mask)
        bright_ratio = bright_pixels / total_pixels
        
        # Card typically has 5-20% glare pixels
        # Normalize: 0% = 0.0, 10% = 1.0, >20% = 1.0
        photometric_score = min(1.0, bright_ratio / 0.10)
        
        return photometric_score
    
    def _filter_chromatic_color(self, zone_hsv):
        """
        FILTER C: Chromatic Logic (Color Saturation)
        
        Logic:
        - Card: White/Gray/Black (Low saturation)
        - Cash (Korean Won): Green/Yellow/Pink (High saturation)
        
        Returns: (chromatic_score, detected_bill_type)
        - High score (>0.6) = Cash-like (colorful)
        - Low score (<0.4) = Card-like (grayscale)
        """
        # Extract saturation channel (S in HSV)
        saturation = zone_hsv[:, :, 1]
        
        # Calculate average saturation
        avg_saturation = np.mean(saturation)
        
        # Normalize: 0-255 → 0.0-1.0
        chromatic_score = avg_saturation / 255.0
        
        # Also check for specific Korean Won bill colors
        detected_bill = None
        max_pixels = 0
        
        for bill_type, color_info in self.config.CASH_COLORS.items():
            lower = np.array(color_info['lower'])
            upper = np.array(color_info['upper'])
            
            # Create mask for this bill color
            mask = cv2.inRange(zone_hsv, lower, upper)
            pixel_count = np.count_nonzero(mask)
            
            # Check if this bill color is prominent
            if pixel_count > max_pixels and pixel_count > 50:  # At least 50 pixels
                max_pixels = pixel_count
                detected_bill = color_info['name']
        
        # Boost chromatic score if specific bill detected
        if detected_bill and max_pixels > 100:
            chromatic_score = max(chromatic_score, 0.7)
        
        return chromatic_score, detected_bill
    
    def _draw_handover_debug(self, frame, x1, y1, x2, y2, mid_x, mid_y, 
                            scores, material_type, confidence):
        """Draw debug visualization for handover zone analysis"""
        # Draw handover zone (cyan dashed rectangle)
        dash_length = 8
        color = (255, 255, 0)  # Cyan for zone
        
        for i in range(x1, x2, dash_length * 2):
            cv2.line(frame, (i, y1), (min(i + dash_length, x2), y1), color, 2)
            cv2.line(frame, (i, y2), (min(i + dash_length, x2), y2), color, 2)
        for i in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, i), (x1, min(i + dash_length, y2)), color, 2)
            cv2.line(frame, (x2, i), (x2, min(i + dash_length, y2)), color, 2)
        
        # Draw midpoint
        cv2.circle(frame, (mid_x, mid_y), 6, (255, 255, 0), -1)
        cv2.circle(frame, (mid_x, mid_y), 6, (0, 0, 0), 2)
        
        # Draw label
        label = "HANDOVER ZONE"
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw analysis results (below zone)
        y_offset = y2 + 20
        cv2.putText(frame, f"Geometric: {scores['geometric']:.2f}", 
                   (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"Glare: {scores['photometric']:.2f}", 
                   (x1, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, f"Color: {scores['chromatic']:.2f}", 
                   (x1, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Draw final detection
        if material_type:
            cv2.putText(frame, f"{material_type} ({confidence:.0%})", 
                       (x1, y_offset + 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)
    
    # Alias for backward compatibility
    def _detect_cash_color(self, frame, hand1, hand2, draw_debug=False):
        """Legacy method - redirects to new handover zone analysis"""
        return self._analyze_handover_zone(frame, hand1, hand2, draw_debug)
    
    def _calculate_hand_velocity(self, person_id, hand_pos, hand_type, current_frame):
        """
        Calculate hand velocity (pixels/frame) for violence detection
        
        Fast hand movements toward cashier = potential violence
        Returns: velocity (pixels/frame), direction
        """
        if not self.config.DETECT_HAND_VELOCITY:
            return 0, None
        
        # Initialize history for this person if needed
        if person_id not in self.hand_history:
            self.hand_history[person_id] = {'left': [], 'right': []}
        
        # Add current hand position to history
        self.hand_history[person_id][hand_type].append((hand_pos[0], hand_pos[1], current_frame))
        
        # Keep only recent history (last N frames)
        if len(self.hand_history[person_id][hand_type]) > self.max_history_frames:
            self.hand_history[person_id][hand_type] = self.hand_history[person_id][hand_type][-self.max_history_frames:]
        
        # Need at least 2 positions to calculate velocity
        history = self.hand_history[person_id][hand_type]
        if len(history) < 2:
            return 0, None
        
        # Calculate velocity between first and last position
        (x1, y1, frame1) = history[0]
        (x2, y2, frame2) = history[-1]
        
        frame_diff = frame2 - frame1
        if frame_diff == 0:
            return 0, None
        
        # Calculate distance traveled
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Velocity = distance / time
        velocity = distance / frame_diff
        
        # Calculate direction (angle in degrees, 0 = right, 90 = down, 180 = left, 270 = up)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        return velocity, angle
    
    def _bbox_overlaps_zone(self, bbox, zone, min_overlap_ratio=0.3):
        """
        Check if bounding box significantly overlaps with cashier zone
        bbox: (x_min, y_min, x_max, y_max)
        zone: [x, y, width, height]
        min_overlap_ratio: Minimum percentage of bbox that must be in zone (0.0 to 1.0)
        """
        x_min, y_min, x_max, y_max = bbox
        zone_x, zone_y, zone_w, zone_h = zone
        zone_x_max = zone_x + zone_w
        zone_y_max = zone_y + zone_h
        
        # Calculate intersection area
        x_inter_min = max(x_min, zone_x)
        y_inter_min = max(y_min, zone_y)
        x_inter_max = min(x_max, zone_x_max)
        y_inter_max = min(y_max, zone_y_max)
        
        # Check if there's NO overlap
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return False
        
        # Calculate overlap area
        overlap_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        
        # Calculate person's bounding box area
        bbox_area = (x_max - x_min) * (y_max - y_min)
        
        if bbox_area == 0:
            return False
        
        # Calculate overlap ratio (what % of person is in zone)
        overlap_ratio = overlap_area / bbox_area
        
        # Person is cashier if at least min_overlap_ratio of their body is in zone
        return overlap_ratio >= min_overlap_ratio
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        bbox format: (x_min, y_min, x_max, y_max)
        Returns: IoU score (0.0 to 1.0)
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection area
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)
        
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0  # No intersection
        
        intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
        
        # Calculate union area
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _assign_stable_ids(self, people):
        """
        Assign stable IDs to people across frames using IoU + distance tracking.
        This is MORE ROBUST than position-only tracking.
        Uses bounding box overlap (IoU) as primary metric.
        """
        # Build mapping of all tracked stable_ids to their last known data
        stable_id_to_data = {}
        for data in self.person_id_map.values():
            stable_id = data['stable_id']
            stable_id_to_data[stable_id] = data
        
        # Match detected people with existing stable IDs using IoU
        matched = {}
        used_stable_ids = set()
        
        for person_idx, person in enumerate(people):
            best_match_id = None
            best_match_score = 0.0  # Use IoU as primary score
            
            # Try to match with existing tracked people
            for stable_id, tracked_data in stable_id_to_data.items():
                if stable_id in used_stable_ids:
                    continue
                
                # Calculate IoU between bounding boxes (primary metric)
                iou = self._calculate_iou(person['bbox'], tracked_data['bbox'])
                
                # Calculate distance between centers (secondary metric)
                dist = self._distance(person['center'], tracked_data['center'])
                
                # Combined score: IoU is primary (0-1), distance is secondary
                # If IoU > 0.3, it's a good match regardless of distance
                # Otherwise, use distance threshold (400px)
                if iou > 0.3:
                    score = iou  # Strong match via bounding box overlap
                elif dist < 400:
                    score = 0.2 * (1.0 - dist/400)  # Weak match via proximity
                else:
                    score = 0.0  # No match
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_id = stable_id
            
            if best_match_id is not None and best_match_score > 0.15:
                # Matched with existing person - KEEP same stable_id
                matched[person_idx] = best_match_id
                used_stable_ids.add(best_match_id)
            else:
                # New person - assign new stable ID
                new_stable_id = self.next_stable_id
                self.next_stable_id += 1
                
                matched[person_idx] = new_stable_id
                used_stable_ids.add(new_stable_id)
            
            # Update person with stable ID
            person['stable_id'] = matched[person_idx]
        
        # Update tracking map (stable_id -> last bbox + position)
        new_map = {}
        for person_idx, person in enumerate(people):
            new_map[matched[person_idx]] = {
                'center': person['center'],
                'bbox': person['bbox'],
                'stable_id': matched[person_idx],
                'frames_tracked': stable_id_to_data.get(matched[person_idx], {}).get('frames_tracked', 0) + 1
            }
        
        self.person_id_map = new_map
        
        return people
    
    def _draw_transactions(self, frame, people, transactions):
        """Draw hands and transactions"""
        
        # Draw cashier zone if defined
        if self.config.CASHIER_ZONE:
            zone_x, zone_y, zone_w, zone_h = self.config.CASHIER_ZONE
            # Draw semi-transparent zone
            overlay = frame.copy()
            cv2.rectangle(overlay, (zone_x, zone_y), (zone_x + zone_w, zone_y + zone_h), 
                         (0, 255, 255), 3)  # Yellow rectangle
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw label
            cv2.putText(frame, "CASHIER ZONE", (zone_x + 10, zone_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw all hands with labels
        if self.config.DRAW_HANDS:
            for person in people:
                # Cashier (P1) hands are GOLD, others are BLUE/RED
                is_cashier = person.get('role') == 'cashier'
                right_color = (0, 215, 255) if is_cashier else (255, 0, 0)  # Gold or Blue
                left_color = (0, 215, 255) if is_cashier else (0, 0, 255)   # Gold or Red
                
                # Draw right hand
                if person['right_hand']:
                    cv2.circle(frame, person['right_hand'], 8, right_color, -1)
                    label = f"P{person['id']}-R" if not is_cashier else f"P{person['id']}-R (CASHIER)"
                    cv2.putText(frame, label, 
                               (person['right_hand'][0] + 10, person['right_hand'][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 2)
                
                # Draw left hand
                if person['left_hand']:
                    cv2.circle(frame, person['left_hand'], 8, left_color, -1)
                    label = f"P{person['id']}-L" if not is_cashier else f"P{person['id']}-L (CASHIER)"
                    cv2.putText(frame, label, 
                               (person['left_hand'][0] + 10, person['left_hand'][1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 2)
                
                # Draw smaller CASHIER label box at person's center
                if is_cashier:
                    cx, cy = person['center']
                    # Draw background box (smaller)
                    box_width, box_height = 90, 25
                    cv2.rectangle(frame, 
                                (cx - box_width//2, cy - box_height//2),
                                (cx + box_width//2, cy + box_height//2),
                                (0, 215, 255), -1)  # Gold filled box
                    # Draw black border
                    cv2.rectangle(frame, 
                                (cx - box_width//2, cy - box_height//2),
                                (cx + box_width//2, cy + box_height//2),
                                (0, 0, 0), 2)  # Black border
                    # Draw text (smaller)
                    cv2.putText(frame, "CASHIER", 
                               (cx - 38, cy + 6),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # DEBUG MODE: Show ONLY cashier-to-customer distance
        if self.config.DEBUG_MODE and len(people) >= 2:
            # Find cashier
            cashier = None
            customers = []
            for person in people:
                if person.get('role') == 'cashier':
                    cashier = person
                else:
                    customers.append(person)
            
            if cashier and customers:
                y_offset = 60
                
                # Find nearest customer and show that distance only
                min_customer_dist = float('inf')
                nearest_customer = None
                
                for customer in customers:
                    dist = self._distance(cashier['center'], customer['center'])
                    if dist < min_customer_dist:
                        min_customer_dist = dist
                        nearest_customer = customer
                
                if nearest_customer:
                    # Check closest hands between cashier and nearest customer
                    min_dist = float('inf')
                    closest_pair = None
                    
                    hand_pairs = [
                        (cashier.get('right_hand'), nearest_customer.get('right_hand'), 'R-R'),
                        (cashier.get('right_hand'), nearest_customer.get('left_hand'), 'R-L'),
                        (cashier.get('left_hand'), nearest_customer.get('right_hand'), 'L-R'),
                        (cashier.get('left_hand'), nearest_customer.get('left_hand'), 'L-L')
                    ]
                    
                    for h1, h2, hand_type in hand_pairs:
                        if h1 and h2:
                            dist = self._distance(h1, h2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_pair = (h1, h2, hand_type)
                    
                    if closest_pair:
                        h1, h2, hand_type = closest_pair
                        # Color: RED if too far, YELLOW if close but not confirmed, GREEN if confirmed
                        if min_dist <= self.config.HAND_TOUCH_DISTANCE:
                            color = (0, 255, 255)  # Yellow - within threshold but not confirmed
                            status = "CLOSE"
                        else:
                            color = (0, 0, 255)  # Red - too far
                            status = "FAR"
                        
                        debug_text = f"CASHIER<->Customer ({hand_type}): {min_dist:.0f}px [{status}]"
                        cv2.putText(frame, debug_text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += 25
        
        # Draw transactions (hand touches)
        if self.config.DRAW_CONNECTIONS and transactions:
            for trans in transactions:
                # Draw thick line between touching hands (GREEN)
                cv2.line(frame, trans['p1_hand'], trans['p2_hand'], (0, 255, 0), 4)
                
                # Draw MATERIAL BOX if detected (Cash = Green, Card = Blue)
                if trans.get('cash_detected') and trans.get('cash_bbox'):
                    cx1, cy1, cx2, cy2 = trans['cash_bbox']
                    material_type = trans.get('cash_type', 'MATERIAL')
                    
                    # Choose color based on material type
                    if '💳' in material_type or 'Card' in material_type:
                        box_color = (255, 100, 0)  # Blue for card
                        bg_color = (255, 100, 0)
                    else:
                        box_color = (0, 255, 0)  # Green for cash
                        bg_color = (0, 255, 0)
                    
                    # Draw rectangle around detected material
                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), box_color, 3)
                    
                    # Draw material type label
                    material_label = material_type
                    label_size = cv2.getTextSize(material_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Background for label
                    cv2.rectangle(frame, (cx1, cy1 - 25), (cx1 + label_size[0] + 10, cy1), 
                                 bg_color, -1)
                    cv2.rectangle(frame, (cx1, cy1 - 25), (cx1 + label_size[0] + 10, cy1), 
                                 (0, 0, 0), 2)
                    
                    # Draw label text
                    cv2.putText(frame, material_label, (cx1 + 5, cy1 - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Draw analysis scores below box
                    scores = trans.get('analysis_scores', {})
                    score_text = f"G:{scores.get('geometric', 0):.1f} L:{scores.get('photometric', 0):.1f} C:{scores.get('chromatic', 0):.1f}"
                    cv2.putText(frame, score_text, (cx1 + 5, cy2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_color, 2)
                
                # Calculate midpoint for label
                mid_x = (trans['p1_hand'][0] + trans['p2_hand'][0]) // 2
                mid_y = (trans['p1_hand'][1] + trans['p2_hand'][1]) // 2
                
                # Draw transaction label above the line (shows material type)
                label_y = min(trans['p1_hand'][1], trans['p2_hand'][1]) - 15
                
                # Determine label text and color based on material
                material_type = trans.get('cash_type', 'EXCHANGE')
                if '💳' in material_type or 'Card' in material_type:
                    text = "CARD TRANSACTION"
                    label_color = (255, 100, 0)  # Blue
                else:
                    text = "CASH TRANSACTION"
                    label_color = (0, 255, 0)  # Green
                
                # Background box for label
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                box_x1 = mid_x - text_size[0]//2 - 5
                box_y1 = label_y - text_size[1] - 5
                box_x2 = mid_x + text_size[0]//2 + 5
                box_y2 = label_y + 5
                
                # Draw background box
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), label_color, -1)
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 2)
                
                # Draw text
                cv2.putText(frame, text, (mid_x - text_size[0]//2, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw counter and frame number at top
        counter_text = f"Transactions: {len(transactions)}"
        cv2.putText(frame, counter_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show material analysis status
        if self.config.DETECT_CASH_COLOR:
            status_text = "Material Analysis: ON"
            status_color = (0, 255, 255)  # Yellow
            cv2.putText(frame, status_text, (frame.shape[1] - 280, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Show detected materials in this frame (Cash or Card)
            if transactions:
                for idx, trans in enumerate(transactions):
                    if trans.get('cash_detected'):
                        material_type = trans.get('cash_type', 'MATERIAL')
                        
                        # Color code: Blue for card, Yellow for cash
                        if '💳' in material_type or 'Card' in material_type:
                            display_color = (255, 100, 0)  # Blue
                        else:
                            display_color = (0, 255, 0)  # Green
                        
                        y_pos = 60 + (idx * 25)
                        cv2.putText(frame, material_type, (frame.shape[1] - 280, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)
        
        # Always show frame number to indicate processing is active
        frame_text = f"Frame: {self.stats['frames']}"
        cv2.putText(frame, frame_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_video(self, video_path, output_path):
        """Process one video"""
        # RESET all tracking state for new video
        self.transaction_history = {}
        self.person_id_map = {}
        self.next_stable_id = 1  # Reset ID counter for new video
        self.cashier_persistence = {}  # Reset cashier persistence
        self.stats = {
            'frames': 0,
            'transactions': 0,
            'confirmed_transactions': 0,
            'cash_detections': 0,
            'cash_types': {}
        }
        
        print(f"\n{'='*70}")
        print(f"📹 Processing: {video_path}")
        print(f"{'='*70}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Resolution: {width}x{height}")
        print(f"📊 FPS: {fps}")
        print(f"📊 Total frames: {total_frames}")
        print()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_num = 0
        start_time = time.time()
        transaction_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"  ℹ️  End of video reached at frame {frame_num}/{total_frames}")
                break
            
            frame_num += 1
            
            try:
                # Detect hand touches
                output_frame, transactions = self.detect_hand_touches(frame)
                
                if transactions:
                    transaction_frames += 1
                    print(f"  💰 Frame {frame_num}: {len(transactions)} TRANSACTION(S)!")
                    for t in transactions:
                        print(f"     → P{t['p1_id']} ↔ P{t['p2_id']} ({t['hand_type']}, {t['distance']:.0f}px)")
                
                # Write frame
                out.write(output_frame)
                
            except Exception as e:
                print(f"  ⚠️  Error processing frame {frame_num}: {e}")
                # Write original frame if processing fails
                out.write(frame)
            
            # Progress indicator
            if frame_num % 100 == 0:
                print(f"  ⏳ {frame_num}/{total_frames} frames ({100*frame_num/total_frames:.1f}%)")
        
        # Cleanup
        elapsed = time.time() - start_time
        fps_actual = frame_num / elapsed if elapsed > 0 else 0
        
        cap.release()
        out.release()
        
        # Summary
        print()
        print(f"{'='*70}")
        print(f"✅ COMPLETED: {Path(video_path).name}")
        print(f"{'='*70}")
        print(f"Frames processed: {frame_num}")
        print(f"Processing speed: {fps_actual:.1f} FPS")
        print(f"Frames with transactions: {transaction_frames}")
        print(f"Total hand touches detected: {self.stats['transactions']}")
        print(f"✅ CONFIRMED TRANSACTIONS (5+ frames): {self.stats['confirmed_transactions']}")
        print(f"Average per frame: {self.stats['transactions']/frame_num:.2f}")
        
        # Print cash detection statistics
        if self.config.DETECT_CASH_COLOR:
            print(f"\n💵 CASH DETECTION SUMMARY:")
            print(f"   Total cash detections: {self.stats['cash_detections']}")
            if self.stats['cash_types']:
                print(f"   Detected bills:")
                for cash_type, count in sorted(self.stats['cash_types'].items(), key=lambda x: x[1], reverse=True):
                    print(f"      • {cash_type}: {count}x")
            else:
                print(f"   No cash detected in video")
        
        print(f"{'='*70}")
        print(f"💾 Saved: {output_path}")
        print()


def main():
    """Process all videos with camera-specific configurations"""
    
    # Look for camera folders in input directory
    input_dir = Path("input")
    camera_folders = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("camera")])
    
    if not camera_folders:
        print("=" * 70)
        print("⚠️  No camera folders found in input/")
        print("=" * 70)
        print("Expected folder structure:")
        print("  input/")
        print("    camera1/")
        print("      config.json (optional)")
        print("      video1.mp4")
        print("      video2.mp4")
        print("    camera2/")
        print("      config.json (optional)")
        print("      ...")
        print()
        print("Creating example structure...")
        
        # Fallback to old structure
        video_dir = Path("input/videos")
        if video_dir.exists():
            print(f"Found legacy videos folder: {video_dir}")
            print("Processing with default configuration...")
            detector = SimpleHandTouchDetector()
            output_dir = Path("output/videos")
            output_dir.mkdir(exist_ok=True, parents=True)
            
            videos = sorted(list(video_dir.glob("*.mp4")))
            if not videos:
                print("❌ No videos found")
                return
            
            for idx, video in enumerate(videos, 1):
                print(f"\n{'🎬 VIDEO {idx}/{len(videos)}':.^70}")
                output_path = output_dir / f"hand_touch_{video.name}"
                detector.process_video(str(video), str(output_path))
        else:
            print("❌ No videos found. Please create camera folders.")
        return
    
    print("=" * 70)
    print(f"🎥 Found {len(camera_folders)} camera folder(s)")
    print("=" * 70)
    for cam in camera_folders:
        print(f"  - {cam.name}")
    print()
    
    # Process each camera folder
    total_videos = 0
    for camera_folder in camera_folders:
        camera_name = camera_folder.name
        print(f"\n{'='*70}")
        print(f"📹 PROCESSING CAMERA: {camera_name}")
        print(f"{'='*70}")
        
        # Look for config file
        config_file = camera_folder / "config.json"
        if config_file.exists():
            config = SimpleHandTouchConfig.from_json(config_file)
        else:
            print(f"⚠️  No config.json found for {camera_name}, using defaults")
            config = SimpleHandTouchConfig()
            config.CAMERA_NAME = camera_name
        
        # Create detector with camera-specific config
        detector = SimpleHandTouchDetector(config)
        
        # Find all videos in camera folder
        videos = sorted(list(camera_folder.glob("*.mp4")))
        if not videos:
            print(f"⚠️  No videos found in {camera_folder}")
            continue
        
        print(f"Found {len(videos)} video(s) in {camera_name}")
        print()
        
        # Create output directory for this camera
        output_dir = Path("output") / camera_name
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process all videos for this camera
        for idx, video in enumerate(videos, 1):
            print(f"\n{'🎬 VIDEO {idx}/{len(videos)}':.^70}")
            output_path = output_dir / f"hand_touch_{video.name}"
            detector.process_video(str(video), str(output_path))
            total_videos += 1
    
    print()
    print("=" * 70)
    print(f"✅ ALL VIDEOS PROCESSED! (Total: {total_videos})")
    print("=" * 70)


if __name__ == "__main__":
    main()
