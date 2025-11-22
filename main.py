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
    CASH_COLOR_THRESHOLD = 50  # Minimum pixels of cash color to confirm money (TEMP: lowered to capture debug data)
    CASH_DETECTION_CONFIDENCE = 0.60  # Minimum confidence (0.0-1.0) to confirm cash (TEMP: lowered)
    MAX_CASH_TRANSACTION_SECONDS = 3  # Max duration for normal cash transaction
    
    # Color ranges for Korean currency (HSV format)
    CASH_COLORS = {
        '50000_won': {'lower': [15, 100, 100], 'upper': [30, 255, 255], 'name': '50,000원 (Yellow)'},
        '10000_won': {'lower': [40, 50, 50], 'upper': [80, 255, 255], 'name': '10,000원 (Green)'},
        '5000_won': {'lower': [0, 100, 100], 'upper': [15, 255, 255], 'name': '5,000원 (Red/Pink)'},
        '1000_won': {'lower': [100, 50, 50], 'upper': [130, 255, 255], 'name': '1,000원 (Blue)'}
    }
    
    # Violence Detection
    DETECT_HAND_VELOCITY = True  # Enable velocity-based violence detection
    VIOLENCE_VELOCITY_THRESHOLD = 200  # Pixels/frame threshold for violence (higher = less sensitive)
    
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
            'cash_types': {},  # Count each type of bill detected
            'possible_detections': 0  # Hands close but no material
        }
        
        # Track "possible" detections (hands close but no material)
        self.possible_events = []  # Store all possible detections
        self.current_possible_event = None  # Track ongoing possible event
        self.possible_frame_buffer = {}  # Buffer frames during detection: {event_index: [(frame_num, annotated_frame)]}
        self.current_event_index = 0  # Event counter
        
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
                            # Map to full names: L -> left, R -> right
                            c_hand_name = 'left' if c_hand_type == 'L' else 'right'
                            n_hand_name = 'left' if n_hand_type == 'L' else 'right'
                            c_velocity, c_angle = self._calculate_hand_velocity(cashier['id'], c_hand, c_hand_name, self.stats['frames'])
                            n_velocity, n_angle = self._calculate_hand_velocity(customer['id'], n_hand, n_hand_name, self.stats['frames'])
                            
                            # === VIOLENCE DETECTION (3 METHODS) ===
                            # ONLY detect violence from CUSTOMER, not cashier
                            is_violent = False
                            violence_type = None
                            violence_confidence = 0
                            violence_method = None
                            
                            # Only check CUSTOMER velocity for violence (not cashier)
                            if self.config.DETECT_HAND_VELOCITY:
                                violence_threshold = self.config.VIOLENCE_VELOCITY_THRESHOLD
                                # Only customer fast movement = violence
                                if n_velocity > violence_threshold:
                                    is_violent = True
                                    violence_type = f"🚨 Violence (Customer Attack - {n_velocity:.0f} px/f)"
                                    violence_confidence = min(n_velocity / violence_threshold, 1.0)
                                    violence_method = "Velocity"
                            
                            # If violent movement detected, mark as violence instead of cash
                            if is_violent:
                                print(f"  🚨 VIOLENCE DETECTED ({violence_method}): {violence_type}")
                                print(f"     Confidence: {violence_confidence:.0%}")
                                print(f"     Between: P{cashier['id']} ↔ P{customer['id']} ({hand_type})")
                                if 'violence_count' not in self.stats:
                                    self.stats['violence_count'] = 0
                                self.stats['violence_count'] += 1
                                
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
                                    'cash_type': violence_type,  # Violence label
                                    'cash_bbox': None,
                                    'cash_pixels': 0,
                                    'analysis_scores': {
                                        'violence_confidence': violence_confidence, 
                                        'method': violence_method,
                                        'velocity': max(c_velocity, n_velocity)
                                    },
                                    'is_violence': True  # NEW: Flag for violence
                                }
                                continue  # Skip material analysis
                            
                            # Check for cash/card with 3-Phase Handover Zone Analysis
                            material_detected, material_type, material_bbox, analysis_scores = self._analyze_handover_zone(frame, c_hand, n_hand, draw_debug=True)
                            
                            # Track ALL hand-close events to Possible folder (for debugging)
                            # This includes BOTH material detected and NOT detected
                            if dist < closest_distance:  # Only track closest pair
                                geom = analysis_scores.get('geometric', 0)
                                glare = analysis_scores.get('photometric', 0)
                                color = analysis_scores.get('chromatic', 0)
                                internal_conf = analysis_scores.get('confidence', 0)
                                
                                # Print message based on detection result
                                if material_detected:
                                    print(f"  ✅ Material detected: {material_type} (conf:{internal_conf:.2f}) [G:{geom:.2f} L:{glare:.2f} C:{color:.2f}]")
                                else:
                                    print(f"  ⚠️  Hands close ({dist:.0f}px) but NO MATERIAL - P{cashier['id']}↔P{customer['id']} ({hand_type}) [G:{geom:.2f} L:{glare:.2f} C:{color:.2f}]")
                                
                                # Track ALL hand-close events (detected OR not) for debugging
                                try:
                                    self._track_possible_detection(
                                        frame_num=self.stats['frames'],
                                        p1_id=cashier['id'],
                                        p2_id=customer['id'],
                                        hand_type=hand_type,
                                        distance=dist,
                                        scores={'geometric': geom, 'photometric': glare, 'chromatic': color},
                                        p1_hand=c_hand,
                                        p2_hand=n_hand,
                                        material_detected=material_detected,
                                        material_type=material_type,
                                        confidence=internal_conf,
                                        annotated_frame=frame  # Pass frame for buffering
                                    )
                                except Exception as e:
                                    print(f"  ❌ Error tracking possible detection: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
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
            
            if self.config.DEBUG_MODE and self.transaction_history[pair_key] == 1:
                print(f"  🔄 NEW PAIR: P{trans['p1_id']}↔P{trans['p2_id']} started ({dist:.0f}px)")
            
            # Confirm transaction if it lasts MIN_TRANSACTION_FRAMES or more
            if self.transaction_history[pair_key] >= self.config.MIN_TRANSACTION_FRAMES:
                trans['confirmed'] = True
                trans['duration'] = self.transaction_history[pair_key]
                confirmed_transactions.append(trans)
                
                # Count as confirmed transaction (only once when first confirmed)
                if self.transaction_history[pair_key] == self.config.MIN_TRANSACTION_FRAMES:
                    self.stats['confirmed_transactions'] += 1
                    if self.config.DEBUG_MODE:
                        print(f"  ✅ CONFIRMED: P{trans['p1_id']}↔P{trans['p2_id']} ({trans['distance']:.0f}px, {trans['duration']} frames)")
                    
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
        
        # STEP 5: Update possible event buffer with final annotated frame
        # This ensures clips show EXACTLY what was detected during main processing
        if self.current_possible_event is not None:
            event_idx = self.current_possible_event.get('event_index')
            if event_idx and event_idx in self.possible_frame_buffer:
                # Replace all frames for this event with annotated version
                # (since we only track the current frame per call, just update the last one)
                if len(self.possible_frame_buffer[event_idx]) > 0:
                    last_frame_num = self.possible_frame_buffer[event_idx][-1][0]
                    # Check if this is the current frame
                    if last_frame_num == self.stats['frames']:
                        # Draw possible detection overlay on this frame
                        annotated_possible = output_frame.copy()
                        # Add velocity arrows if debug mode
                        if self.config.DEBUG_MODE:
                            annotated_possible = self._draw_velocity_arrows(annotated_possible, people)
                        self.possible_frame_buffer[event_idx][-1] = (last_frame_num, annotated_possible.copy())
        
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
        chromatic_score, detected_bill, color_analysis = self._filter_chromatic_color(hsv_zone)
        
        # ═══════════════════════════════════════════════════════════════
        # DECISION LOGIC: Combine all filters
        # ═══════════════════════════════════════════════════════════════
        
        scores = {
            'geometric': geometric_score,      # > 0.7 = Card, < 0.3 = Cash
            'photometric': photometric_score,  # > 0.5 = Card (glare), < 0.5 = Cash (matte)
            'chromatic': chromatic_score,      # > 0.6 = Cash (colorful), < 0.4 = Card (gray)
            'bill_type': detected_bill,
            'color_analysis': color_analysis
        }
        
        # Decision tree (prioritize strong signals)
        material_type = None
        confidence = 0
        decision_reason = ""
        
        # Strong Card signal: High glare + Low color + Rectangular shape
        if photometric_score > 0.5 and chromatic_score < 0.4 and geometric_score > 0.6:
            material_type = "💳 Card"
            confidence = (photometric_score + geometric_score + (1 - chromatic_score)) / 3
            decision_reason = "High glare + Low color + Rectangular shape"
        
        # Strong Cash signal: High color + Low glare + Detected bill type
        elif chromatic_score > 0.7 and photometric_score < 0.5 and detected_bill:
            material_type = f"💵 {detected_bill}"
            confidence = chromatic_score
            decision_reason = f"High color ({chromatic_score:.2f}) + Low glare + Bill detected"
        
        # Weak Cash signal: High color only (more strict now)
        elif chromatic_score > 0.75 and detected_bill:
            material_type = f"💵 {detected_bill}"
            confidence = chromatic_score * 0.75  # Lower confidence
            decision_reason = f"High color only ({chromatic_score:.2f})"
        
        # Nothing detected or ambiguous
        else:
            material_type = None
            confidence = 0
            decision_reason = f"Not enough: Geom={geometric_score:.2f} Photo={photometric_score:.2f} Chrom={chromatic_score:.2f}"
        
        # Add confidence and decision to scores dictionary
        scores['confidence'] = confidence
        scores['decision_reason'] = decision_reason
        
        # Store JSON debug data (will be saved with video clip later)
        if self.config.DEBUG_MODE:
            debug_data = {
                "zone_size": f"{roi_x2-roi_x1}x{roi_y2-roi_y1} pixels",
                "hand_distance": f"{hand_distance:.1f} pixels",
                "scores": {
                    "geometric": round(geometric_score, 3),
                    "photometric": round(photometric_score, 3),
                    "chromatic": round(chromatic_score, 3),
                    "confidence": round(confidence, 3)
                },
                "color_detection": color_analysis,
                "thresholds": {
                    "min_pixels": self.config.CASH_COLOR_THRESHOLD,
                    "min_confidence": self.config.CASH_DETECTION_CONFIDENCE,
                    "internal_threshold": 0.5
                },
                "decision": {
                    "material_type": material_type or "None",
                    "detected_bill": detected_bill or "None",
                    "confidence_percent": round(confidence * 100, 1),
                    "reason": decision_reason,
                    "passed_internal": confidence > 0.5,
                    "passed_external": confidence > self.config.CASH_DETECTION_CONFIDENCE
                }
            }
            scores['debug_data'] = debug_data
        
        # Debug visualization
        if draw_debug and self.config.DEBUG_MODE:
            self._draw_handover_debug(frame, roi_x1, roi_y1, roi_x2, roi_y2, 
                                     mid_x, mid_y, scores, material_type, confidence, color_analysis)
        
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
        FILTER C: Chromatic Logic (Color Saturation) - ENHANCED 🔥
        
        Logic:
        - Card: White/Gray/Black (Low saturation)
        - Cash (Korean Won): Green/Yellow/Pink (High saturation)
        
        Enhancements:
        - Morphological operations to remove noise
        - Contour analysis for better shape detection
        
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
        
        # Also check for specific Korean Won bill colors with ENHANCEMENTS
        detected_bill = None
        max_pixels = 0
        color_analysis = {}  # Store detailed analysis for each bill
        
        for bill_type, color_info in self.config.CASH_COLORS.items():
            lower = np.array(color_info['lower'])
            upper = np.array(color_info['upper'])
            
            # Create mask for this bill color
            mask = cv2.inRange(zone_hsv, lower, upper)
            
            pixel_count = np.count_nonzero(mask)
            original_pixel_count = pixel_count
            
            # ENHANCEMENT 1: Morphological operations (remove noise) - OPTIONAL
            if pixel_count > 200:  # Only apply if we have significant detection
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
                pixel_count = np.count_nonzero(mask)  # Recount after cleanup
            
            # Store analysis for this bill type
            color_analysis[color_info['name']] = {
                'pixel_count': pixel_count,
                'original_pixel_count': original_pixel_count,
                'hsv_range': f"H:{lower[0]}-{upper[0]} S:{lower[1]}-{upper[1]} V:{lower[2]}-{upper[2]}",
                'threshold': self.config.CASH_COLOR_THRESHOLD,
                'passed': pixel_count > self.config.CASH_COLOR_THRESHOLD
            }
            
            # Check if this bill color is prominent (use configured threshold)
            min_threshold = self.config.CASH_COLOR_THRESHOLD
            if pixel_count > max_pixels and pixel_count > min_threshold:
                max_pixels = pixel_count
                detected_bill = color_info['name']
        
        # Boost chromatic score if specific bill detected
        if detected_bill and max_pixels > self.config.CASH_COLOR_THRESHOLD:
            chromatic_score = max(chromatic_score, 0.7)
        
        return chromatic_score, detected_bill, color_analysis
    
    def _draw_handover_debug(self, frame, x1, y1, x2, y2, mid_x, mid_y, 
                            scores, material_type, confidence, color_analysis=None):
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
        
        # Draw color analysis (pixel counts for each bill)
        if color_analysis:
            y_offset = y2 + 20
            for bill_name, analysis in color_analysis.items():
                pixel_count = analysis['pixel_count']
                passed = analysis['passed']
                color_text = (0, 255, 0) if passed else (0, 0, 255)
                
                # Show pixel count for each bill type
                text = f"{bill_name}: {pixel_count}px"
                if passed:
                    text += " ✓"
                
                cv2.putText(frame, text, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_text, 1)
                y_offset += 18
        
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
    
    def _track_possible_detection(self, frame_num, p1_id, p2_id, hand_type, distance, scores, p1_hand, p2_hand, 
                                  material_detected=False, material_type=None, confidence=0.0, annotated_frame=None):
        """
        Track ALL hand-close events (material detected OR not) for debugging
        Buffers annotated frames during detection for accurate clips
        """
        # Check if this continues the current event
        pair_key = f"{p1_id}-{p2_id}"
        
        if self.current_possible_event is None:
            # Start new event
            self.current_event_index += 1
            event_idx = self.current_event_index
            
            self.current_possible_event = {
                'event_index': event_idx,
                'start_frame': frame_num,
                'end_frame': frame_num,
                'pair': pair_key,
                'p1_id': p1_id,
                'p2_id': p2_id,
                'hand_type': hand_type,
                'distance': distance,
                'scores': scores,
                'p1_hand': p1_hand,
                'p2_hand': p2_hand,
                'frame_count': 1,
                'avg_scores': scores.copy(),
                'material_detected': material_detected,
                'material_type': material_type,
                'confidence': confidence
            }
            
            # Initialize frame buffer for this event
            self.possible_frame_buffer[event_idx] = []
            
            # Buffer this frame (save ACTUAL detection, not re-run later!)
            if annotated_frame is not None:
                self.possible_frame_buffer[event_idx].append((frame_num, annotated_frame.copy()))
        elif self.current_possible_event['pair'] == pair_key and (frame_num - self.current_possible_event['end_frame']) <= 5:
            # Continue existing event (within 5 frames)
            event = self.current_possible_event
            event['end_frame'] = frame_num
            event['frame_count'] += 1
            
            # Update average scores
            for key in ['geometric', 'photometric', 'chromatic']:
                event['avg_scores'][key] = (
                    (event['avg_scores'][key] * (event['frame_count'] - 1) + scores[key]) 
                    / event['frame_count']
                )
            
            # Update material detection status (if material detected at any frame, mark as detected)
            if material_detected and not event['material_detected']:
                event['material_detected'] = True
                event['material_type'] = material_type
                event['confidence'] = confidence
            elif material_detected and confidence > event.get('confidence', 0):
                # Update to higher confidence detection
                event['material_type'] = material_type
                event['confidence'] = confidence
            
            # Buffer this frame too
            if annotated_frame is not None:
                event_idx = event['event_index']
                if event_idx in self.possible_frame_buffer:
                    self.possible_frame_buffer[event_idx].append((frame_num, annotated_frame.copy()))
        else:
            # Save previous event and start new one (NO THRESHOLD - save all)
            if self.current_possible_event['frame_count'] >= 1:  # Changed from 3 to 1 - save ALL events
                self.possible_events.append(self.current_possible_event.copy())
                self.stats['possible_detections'] += 1
            
            # Start new event with new index
            self.current_event_index += 1
            event_idx = self.current_event_index
            
            self.current_possible_event = {
                'event_index': event_idx,
                'start_frame': frame_num,
                'end_frame': frame_num,
                'pair': pair_key,
                'p1_id': p1_id,
                'p2_id': p2_id,
                'hand_type': hand_type,
                'distance': distance,
                'scores': scores,
                'p1_hand': p1_hand,
                'p2_hand': p2_hand,
                'frame_count': 1,
                'avg_scores': scores.copy(),
                'material_detected': material_detected,
                'material_type': material_type,
                'confidence': confidence
            }
            
            # Initialize frame buffer for this new event
            self.possible_frame_buffer[event_idx] = []
            if annotated_frame is not None:
                self.possible_frame_buffer[event_idx].append((frame_num, annotated_frame.copy()))
    
    def _save_possible_detections(self, video_path, output_base_dir, fps=30):
        """
        Save all "possible" detections to Possible folder with JSON data + video clips
        Groups events that occur within same time period
        """
        # Finalize last event if exists (save ALL, no threshold)
        if self.current_possible_event and self.current_possible_event['frame_count'] >= 1:
            self.possible_events.append(self.current_possible_event.copy())
            self.stats['possible_detections'] += 1
        
        if not self.possible_events:
            return
        
        # Create Possible folder
        possible_dir = Path(output_base_dir) / "Possible"
        possible_dir.mkdir(exist_ok=True, parents=True)
        
        # Group events by time proximity (within 2 seconds)
        grouped_events = self._group_events_by_time(self.possible_events, fps, max_gap_seconds=2.0)
        
        print(f"\n💾 Saving {len(self.possible_events)} hand-close detection(s) to: {possible_dir}")
        print(f"   Grouped into: {len(grouped_events)} clip(s)")
        
        # Save each group (save ALL for debugging, even 1-frame events)
        video_name = Path(video_path).stem
        saved_count = 0
        for group_idx, event_group in enumerate(grouped_events, 1):
            # Use ALL events (including 1-frame) for debugging purposes
            valid_events = event_group
            if not valid_events:
                continue  # Skip empty groups
            
            # Calculate time range
            start_frame = min(e['start_frame'] for e in valid_events)
            end_frame = max(e['end_frame'] for e in valid_events)
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            
            # Create JSON data for this group (only valid events)
            # Don't increment saved_count yet - wait until clip is validated
            temp_group_id = saved_count + 1
            json_data = {
                'video_source': video_name,
                'group_id': temp_group_id,  # Temporary ID
                'time_range': {
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time_seconds': round(start_time, 2),
                    'end_time_seconds': round(end_time, 2),
                    'duration_seconds': round(duration, 2)
                },
                'event_count': len(valid_events),
                'important_note': '✅ Continuous clips: 1s before + detection + 1s after. Frame numbers are ORIGINAL from main video.',
                'how_to_use': 'Video shows "Original Frame: XXXX" continuously. Buffered frames have detection overlays, padding frames are clean.',
                'clip_info': {
                    'padding_before': '1 second before detection',
                    'padding_after': '1 second after detection',
                    'continuous': 'All frames shown sequentially (e.g., 1000→1001→1002→1003...)'
                },
                'events': []
            }
            
            # Add each valid event in this group
            for event in valid_events:
                json_data['events'].append({
                    'frame_range': f"{event['start_frame']}-{event['end_frame']}",
                    'persons': f"P{event['p1_id']} ↔ P{event['p2_id']}",
                    'hand_type': event['hand_type'],
                    'distance_px': round(event['distance'], 1),
                    'frame_count': event['frame_count'],
                    'material_detected': event.get('material_detected', False),
                    'material_type': event.get('material_type', 'None'),
                    'detection_confidence': round(event.get('confidence', 0.0), 3),
                    'material_scores': {
                        'geometric': round(event['avg_scores']['geometric'], 3),
                        'photometric': round(event['avg_scores']['photometric'], 3),
                        'chromatic': round(event['avg_scores']['chromatic'], 3)
                    },
                    'interpretation': {
                        'geometric': 'Card-like' if event['avg_scores']['geometric'] > 0.7 else 'Cash-like' if event['avg_scores']['geometric'] < 0.3 else 'Neutral',
                        'photometric': 'Glare detected' if event['avg_scores']['photometric'] > 0.5 else 'Matte surface',
                        'chromatic': 'Colorful' if event['avg_scores']['chromatic'] > 0.6 else 'Grayscale' if event['avg_scores']['chromatic'] < 0.4 else 'Neutral'
                    },
                    'debug_note': '✅ DETECTED' if event.get('material_detected', False) else '⚠️ NOT DETECTED'
                })
            
            # Temp filenames (will be renamed if successful)
            temp_json_filename = f"possible_{video_name}_groupTEMP_{start_frame:06d}-{end_frame:06d}.json"
            temp_json_path = possible_dir / temp_json_filename
            temp_clip_filename = f"possible_{video_name}_groupTEMP_{start_frame:06d}-{end_frame:06d}.mp4"
            temp_clip_path = possible_dir / temp_clip_filename
            
            padding_seconds = 2  # 2 seconds before and after
            clip_start_frame = max(0, start_frame - int(padding_seconds * fps))
            clip_end_frame = end_frame + int(padding_seconds * fps)
            
            try:
                # Create clip from BUFFERED frames (EXACT frames that were detected!)
                frames_used = self._create_clip_from_buffer(valid_events, temp_clip_path, fps, video_path, clip_start_frame, clip_end_frame)
                
                if frames_used == 0:
                    continue
                
                # Clip created! Increment counter and rename files
                saved_count += 1
                json_data['group_id'] = saved_count  # Update with final ID
                
                # Final filenames
                json_filename = f"possible_{video_name}_group{saved_count:03d}_{start_frame:06d}-{end_frame:06d}.json"
                json_path = possible_dir / json_filename
                clip_filename = f"possible_{video_name}_group{saved_count:03d}_{start_frame:06d}-{end_frame:06d}.mp4"
                clip_path = possible_dir / clip_filename
                
                # Save JSON
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                
                # Rename clip
                temp_clip_path.rename(clip_path)
                
                print(f"  ✅ Group {saved_count}: {len(valid_events)} event(s) at {start_time:.1f}s-{end_time:.1f}s")
                
            except Exception as e:
                if temp_clip_path.exists():
                    temp_clip_path.unlink()
        
        print(f"✅ Saved {saved_count} clip(s) to: {possible_dir}\n")
        
        self.possible_frame_buffer.clear()
    
    def _create_clip_from_buffer(self, valid_events, output_path, fps, video_path, clip_start_frame, clip_end_frame):
        """
        Create video clip from BUFFERED frames with padding to ensure minimum 5 seconds
        Fills gaps between buffered frames from original video for continuity
        """
        # Padding: 1 second before detection, 1 second after detection
        PADDING_BEFORE_SECONDS = 1
        PADDING_AFTER_SECONDS = 1
        PADDING_BEFORE_FRAMES = int(PADDING_BEFORE_SECONDS * fps)
        PADDING_AFTER_FRAMES = int(PADDING_AFTER_SECONDS * fps)
        
        # Collect all buffered frames for this group
        buffered_frames_dict = {}  # {frame_num: annotated_frame}
        for event in valid_events:
            event_idx = event.get('event_index')
            if event_idx and event_idx in self.possible_frame_buffer:
                for frame_num, frame in self.possible_frame_buffer[event_idx]:
                    buffered_frames_dict[frame_num] = frame
        
        if not buffered_frames_dict:
            self._extract_possible_clip(video_path, output_path, clip_start_frame, clip_end_frame, fps)
            return 0
        
        # Find the actual frame range from buffered frames
        buffered_frame_numbers = sorted(buffered_frames_dict.keys())
        actual_start = buffered_frame_numbers[0]
        actual_end = buffered_frame_numbers[-1]
        
        # Add padding: 1 second before, 1 second after
        clip_start_frame = max(0, actual_start - PADDING_BEFORE_FRAMES)
        clip_end_frame = actual_end + PADDING_AFTER_FRAMES
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        first_frame = buffered_frames_dict[buffered_frame_numbers[0]]
        height, width = first_frame.shape[:2]
        
        temp_avi_path = str(output_path).replace('.mp4', '_temp.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_avi_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            raise Exception(f"Cannot create video writer: {output_path}")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame)
        
        frames_written = 0
        buffered_count = 0
        
        for current_frame in range(clip_start_frame, clip_end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
            
            if current_frame in buffered_frames_dict:
                overlay = buffered_frames_dict[current_frame].copy()
                frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
                buffered_count += 1
            
            frame_text = f"Frame: {current_frame}"
            cv2.putText(frame, frame_text, (width - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            out.write(frame)
            frames_written += 1
        
        cap.release()
        out.release()
        
        # Convert to MP4
        try:
            import subprocess
            import shutil
            if shutil.which('ffmpeg'):
                cmd = [
                    'ffmpeg', '-i', temp_avi_path,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-y',
                    str(output_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and Path(output_path).exists():
                    Path(temp_avi_path).unlink(missing_ok=True)
                else:
                    Path(temp_avi_path).rename(output_path.replace('.mp4', '.avi'))
            else:
                Path(temp_avi_path).rename(output_path.replace('.mp4', '.avi'))
        except Exception as e:
            print(f"      ⚠️  Conversion error, using AVI: {e}")
            if Path(temp_avi_path).exists():
                Path(temp_avi_path).rename(output_path.replace('.mp4', '.avi'))
        
        return frames_written
    
    def _extract_possible_clip(self, video_path, output_path, start_frame, end_frame, fps):
        """
        Extract a video clip for a possible detection
        Re-processes frames with detection overlay + velocity arrows
        VALIDATES that hands are actually close in the extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Try MJPEG first (more compatible), fallback to mp4v
        temp_avi_path = str(output_path).replace('.mp4', '_temp.avi')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_avi_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise Exception(f"Cannot create video writer: {output_path}")
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Save original settings
        original_debug = self.config.DEBUG_MODE
        original_velocity = self.config.DETECT_HAND_VELOCITY
        
        # Enable velocity visualization for possible clips
        self.config.DEBUG_MODE = True
        self.config.DETECT_HAND_VELOCITY = True
        
        # Reset detector state for clip
        old_history = self.transaction_history.copy()
        old_person_map = self.person_id_map.copy()
        old_stable_id = self.next_stable_id
        old_persistence = self.cashier_persistence.copy()
        old_hand_history = self.hand_history.copy()
        
        self.transaction_history = {}
        self.person_id_map = {}
        self.next_stable_id = 1
        self.cashier_persistence = {}
        self.hand_history = {}
        
        frames_written = 0
        frames_with_close_hands = 0  # Count frames where hands are actually close
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Re-run detection to get annotated frame with velocity arrows
            try:
                annotated_frame, detections = self.detect_hand_touches(frame)
                
                # Count frames where hands are actually close (detections found)
                if detections:
                    frames_with_close_hands += 1
                
                # Add banner at top showing detection status
                # Check if material was detected in any of these detections
                has_material = any(d.get('cash_detected', False) for d in detections) if detections else False
                
                if has_material:
                    label_text = f"✅ MATERIAL DETECTED - ORIGINAL Frame {frame_idx}"
                    banner_color = (0, 255, 0)  # Green for successful detection
                elif detections:
                    label_text = f"⚠️ HANDS CLOSE - NO MATERIAL - ORIGINAL Frame {frame_idx}"
                    banner_color = (0, 165, 255)  # Orange for failed detection
                else:
                    label_text = f"⚠️ ORIGINAL Frame {frame_idx} - People moved/IDs changed"
                    banner_color = (0, 100, 200)  # Blue warning
                    
                    # Add explanation text
                    cv2.putText(annotated_frame, "Event was detected during main processing",
                               (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 2)
                    cv2.putText(annotated_frame, "but people moved away by clip extraction time",
                               (10, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 2)
                
                banner_width = 750 if not detections else 550
                cv2.rectangle(annotated_frame, (10, 10), (banner_width, 50), banner_color, -1)
                cv2.rectangle(annotated_frame, (10, 10), (banner_width, 50), (255, 255, 255), 3)
                cv2.putText(annotated_frame, label_text, (20, 37),
                           cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2)
                
                # Add material analysis info panel (bottom-left corner)
                panel_x = 10
                panel_y = height - 160
                panel_width = 400
                panel_height = 150
                
                # Semi-transparent background
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                # Border
                cv2.rectangle(annotated_frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height),
                             (0, 165, 255), 2)
                
                # Title - changes based on detection status
                if has_material:
                    title_text = "MATERIAL ANALYSIS - DETECTED!"
                    title_color = (0, 255, 0)
                else:
                    title_text = "MATERIAL ANALYSIS - NOT DETECTED"
                    title_color = (0, 165, 255)
                
                cv2.putText(annotated_frame, title_text,
                           (panel_x + 10, panel_y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, title_color, 2)
                
                # Legend
                y_offset = panel_y + 50
                cv2.putText(annotated_frame, "Geometric (Shape): <0.3=Cash, >0.7=Card",
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 25
                cv2.putText(annotated_frame, "Photometric (Glare): >0.5=Card reflection",
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 25
                cv2.putText(annotated_frame, "Chromatic (Color): >0.6=Cash, <0.4=Card",
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 25
                cv2.putText(annotated_frame, "See JSON for all scores + detection status",
                           (panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
                
                out.write(annotated_frame)
                frames_written += 1
            except Exception as e:
                print(f"  ⚠️  Frame {frame_idx} detection error: {e}")
                # Write original frame with error message
                cv2.putText(frame, f"Detection error on frame {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                out.write(frame)
                frames_written += 1
        
        cap.release()
        out.release()
        
        # Restore original settings
        self.config.DEBUG_MODE = original_debug
        self.config.DETECT_HAND_VELOCITY = original_velocity
        self.transaction_history = old_history
        self.person_id_map = old_person_map
        self.next_stable_id = old_stable_id
        self.cashier_persistence = old_persistence
        self.hand_history = old_hand_history
        
        if frames_written == 0:
            raise Exception("No frames extracted")
        
        # For debugging purposes, save ALL clips but warn about low interaction
        # Don't filter out - let user see what was tracked
        close_hands_ratio = frames_with_close_hands / frames_written if frames_written > 0 else 0
        
        # Just print a warning if low interaction (but don't skip)
        if close_hands_ratio < 0.1:  # Less than 10%
            print(f"      ⚠️  Warning: Only {frames_with_close_hands}/{frames_written} frames ({close_hands_ratio:.1%}) show close hands")
        
        # Convert AVI to MP4 using ffmpeg if available
        try:
            import subprocess
            import shutil
            if shutil.which('ffmpeg'):
                # Convert to MP4
                cmd = [
                    'ffmpeg', '-i', temp_avi_path,
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '23',
                    '-y',  # Overwrite
                    str(output_path)
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and Path(output_path).exists():
                    # Delete temp AVI
                    Path(temp_avi_path).unlink(missing_ok=True)
                else:
                    # FFmpeg failed, rename AVI to MP4
                    Path(temp_avi_path).rename(output_path)
            else:
                # No ffmpeg, just rename AVI to MP4
                Path(temp_avi_path).rename(output_path)
        except Exception as e:
            print(f"  ⚠️  Conversion error, using AVI: {e}")
            # Just rename the AVI file
            if Path(temp_avi_path).exists():
                Path(temp_avi_path).rename(output_path.replace('.mp4', '.avi'))
    
    def _group_events_by_time(self, events, fps, max_gap_seconds=2.0):
        """
        Group events that occur within max_gap_seconds of each other
        Returns list of event groups
        """
        if not events:
            return []
        
        # Sort events by start frame
        sorted_events = sorted(events, key=lambda e: e['start_frame'])
        
        groups = []
        current_group = [sorted_events[0]]
        max_gap_frames = max_gap_seconds * fps
        
        for event in sorted_events[1:]:
            # Check if this event is close to the last event in current group
            last_event_end = current_group[-1]['end_frame']
            if event['start_frame'] - last_event_end <= max_gap_frames:
                # Add to current group
                current_group.append(event)
            else:
                # Start new group
                groups.append(current_group)
                current_group = [event]
        
        # Add last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
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
    
    def _analyze_violence_patterns(self, frame, hand1, hand2, draw_debug=False):
        """
        ═══════════════════════════════════════════════════════════════
        🚨 VIOLENCE PATTERN ANALYSIS (NEW METHOD!)
        ═══════════════════════════════════════════════════════════════
        
        Same zone as cash detection, but analyze for violence indicators:
        
        Violence Indicators:
        1. Clenched fist (closed hand shape)
        2. Grabbing motion (hands overlap/clutching)
        3. Aggressive red/dark colors (blood, dark clothing in struggle)
        4. Irregular motion patterns (shaking, jerking)
        5. High contact pressure (hands pressed tightly)
        
        Returns: (is_violent, violence_type, confidence)
        ═══════════════════════════════════════════════════════════════
        """
        # Create handover zone (same as cash detection)
        x1, y1 = int(hand1[0]), int(hand1[1])
        x2, y2 = int(hand2[0]), int(hand2[1])
        
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        
        hand_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        zone_size = max(int(hand_distance * 0.4), 30)
        
        roi_x1 = max(0, mid_x - zone_size)
        roi_y1 = max(0, mid_y - zone_size)
        roi_x2 = min(frame.shape[1], mid_x + zone_size)
        roi_y2 = min(frame.shape[0], mid_y + zone_size)
        
        zone = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if zone.size == 0 or zone.shape[0] < 10 or zone.shape[1] < 10:
            return False, None, 0
        
        # Convert to HSV and grayscale
        hsv_zone = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        gray_zone = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        
        violence_scores = {}
        
        # INDICATOR 1: Clenched fist / Closed hand (detect dark closed shape)
        violence_scores['fist'] = self._detect_clenched_fist(gray_zone)
        
        # INDICATOR 2: Grabbing motion (hands overlap/pressed together)
        violence_scores['grabbing'] = self._detect_grabbing_motion(hand_distance, zone)
        
        # INDICATOR 3: Aggressive colors (red/blood, dark struggle)
        violence_scores['aggressive_color'] = self._detect_aggressive_colors(hsv_zone)
        
        # INDICATOR 4: Irregular motion (calculated from velocity)
        violence_scores['motion'] = 0.0  # Will be filled by calling function
        
        # Calculate overall violence confidence
        # Weights: grabbing (40%), fist (30%), color (20%), motion (10%)
        confidence = (
            violence_scores['grabbing'] * 0.4 +
            violence_scores['fist'] * 0.3 +
            violence_scores['aggressive_color'] * 0.2 +
            violence_scores['motion'] * 0.1
        )
        
        # Determine violence type
        violence_type = None
        if confidence > 0.6:
            if violence_scores['grabbing'] > 0.7:
                violence_type = "🚨 Violence (Grabbing)"
            elif violence_scores['fist'] > 0.7:
                violence_type = "🚨 Violence (Clenched Fist)"
            elif violence_scores['aggressive_color'] > 0.7:
                violence_type = "🚨 Violence (Aggressive Contact)"
            else:
                violence_type = "🚨 Violence (Physical Contact)"
        
        # Debug visualization
        if draw_debug and self.config.DEBUG_MODE and violence_type:
            # Draw violence zone (RED dashed rectangle)
            dash_length = 8
            color = (0, 0, 255)  # Red for violence
            
            for i in range(roi_x1, roi_x2, dash_length * 2):
                cv2.line(frame, (i, roi_y1), (min(i + dash_length, roi_x2), roi_y1), color, 3)
                cv2.line(frame, (i, roi_y2), (min(i + dash_length, roi_x2), roi_y2), color, 3)
            for i in range(roi_y1, roi_y2, dash_length * 2):
                cv2.line(frame, (roi_x1, i), (roi_x1, min(i + dash_length, roi_y2)), color, 3)
                cv2.line(frame, (roi_x2, i), (roi_x2, min(i + dash_length, roi_y2)), color, 3)
            
            # Draw violence label
            cv2.putText(frame, "VIOLENCE ZONE", (roi_x1 + 5, roi_y1 - 5),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
            
            # Show scores
            y_offset = roi_y2 + 20
            cv2.putText(frame, f"Grab: {violence_scores['grabbing']:.2f}", 
                       (roi_x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(frame, f"Fist: {violence_scores['fist']:.2f}", 
                       (roi_x1, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(frame, f"{violence_type} ({confidence:.0%})", 
                       (roi_x1, y_offset + 35), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
        
        is_violent = confidence > 0.6
        return is_violent, violence_type, confidence
    
    def _detect_clenched_fist(self, gray_zone):
        """Detect if hand is clenched (fist) vs open (passing object)"""
        # Clenched fist = compact dark blob
        # Open hand = spread fingers
        
        # Apply threshold to detect dark areas (hand shadow/closed fingers)
        _, binary = cv2.threshold(gray_zone, 80, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get largest contour (hand)
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area < 50:
            return 0.0
        
        # Calculate solidity (area / convex hull area)
        # Clenched fist = high solidity (compact)
        # Open hand = low solidity (spread fingers)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        
        if hull_area == 0:
            return 0.0
        
        solidity = area / hull_area
        
        # Clenched fist: solidity > 0.85 (very compact)
        # Open hand: solidity < 0.7 (spread)
        if solidity > 0.85:
            return 0.9  # High confidence fist
        elif solidity > 0.75:
            return 0.6  # Medium confidence
        else:
            return 0.2  # Likely open hand
    
    def _detect_grabbing_motion(self, hand_distance, zone):
        """Detect if hands are grabbing/clutching vs gentle pass"""
        # Grabbing = very close contact (<20px)
        # Passing = moderate distance (40-80px)
        
        if hand_distance < 20:
            return 0.9  # Very close = likely grabbing
        elif hand_distance < 40:
            return 0.7  # Close = possible grabbing
        elif hand_distance < 60:
            return 0.3  # Moderate = likely normal
        else:
            return 0.1  # Far = definitely not grabbing
    
    def _detect_aggressive_colors(self, hsv_zone):
        """Detect aggressive colors (red/blood, dark clothing in struggle)"""
        # Extract hue and value channels
        hue = hsv_zone[:, :, 0]
        saturation = hsv_zone[:, :, 1]
        value = hsv_zone[:, :, 2]
        
        # Detect red colors (blood, aggressive)
        # Red is at hue 0-10 or 160-180
        red_mask1 = cv2.inRange(hsv_zone, np.array([0, 100, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv_zone, np.array([160, 100, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = np.count_nonzero(red_mask)
        
        # Detect very dark areas (struggle, black clothing)
        dark_mask = cv2.inRange(hsv_zone, np.array([0, 0, 0]), np.array([180, 255, 80]))
        dark_pixels = np.count_nonzero(dark_mask)
        
        total_pixels = hsv_zone.shape[0] * hsv_zone.shape[1]
        
        # Red indicator (blood, aggression)
        red_ratio = red_pixels / total_pixels
        red_score = min(red_ratio * 10, 1.0)  # 10% red = max score
        
        # Dark indicator (struggle, dark clothing)
        dark_ratio = dark_pixels / total_pixels
        dark_score = min(dark_ratio * 2, 1.0)  # 50% dark = max score
        
        # Combine scores
        aggressive_score = max(red_score, dark_score * 0.5)  # Red is stronger indicator
        
        return aggressive_score
    
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
    
    def _draw_velocity_arrows(self, frame, people):
        """Draw velocity arrows on hands to show movement speed and direction"""
        if not self.config.DEBUG_MODE:
            return frame
        
        for person in people:
            if person['id'] not in self.hand_history:
                continue
            
            # Draw velocity vectors for both hands
            for hand_type in ['left', 'right']:
                if hand_type not in self.hand_history[person['id']]:
                    continue
                if len(self.hand_history[person['id']][hand_type]) < 2:
                    continue
                
                history = self.hand_history[person['id']][hand_type]
                current = history[-1]
                previous = history[-2]
                
                # Ensure current and previous are proper (x, y) tuples
                if not isinstance(current, (tuple, list)) or len(current) < 2:
                    continue
                if not isinstance(previous, (tuple, list)) or len(previous) < 2:
                    continue
                
                # Extract only x, y coordinates (ignore confidence if present)
                curr_x, curr_y = int(current[0]), int(current[1])
                prev_x, prev_y = int(previous[0]), int(previous[1])
                
                # Calculate velocity vector
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                velocity = math.sqrt(dx*dx + dy*dy)
                
                # Draw velocity vector (arrow)
                if velocity > 10:  # Only show if significant movement
                    try:
                        # Scale arrow for visibility (multiply by 3)
                        end_x = int(curr_x + dx * 3)
                        end_y = int(curr_y + dy * 3)
                        
                        # Color based on velocity (green = slow, yellow = medium, red = fast)
                        if velocity < 50:
                            vel_color = (0, 255, 0)  # Green
                        elif velocity < 100:
                            vel_color = (0, 255, 255)  # Yellow
                        elif velocity < 150:
                            vel_color = (0, 165, 255)  # Orange
                        else:
                            vel_color = (0, 0, 255)  # Red (violence threshold)
                        
                        # Draw arrow - ensure points are (x, y) tuples of ints
                        start_pt = (curr_x, curr_y)
                        end_pt = (end_x, end_y)
                        
                        cv2.arrowedLine(frame, start_pt, end_pt, vel_color, 2, tipLength=0.3)
                        
                        # Draw velocity text
                        vel_text = f"{velocity:.0f} px/f"
                        cv2.putText(frame, vel_text, 
                                  (curr_x - 30, curr_y + 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, vel_color, 1)
                    except Exception as e:
                        # Silently skip if arrow drawing fails
                        pass
        
        return frame
    
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
                
                # VELOCITY VISUALIZATION (if debug mode)
                if self.config.DEBUG_MODE and person['id'] in self.hand_history:
                    # Draw velocity vectors for both hands
                    for hand_type in ['left', 'right']:
                        if hand_type in self.hand_history[person['id']] and len(self.hand_history[person['id']][hand_type]) >= 2:
                            history = self.hand_history[person['id']][hand_type]
                            current = history[-1]
                            previous = history[-2]
                            
                            # Ensure current and previous are proper (x, y) tuples
                            if not isinstance(current, (tuple, list)) or len(current) < 2:
                                continue
                            if not isinstance(previous, (tuple, list)) or len(previous) < 2:
                                continue
                            
                            # Extract only x, y coordinates (ignore confidence if present)
                            curr_x, curr_y = int(current[0]), int(current[1])
                            prev_x, prev_y = int(previous[0]), int(previous[1])
                            
                            # Calculate velocity vector
                            dx = curr_x - prev_x
                            dy = curr_y - prev_y
                            velocity = math.sqrt(dx*dx + dy*dy)
                            
                            # Draw velocity vector (arrow)
                            if velocity > 10:  # Only show if significant movement
                                try:
                                    # Scale arrow for visibility (multiply by 3)
                                    end_x = int(curr_x + dx * 3)
                                    end_y = int(curr_y + dy * 3)
                                    
                                    # Color based on velocity (green = slow, yellow = medium, red = fast)
                                    if velocity < 50:
                                        vel_color = (0, 255, 0)  # Green
                                    elif velocity < 100:
                                        vel_color = (0, 255, 255)  # Yellow
                                    elif velocity < 150:
                                        vel_color = (0, 165, 255)  # Orange
                                    else:
                                        vel_color = (0, 0, 255)  # Red (violence threshold)
                                    
                                    # Draw arrow - ensure points are (x, y) tuples of ints
                                    start_pt = (curr_x, curr_y)
                                    end_pt = (end_x, end_y)
                                    
                                    # Verify tuples are valid
                                    assert len(start_pt) == 2 and len(end_pt) == 2, f"Invalid tuple length: start={len(start_pt)}, end={len(end_pt)}"
                                    
                                    cv2.arrowedLine(frame, start_pt, end_pt, vel_color, 2, tipLength=0.3)
                                    
                                    # Draw velocity text
                                    vel_text = f"{velocity:.0f} px/f"
                                    cv2.putText(frame, vel_text, 
                                              (curr_x - 30, curr_y + 25),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, vel_color, 1)
                                except Exception as e:
                                    # Silently skip if arrow drawing fails (don't crash detection)
                                    if self.config.DEBUG_MODE:
                                        print(f"⚠️  Velocity arrow error: {e}")
                
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
                
                # Draw transaction label above the line (shows detection type + number)
                label_y = min(trans['p1_hand'][1], trans['p2_hand'][1]) - 15
                
                # Determine label text and color based on material/violence
                material_type = trans.get('cash_type', 'EXCHANGE')
                is_violence = trans.get('is_violence', False)
                
                if is_violence or (material_type and ('🚨' in material_type or 'Violence' in material_type)):
                    # Violence detection
                    text = "VIOLENCE DETECTED"
                    label_color = (0, 0, 255)  # Red
                elif material_type and ('💳' in material_type or 'Card' in material_type):
                    # Card transaction
                    text = "CARD TRANSACTION"
                    label_color = (255, 100, 0)  # Blue
                else:
                    # Cash transaction
                    text = "CASH TRANSACTION"
                    label_color = (0, 255, 0)  # Green
                
                # Background box for label
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)[0]
                box_x1 = mid_x - text_size[0]//2 - 8
                box_y1 = label_y - text_size[1] - 8
                box_x2 = mid_x + text_size[0]//2 + 8
                box_y2 = label_y + 8
                
                # Draw background box
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), label_color, -1)
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 3)
                
                # Draw text
                cv2.putText(frame, text, (mid_x - text_size[0]//2, label_y),
                           cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw counters at top (separated by type)
        violence_count = sum(1 for t in transactions if t.get('is_violence', False))
        cash_count = len(transactions) - violence_count
        
        # Violence counter (RED)
        if violence_count > 0:
            cv2.putText(frame, f"VIOLENCE: {violence_count}", (10, 30),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 3)
            cv2.putText(frame, f"VIOLENCE: {violence_count}", (10, 30),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Cash counter (GREEN)
        if cash_count > 0:
            y_offset = 70 if violence_count > 0 else 30
            cv2.putText(frame, f"CASH: {cash_count}", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 3)
            cv2.putText(frame, f"CASH: {cash_count}", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
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
        
        # DETECTION COUNTER (top-right corner)
        if self.config.DEBUG_MODE:
            y_start = 30
            x_start = frame.shape[1] - 200
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_start - 10, y_start - 25), 
                         (frame.shape[1] - 10, y_start + 60), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Detection counts
            violence_count = self.stats.get('violence_count', 0)
            cash_count = self.stats.get('cash_detections', 0)
            
            cv2.putText(frame, f"DETECTIONS:", (x_start, y_start),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Violence: {violence_count}", (x_start, y_start + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.putText(frame, f"Cash: {cash_count}", (x_start, y_start + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
            # Velocity legend (bottom-right)
            if self.config.DETECT_HAND_VELOCITY:
                y_legend = frame.shape[0] - 120
                x_legend = frame.shape[1] - 180
                
                # Background
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (x_legend - 10, y_legend - 10), 
                             (frame.shape[1] - 10, frame.shape[0] - 40), 
                             (0, 0, 0), -1)
                cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
                
                # Legend title
                cv2.putText(frame, "VELOCITY:", (x_legend, y_legend),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                
                # Color scale
                cv2.arrowedLine(frame, (x_legend, y_legend + 15), 
                               (x_legend + 30, y_legend + 15), (0, 255, 0), 2)
                cv2.putText(frame, "<50 px/f", (x_legend + 35, y_legend + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                
                cv2.arrowedLine(frame, (x_legend, y_legend + 35), 
                               (x_legend + 30, y_legend + 35), (0, 255, 255), 2)
                cv2.putText(frame, "50-100", (x_legend + 35, y_legend + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                
                cv2.arrowedLine(frame, (x_legend, y_legend + 55), 
                               (x_legend + 30, y_legend + 55), (0, 0, 255), 2)
                cv2.putText(frame, ">100 VIOLENCE", (x_legend + 35, y_legend + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
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
            'cash_types': {},
            'possible_detections': 0
        }
        # Reset possible detection tracking
        self.possible_events = []
        self.current_possible_event = None
        self.possible_frame_buffer = {}
        self.current_event_index = 0
        
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
        
        # Print comprehensive detection statistics
        print(f"\n{'='*70}")
        print(f"📊 DETECTION SUMMARY REPORT")
        print(f"{'='*70}")
        
        # Violence statistics
        violence_count = self.stats.get('violence_count', 0)
        print(f"\n🚨 VIOLENCE DETECTIONS: {violence_count}")
        if violence_count > 0:
            print(f"   ⚠️  Violence events detected in video")
            print(f"   Check output video for details")
        else:
            print(f"   ✅ No violence detected")
        
        # Cash detection statistics
        if self.config.DETECT_CASH_COLOR:
            print(f"\n💵 CASH DETECTIONS: {self.stats['cash_detections']}")
            if self.stats['cash_types']:
                print(f"   Breakdown by bill type:")
                for cash_type, count in sorted(self.stats['cash_types'].items(), key=lambda x: x[1], reverse=True):
                    # Exclude violence types from cash summary
                    if '🚨' not in cash_type and 'Violence' not in cash_type:
                        print(f"      • {cash_type}: {count}x")
            else:
                print(f"   No cash transactions detected")
        
        # Overall statistics
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total frames processed: {self.stats['frames']}")
        print(f"   Confirmed transactions: {self.stats['confirmed_transactions']}")
        total_detections = violence_count + self.stats['cash_detections']
        print(f"   Total detections: {total_detections}")
        
        # Possible detections (hands close but no material)
        if self.stats.get('possible_detections', 0) > 0:
            print(f"\n⚠️  POSSIBLE DETECTIONS (Hands close, no material):")
            print(f"   Count: {self.stats['possible_detections']}")
            print(f"   Saved to: Possible/ folder with JSON data")
        
        print(f"\n{'='*70}")
        print(f"💾 Output saved: {output_path}")
        print(f"{'='*70}")
        
        # Save possible detections (hands close but no material) to Possible folder
        if self.possible_events or self.current_possible_event:
            print(f"\n🔍 Possible detections found:")
            print(f"   - Saved events: {len(self.possible_events)}")
            print(f"   - Current event: {'Yes' if self.current_possible_event else 'No'}")
            output_base_dir = Path(output_path).parent
            try:
                self._save_possible_detections(video_path, output_base_dir, fps)
            except Exception as e:
                print(f"❌ Error saving possible detections: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n⚠️  No possible detections to save (hands were detected close but tracking didn't capture events)")
        
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
