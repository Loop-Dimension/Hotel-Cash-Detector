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
                            # Check for cash color between hands (with debug visualization)
                            cash_detected, cash_type, cash_bbox, pixel_count = self._detect_cash_color(frame, c_hand, n_hand, draw_debug=True)
                            
                            # Debug: Print when hands are close but NO cash detected
                            if self.config.DETECT_CASH_COLOR and not cash_detected and self.config.DEBUG_MODE:
                                if dist < closest_distance:  # Only print for closest pair
                                    print(f"  ⚠️  Hands close ({dist:.0f}px) but NO CASH detected - P{cashier['id']}↔P{customer['id']} ({hand_type})")
                            
                            # Only confirm if BOTH hand proximity AND cash color detected
                            if self.config.DETECT_CASH_COLOR and not cash_detected:
                                continue  # Skip this if cash color not found
                            
                            closest_distance = dist
                            closest_transaction = {
                                'p1_id': cashier['id'],
                                'p2_id': customer['id'],
                                'p1_hand': c_hand,
                                'p2_hand': n_hand,
                                'hand_type': hand_type,
                                'distance': dist,
                                'cashier_customer_pair': f"C{cashier['id']}-P{customer['id']}",
                                'cash_detected': cash_detected,
                                'cash_type': cash_type,
                                'cash_bbox': cash_bbox,
                                'cash_pixels': pixel_count
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
                    
                    # Debug: Print cash detection details
                    if self.config.DETECT_CASH_COLOR and trans.get('cash_detected'):
                        self.stats['cash_detections'] += 1
                        
                        # Track cash type
                        cash_type = trans.get('cash_type', 'Unknown')
                        if cash_type not in self.stats['cash_types']:
                            self.stats['cash_types'][cash_type] = 0
                        self.stats['cash_types'][cash_type] += 1
                        
                        print(f"\n  💵 CASH DETECTED!")
                        print(f"     Type: {cash_type}")
                        print(f"     Pixels: {trans.get('cash_pixels', 0)}")
                        if trans.get('cash_bbox'):
                            cx1, cy1, cx2, cy2 = trans['cash_bbox']
                            print(f"     Location: ({cx1}, {cy1}) to ({cx2}, {cy2})")
                            print(f"     Size: {cx2-cx1}x{cy2-cy1} pixels")
                        print(f"     Between: P{trans['p1_id']} ↔ P{trans['p2_id']} ({trans['hand_type']})")
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
    
    def _detect_cash_color(self, frame, hand1, hand2, draw_debug=False):
        """
        Detect Korean won bill colors in the region between two hands
        Focus on the MIDDLE AREA between hands (where cash actually is)
        Returns: (detected, cash_type, cash_bbox, pixel_count)
        """
        if not self.config.DETECT_CASH_COLOR:
            return False, None, None, 0
        
        # Create region of interest (ROI) CENTERED between two hands
        x1, y1 = int(hand1[0]), int(hand1[1])
        x2, y2 = int(hand2[0]), int(hand2[1])
        
        # Calculate midpoint between hands
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        
        # Calculate distance between hands
        hand_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Create SMALL focused box around midpoint (where cash actually is)
        # Box size scales with hand distance but stays reasonable
        box_size = min(int(hand_distance * 0.8), 150)  # Max 150px, 80% of hand distance
        
        roi_x1 = max(0, mid_x - box_size)
        roi_y1 = max(0, mid_y - box_size)
        roi_x2 = min(frame.shape[1], mid_x + box_size)
        roi_y2 = min(frame.shape[0], mid_y + box_size)
        
        # Debug: Draw search area (dashed yellow rectangle) and midpoint
        if draw_debug and self.config.DEBUG_MODE:
            # Draw SMALL focused search box (yellow dashed)
            dash_length = 10
            for i in range(roi_x1, roi_x2, dash_length * 2):
                cv2.line(frame, (i, roi_y1), (min(i + dash_length, roi_x2), roi_y1), (0, 255, 255), 2)
                cv2.line(frame, (i, roi_y2), (min(i + dash_length, roi_x2), roi_y2), (0, 255, 255), 2)
            for i in range(roi_y1, roi_y2, dash_length * 2):
                cv2.line(frame, (roi_x1, i), (roi_x1, min(i + dash_length, roi_y2)), (0, 255, 255), 2)
                cv2.line(frame, (roi_x2, i), (roi_x2, min(i + dash_length, roi_y2)), (0, 255, 255), 2)
            
            # Draw midpoint (where we're looking for cash)
            cv2.circle(frame, (mid_x, mid_y), 8, (0, 255, 255), -1)  # Yellow dot
            cv2.circle(frame, (mid_x, mid_y), 8, (0, 0, 0), 2)  # Black border
            
            # Draw label
            cv2.putText(frame, "CASH SEARCH", (roi_x1 + 5, roi_y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Extract ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        if roi.size == 0:
            return False, None, None, 0
        
        # Convert to HSV for better color detection
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check for each Korean won bill color
        best_match = None
        max_pixels = 0
        cash_mask = None
        
        for bill_type, color_info in self.config.CASH_COLORS.items():
            lower = np.array(color_info['lower'])
            upper = np.array(color_info['upper'])
            
            # Create mask for this color
            mask = cv2.inRange(hsv_roi, lower, upper)
            
            # Count pixels matching this color
            pixel_count = np.count_nonzero(mask)
            
            if pixel_count > max_pixels and pixel_count > self.config.CASH_COLOR_THRESHOLD:
                max_pixels = pixel_count
                best_match = color_info['name']
                cash_mask = mask
        
        # If cash detected, find its bounding box
        if best_match and cash_mask is not None:
            # Find contours of cash
            contours, _ = cv2.findContours(cash_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (main cash bill)
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Convert back to frame coordinates
                cash_bbox = (roi_x1 + x, roi_y1 + y, roi_x1 + x + w, roi_y1 + y + h)
                
                return True, best_match, cash_bbox, max_pixels
        
        return False, None, None, 0
    
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
                
                # Draw CASH BOX if detected (BRIGHT CYAN SQUARE)
                if trans.get('cash_detected') and trans.get('cash_bbox'):
                    cx1, cy1, cx2, cy2 = trans['cash_bbox']
                    # Draw bright cyan rectangle around cash
                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 0), 3)  # Bright cyan
                    
                    # Draw cash type label
                    cash_label = trans.get('cash_type', 'CASH')
                    cash_label_size = cv2.getTextSize(cash_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Background for cash label
                    cv2.rectangle(frame, (cx1, cy1 - 25), (cx1 + cash_label_size[0] + 10, cy1), 
                                 (255, 255, 0), -1)  # Cyan background
                    cv2.rectangle(frame, (cx1, cy1 - 25), (cx1 + cash_label_size[0] + 10, cy1), 
                                 (0, 0, 0), 2)  # Black border
                    
                    # Draw cash label text
                    cv2.putText(frame, cash_label, (cx1 + 5, cy1 - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    
                    # Draw pixel count
                    pixel_text = f"{trans.get('cash_pixels', 0)} px"
                    cv2.putText(frame, pixel_text, (cx1 + 5, cy2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
                
                # Calculate midpoint for label
                mid_x = (trans['p1_hand'][0] + trans['p2_hand'][0]) // 2
                mid_y = (trans['p1_hand'][1] + trans['p2_hand'][1]) // 2
                
                # Draw "CASH TRANSACTION" label above the line
                label_y = min(trans['p1_hand'][1], trans['p2_hand'][1]) - 15
                
                # Background box for label
                text = "CASH TRANSACTION"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                box_x1 = mid_x - text_size[0]//2 - 5
                box_y1 = label_y - text_size[1] - 5
                box_x2 = mid_x + text_size[0]//2 + 5
                box_y2 = label_y + 5
                
                # Draw green background box
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), -1)
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), 2)
                
                # Draw text
                cv2.putText(frame, text, (mid_x - text_size[0]//2, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw counter and frame number at top
        counter_text = f"Transactions: {len(transactions)}"
        cv2.putText(frame, counter_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show cash detection status
        if self.config.DETECT_CASH_COLOR:
            cash_status = "Cash Detection: ON"
            cash_color = (0, 255, 255)  # Yellow
            cv2.putText(frame, cash_status, (frame.shape[1] - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, cash_color, 2)
            
            # Show detected cash types in this frame
            if transactions:
                for idx, trans in enumerate(transactions):
                    if trans.get('cash_detected'):
                        cash_label = f"💵 {trans.get('cash_type', 'Cash')}"
                        y_pos = 60 + (idx * 25)
                        cv2.putText(frame, cash_label, (frame.shape[1] - 250, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
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
