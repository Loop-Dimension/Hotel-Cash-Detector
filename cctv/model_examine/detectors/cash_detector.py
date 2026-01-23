"""
Cash Transaction Detector

3-Track detection system:
- Track 1 (potential_cash): Hand touch detection → immediate event
- Track 2 (cash): Hand touch → Cash drawer deposit → confirmed event
- Track 3 (cash_object): YOLO object detection for cash → direct detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

from .base_detector import BaseDetector, Detection


class CashDetector(BaseDetector):
    """
    Cash Transaction Detector with 3-Track System

    Track 1: potential_cash
        - Detects hand proximity between cashier and customer
        - Generates immediate potential_cash event
        - Cooldown-based

    Track 2: cash
        - After hand touch, tracks cashier's hand
        - If hand enters cash drawer zone within tracking duration → cash event
        - Cooldown-based (separate from Track 1)

    Track 3: cash_object
        - YOLO object detection model for direct cash detection
        - Triggers when cash object detected with confidence >= 0.7
        - Cooldown-based (separate from Track 1 & 2)
    """

    # Model priority for pose detection (Track 1 & 2)
    POSE_MODEL_PRIORITY = [
        "yolo26s-pose.pt",
        "yolov8s-pose.pt",
        "yolov8n-pose.pt"
    ]

    # Model priority for cash object detection (Track 3)
    CASH_OBJECT_MODEL_PRIORITY = [
        "yolo26s.pt",
        "yolo26n.pt"
    ]

    def __init__(self, config: Dict = None):
        """
        Initialize Cash Detector

        Config options:
            - cashier_zone: Polygon [[x,y], ...] for cashier area
            - cash_drawer_zone: Polygon for cash drawer area
            - hand_touch_distance: Pixel threshold for hand proximity (default: 100)
            - detection_mode: 'simple' or 'strict' (default: 'strict')
            - hand_tracking_duration: Frames to track after touch (default: 600)
            - potential_cash_cooldown: Frames between potential_cash events (default: 300)
            - transaction_cooldown: Frames between cash events (default: 300)
            - cash_object_cooldown: Frames between cash_object events (default: 300)
            - cash_object_confidence: Min confidence for cash object detection (default: 0.7)
        """
        super().__init__(config)

        # Zone configurations
        self.cashier_zone = self.config.get('cashier_zone', None)
        self.cash_drawer_zone = self.config.get('cash_drawer_zone', None)

        # Detection parameters
        self.hand_touch_distance = self.config.get('hand_touch_distance', 100)
        self.detection_mode = self.config.get('detection_mode', 'strict')

        # ==================== TRACK 1: potential_cash ====================
        self.last_potential_cash_frame = -1800
        self.potential_cash_cooldown = self.config.get('potential_cash_cooldown', 300)  # 10s at 30fps

        # ==================== TRACK 2: cash (drawer deposit) ====================
        self.pending_transaction = None
        self.tracking_cashier_hands = False
        self.frames_since_touch = 0
        self.touch_frame = -1
        self.hand_tracking_duration = self.config.get('hand_tracking_duration', 600)  # 20s at 30fps
        self.last_transaction_frame = -1800
        self.transaction_cooldown = self.config.get('transaction_cooldown', 300)  # 10s at 30fps

        # ==================== TRACK 3: cash_object (YOLO detection) ====================
        self.cash_object_model = None
        self.last_cash_object_frame = -1800
        self.cash_object_cooldown = self.config.get('cash_object_cooldown', 300)  # 10s at 30fps
        self.cash_object_confidence = self.config.get('cash_object_confidence', 0.55)
        self.cash_object_debug_confidence = self.config.get('cash_object_debug_confidence', 0.2)
        self.last_debug_detections = []

        # Logging
        self.enable_logs = self.config.get('enable_logs', True)

    def initialize(self) -> bool:
        """Load YOLO Pose model and Cash Object model"""
        self._setup_device()

        # Load pose model for Track 1 & 2
        self.model = self._load_yolo_model(self.POSE_MODEL_PRIORITY, task='pose')

        if self.model is None:
            print(f"[CashDetector] Failed to load pose model")
            return False

        # Load cash object model for Track 3
        self.cash_object_model = self._load_yolo_model(self.CASH_OBJECT_MODEL_PRIORITY)

        self.is_initialized = True
        print(f"[CashDetector] Initialized (mode={self.detection_mode})")
        print(f"[CashDetector] Track 1 & 2: Pose model loaded")

        if self.cash_object_model:
            print(f"[CashDetector] Track 3: Cash object model loaded (conf >= {self.cash_object_confidence})")
        else:
            print(f"[CashDetector] Track 3: Cash object model NOT found (disabled)")

        if self.cashier_zone:
            print(f"[CashDetector] Cashier zone: {len(self.cashier_zone)} points")
        if self.cash_drawer_zone:
            print(f"[CashDetector] Cash drawer zone: {len(self.cash_drawer_zone)} points (Track 2 enabled)")

        return True

    def set_zones(self, cashier_zone: List = None, cash_drawer_zone: List = None):
        """Update zone configurations"""
        if cashier_zone is not None:
            self.cashier_zone = cashier_zone
        if cash_drawer_zone is not None:
            self.cash_drawer_zone = cash_drawer_zone

    def is_in_cashier_zone(self, point: Tuple[int, int]) -> bool:
        """Check if point is in cashier zone"""
        return self.point_in_polygon(point, self.cashier_zone) if self.cashier_zone else False

    def is_in_cash_drawer_zone(self, point: Tuple[int, int]) -> bool:
        """Check if point is in cash drawer zone"""
        return self.point_in_polygon(point, self.cash_drawer_zone) if self.cash_drawer_zone else False

    def _reset_tracking(self, reason: str = ""):
        """Reset Track 2 tracking state"""
        if reason and self.enable_logs:
            print(f"  [Track2] Reset: {reason}")
        self.pending_transaction = None
        self.tracking_cashier_hands = False
        self.frames_since_touch = 0

    def _log(self, message: str):
        """Log message if logging enabled"""
        if self.enable_logs:
            print(message)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run cash transaction detection (3-Track parallel system)

        Returns list of Detection objects with event_type:
        - 'potential_cash': Track 1 detection (hand touch only)
        - 'cash': Track 2 detection (hand touch + drawer deposit)
        - 'cash_object': Track 3 detection (YOLO object detection)
        """
        detections = []

        if not self.is_initialized:
            return detections

        self.frame_count += 1
        h, w = frame.shape[:2]

        # ==================== TRACK 3: CASH OBJECT DETECTION ====================
        # Run independently (parallel to Track 1 & 2)
        track3_detection = self._check_track3_cash_object(frame)
        if track3_detection:
            detections.append(track3_detection)

        # ==================== TRACK 1 & 2: POSE-BASED DETECTION ====================
        # Run pose estimation
        results = self.model(frame, verbose=False, conf=self.pose_confidence)

        if not results or len(results) == 0:
            self._handle_no_detection()
            return detections

        result = results[0]

        if result.keypoints is None or result.boxes is None:
            return detections

        # Extract people
        people = self.extract_people(result, self.cashier_zone)

        if len(people) < 2:
            self._handle_insufficient_people()
            return detections

        # Separate by role
        cashiers = [p for p in people if p.get('in_cashier_zone') is True]
        customers = [p for p in people if p.get('in_cashier_zone') is False]

        # ==================== TRACK 2: CHECK DRAWER DEPOSIT ====================
        track2_detection = self._check_track2_deposit(frame, cashiers, customers)
        if track2_detection:
            detections.append(track2_detection)
            return detections  # Return early if Track 2 detection made

        # ==================== DETECT HAND TOUCH ====================
        touch_events = self._detect_hand_touch(people, cashiers, customers)

        if touch_events:
            best_event = min(touch_events, key=lambda x: x['distance'])

            # Track 1: potential_cash
            track1_detection = self._handle_track1(frame, best_event)
            if track1_detection:
                detections.append(track1_detection)

            # Track 2: Start tracking
            self._start_track2(best_event)

        return detections

    def _check_track3_cash_object(self, frame: np.ndarray) -> Optional[Detection]:
        """
        Track 3: Detect cash using YOLO object detection model

        Returns Detection if cash detected with confidence >= 0.7, None otherwise
        """
        if self.cash_object_model is None:
            self.last_debug_detections = []
            return None

        try:
            # Run object detection
            debug_conf = min(self.cash_object_debug_confidence, self.cash_object_confidence)
            results = self.cash_object_model(frame, verbose=False, conf=debug_conf)

            if not results or len(results) == 0:
                self.last_debug_detections = []
                return None

            result = results[0]

            if result.boxes is None or len(result.boxes) == 0:
                self.last_debug_detections = []
                return None

            h, w = frame.shape[:2]

            # Find the best detection (highest confidence)
            best_box = None
            best_conf = 0
            debug_detections = []

            for box in result.boxes:
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                det_bbox = (
                    max(0, int(xyxy[0])),
                    max(0, int(xyxy[1])),
                    min(w, int(xyxy[2])),
                    min(h, int(xyxy[3]))
                )
                debug_detections.append({
                    'confidence': conf,
                    'bbox': det_bbox,
                    'class_id': cls
                })
                if conf >= self.cash_object_confidence and conf > best_conf:
                    best_box = box
                    best_conf = conf

            self.last_debug_detections = debug_detections

            # Check cooldown for confirmed detection
            frames_since_last = self.frame_count - self.last_cash_object_frame
            if frames_since_last < self.cash_object_cooldown:
                return None

            if best_box is None:
                return None

            # Extract bbox
            xyxy = best_box.xyxy[0].cpu().numpy()
            cls = int(best_box.cls[0])

            det_bbox = (
                max(0, int(xyxy[0])),
                max(0, int(xyxy[1])),
                min(w, int(xyxy[2])),
                min(h, int(xyxy[3]))
            )

            self._log(f"\n  [Track3] CASH_OBJECT DETECTED!")
            self._log(f"    - Confidence: {best_conf:.2f}")
            self._log(f"    - Class ID: {cls}")
            self._log(f"    - BBox: {det_bbox}")

            detection = Detection(
                label="CASH",
                confidence=best_conf,
                bbox=det_bbox,
                metadata={
                    'track': 'track3_cash_object',
                    'class_id': cls,
                    'model': 'yolo26s'
                }
            )
            detection.event_type = 'cash_object'

            self.last_cash_object_frame = self.frame_count

            return detection

        except Exception as e:
            if self.enable_logs:
                print(f"[CashDetector] Track 3 error: {e}")
            self.last_debug_detections = []
            return None

    def get_debug_detections(self) -> List[Dict]:
        """Return last low-confidence cash object detections for overlay."""
        return self.last_debug_detections

    def _handle_no_detection(self):
        """Handle frame with no pose detected"""
        if self.tracking_cashier_hands:
            self.frames_since_touch += 1
            if self.frames_since_touch > self.hand_tracking_duration:
                self._reset_tracking("Timeout - no people detected")

    def _handle_insufficient_people(self):
        """Handle frame with less than 2 people"""
        if self.tracking_cashier_hands:
            self.frames_since_touch += 1
            if self.frames_since_touch > self.hand_tracking_duration:
                self._reset_tracking("Timeout - less than 2 people")

    def _check_track2_deposit(self, frame: np.ndarray,
                               cashiers: List[Dict],
                               customers: List[Dict]) -> Optional[Detection]:
        """
        Track 2: Check if cashier's hand enters cash drawer

        Returns Detection if confirmed, None otherwise
        """
        if not self.tracking_cashier_hands:
            return None
        if not self.pending_transaction:
            return None
        if not self.cash_drawer_zone:
            return None

        self.frames_since_touch += 1
        h, w = frame.shape[:2]

        # Verify we still have cashier + customer
        if not cashiers:
            self._reset_tracking("No cashier detected")
            return None
        if not customers:
            self._reset_tracking("No customer detected")
            return None
        if self.frames_since_touch > self.hand_tracking_duration:
            self._reset_tracking("Timeout - no drawer deposit")
            return None

        # Check if any cashier's hand is in the cash drawer zone
        for cashier in cashiers:
            for hand_name, hand_pos in cashier.get('hands', {}).items():
                if self.is_in_cash_drawer_zone((hand_pos[0], hand_pos[1])):
                    # SUCCESS! Cashier deposited in drawer
                    touch_dist = self.pending_transaction.get('distance', 0)

                    self._log(f"\n  [Track2] CASH CONFIRMED!")
                    self._log(f"    - Touch distance: {touch_dist:.1f}px")
                    self._log(f"    - Frames to deposit: {self.frames_since_touch}")
                    self._log(f"    - Cashier hand: {hand_name}")

                    # Create detection bbox around hand
                    det_bbox = (
                        max(0, hand_pos[0] - 80),
                        max(0, hand_pos[1] - 80),
                        min(w, hand_pos[0] + 80),
                        min(h, hand_pos[1] + 80)
                    )

                    # Calculate confidence
                    distance_score = max(0, 1 - touch_dist / self.hand_touch_distance)
                    time_score = max(0, 1 - self.frames_since_touch / self.hand_tracking_duration)
                    confidence = distance_score * 0.6 + time_score * 0.4
                    confidence = max(0.5, min(1.0, confidence))

                    detection = Detection(
                        label="CASH",
                        confidence=confidence,
                        bbox=det_bbox,
                        metadata={
                            'track': 'track2_drawer_deposit',
                            'touch_distance': round(touch_dist, 1),
                            'frames_to_drawer': self.frames_since_touch,
                            'cashier_hand': hand_name,
                            'mode': 'strict'
                        }
                    )
                    detection.event_type = 'cash'

                    self.last_transaction_frame = self.frame_count
                    self._reset_tracking("Detection complete")

                    return detection

        return None

    def _detect_hand_touch(self, people: List[Dict],
                           cashiers: List[Dict],
                           customers: List[Dict]) -> List[Dict]:
        """
        Detect hand touch events between people

        Returns list of touch events with distance, midpoint, etc.
        """
        touch_events = []

        # Determine pairs to check based on mode
        if self.detection_mode == 'strict':
            if not cashiers or not customers:
                return touch_events
            pairs = [(c, cust) for c in cashiers for cust in customers]
        else:
            # Simple mode: check all pairs
            pairs = [
                (people[i], people[j])
                for i in range(len(people))
                for j in range(i + 1, len(people))
            ]

        # Check hand proximity for each pair
        for p1, p2 in pairs:
            for h1_name, h1_pos in p1.get('hands', {}).items():
                for h2_name, h2_pos in p2.get('hands', {}).items():
                    distance = self.calculate_distance(h1_pos[:2], h2_pos[:2])

                    if distance < self.hand_touch_distance:
                        midpoint = (
                            (h1_pos[0] + h2_pos[0]) // 2,
                            (h1_pos[1] + h2_pos[1]) // 2
                        )

                        touch_events.append({
                            'distance': distance,
                            'midpoint': midpoint,
                            'person1': p1,
                            'person2': p2,
                            'h1_name': h1_name,
                            'h2_name': h2_name
                        })

        return touch_events

    def _handle_track1(self, frame: np.ndarray, event: Dict) -> Optional[Detection]:
        """
        Track 1: Create potential_cash detection on hand touch

        Returns Detection if cooldown passed, None otherwise
        """
        frames_since_last = self.frame_count - self.last_potential_cash_frame

        if frames_since_last < self.potential_cash_cooldown:
            return None

        h, w = frame.shape[:2]
        midpoint = event['midpoint']
        p1, p2 = event['person1'], event['person2']

        det_bbox = (
            max(0, midpoint[0] - 100),
            max(0, midpoint[1] - 100),
            min(w, midpoint[0] + 100),
            min(h, midpoint[1] + 100)
        )

        confidence = max(0.5, 1 - event['distance'] / self.hand_touch_distance) * 0.7

        role_info = f"{p1['role']}(P{p1['idx']+1}) <-> {p2['role']}(P{p2['idx']+1})"
        self._log(f"\n  [Track1-{self.detection_mode.upper()}] POTENTIAL_CASH: {role_info}, dist={event['distance']:.0f}px")

        detection = Detection(
            label="CASH",
            confidence=confidence,
            bbox=det_bbox,
            metadata={
                'track': 'track1_hand_touch',
                'distance': round(event['distance'], 1),
                'person1_idx': p1['idx'],
                'person2_idx': p2['idx'],
                'person1_role': p1['role'],
                'person2_role': p2['role'],
                'midpoint': midpoint,
                'mode': self.detection_mode
            }
        )
        detection.event_type = 'potential_cash'

        self.last_potential_cash_frame = self.frame_count

        return detection

    def _start_track2(self, event: Dict):
        """
        Track 2: Start tracking cashier's hand after touch

        Only starts if:
        - In strict mode
        - Cash drawer zone defined
        - Not already tracking
        - Cooldown passed
        """
        if self.detection_mode != 'strict':
            return
        if not self.cash_drawer_zone:
            return
        if self.tracking_cashier_hands:
            return
        if self.frame_count - self.last_transaction_frame <= self.transaction_cooldown:
            return

        self.pending_transaction = event
        self.tracking_cashier_hands = True
        self.frames_since_touch = 0
        self.touch_frame = self.frame_count

        self._log(f"  [Track2] Started tracking for drawer deposit...")
        self._log(f"    - Will track for {self.hand_tracking_duration} frames (~{self.hand_tracking_duration/30:.1f}s)")

    def get_tracking_status(self) -> Dict:
        """Get current tracking status for UI display"""
        return {
            'tracking': self.tracking_cashier_hands,
            'frames_since_touch': self.frames_since_touch,
            'max_frames': self.hand_tracking_duration,
            'progress': self.frames_since_touch / self.hand_tracking_duration if self.hand_tracking_duration > 0 else 0,
            'track3_enabled': self.cash_object_model is not None,
            'track3_cooldown_remaining': max(0, self.cash_object_cooldown - (self.frame_count - self.last_cash_object_frame))
        }
