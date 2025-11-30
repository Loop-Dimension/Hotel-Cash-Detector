"""
Base Detector class for all detection modules
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from datetime import datetime


@dataclass
class Detection:
    """Represents a single detection result"""
    label: str  # CASH, VIOLENCE, FIRE
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    timestamp: datetime = None
    frame_number: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "timestamp": self.timestamp.isoformat(),
            "frame_number": self.frame_number,
            "metadata": self.metadata
        }


class BaseDetector(ABC):
    """Abstract base class for all detectors"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.is_initialized = False
        self.frame_count = 0
        self.detection_history: List[Detection] = []
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the detector (load models, etc.)"""
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform detection on a single frame
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of Detection objects
        """
        pass
    
    def process_frame(self, frame: np.ndarray) -> List[Detection]:
        """Process a frame and update history"""
        self.frame_count += 1
        
        if not self.is_initialized:
            if not self.initialize():
                return []
        
        detections = self.detect(frame)
        
        for det in detections:
            det.frame_number = self.frame_count
            self.detection_history.append(det)
        
        # Keep only recent history (last 1000 detections)
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-500:]
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detections on frame"""
        import cv2
        
        for det in detections:
            if det.bbox:
                x1, y1, x2, y2 = det.bbox
                
                # Get color based on label
                colors = {
                    "CASH": (0, 255, 0),      # Green
                    "VIOLENCE": (0, 0, 255),   # Red
                    "FIRE": (0, 165, 255)      # Orange
                }
                color = colors.get(det.label, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{det.label}: {det.confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def reset(self):
        """Reset the detector state"""
        self.frame_count = 0
        self.detection_history = []
