"""
Pydantic models for ML Service API
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ==================== Detection Configuration ====================

class DetectorConfig(BaseModel):
    """Configuration for a camera detector"""
    camera_id: int

    # Detection toggles
    detect_cash: bool = True
    detect_violence: bool = True
    detect_fire: bool = True

    # Confidence thresholds
    cash_confidence: float = 0.5
    violence_confidence: float = 0.6
    fire_confidence: float = 0.5
    pose_confidence: float = 0.5

    # Zone configurations
    cashier_zone_polygon: Optional[List[List[int]]] = None
    cash_drawer_zone_polygon: Optional[List[List[int]]] = None
    cashier_zone_enabled: bool = True
    cash_drawer_zone_enabled: bool = True

    # Detection parameters
    hand_touch_distance: int = 100
    min_transaction_frames: int = 1
    min_violence_frames: int = 10
    min_fire_frames: int = 3
    min_fire_area: int = 2000

    # GPU settings
    use_gpu: str = "auto"


class ZoneUpdate(BaseModel):
    """Zone polygon update"""
    cashier_zone_polygon: Optional[List[List[int]]] = None
    cash_drawer_zone_polygon: Optional[List[List[int]]] = None


# ==================== Frame Data ====================

class FrameData(BaseModel):
    """Single frame for processing"""
    image_base64: str  # Base64 encoded JPEG/PNG
    frame_number: Optional[int] = None
    timestamp: Optional[str] = None


# ==================== Detection Requests ====================

class DetectionRequest(BaseModel):
    """Request to process a single frame"""
    camera_id: int
    frame: FrameData
    draw_overlay: bool = True
    return_frame: bool = True
    config: Optional[DetectorConfig] = None  # Optional inline config


class BatchDetectionRequest(BaseModel):
    """Request to process multiple frames"""
    camera_id: int
    frames: List[FrameData]
    draw_overlay: bool = False
    return_frames: bool = False


# ==================== Detection Responses ====================

class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionResult(BaseModel):
    """Single detection result"""
    label: str  # CASH, VIOLENCE, FIRE
    confidence: float
    bbox: Optional[BoundingBox] = None
    timestamp: str
    frame_number: int
    metadata: Dict[str, Any] = {}


class DetectionResponse(BaseModel):
    """Response from frame processing"""
    camera_id: int
    frame_number: int
    detections: List[DetectionResult]
    alerts: List[DetectionResult]
    processed_frame_base64: Optional[str] = None
    processing_time_ms: int


class BatchDetectionResponse(BaseModel):
    """Response from batch processing"""
    camera_id: int
    results: List[DetectionResponse]
    total_processing_time_ms: int


# ==================== Gemini Validation ====================

class ValidationRequest(BaseModel):
    """Request to validate a detection with image"""
    camera_id: int
    event_type: str  # cash, violence, fire
    frame: FrameData
    custom_prompt: Optional[str] = None


class VideoValidationRequest(BaseModel):
    """Request to validate with video clip"""
    camera_id: int
    event_type: str
    video_base64: str  # Base64 encoded MP4
    custom_prompt: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response from Gemini validation"""
    is_valid: bool
    confidence: float
    reason: str
    corrected_event_type: str
    processing_time_ms: int


# ==================== Status & Health ====================

class GPUInfo(BaseModel):
    """GPU information"""
    cuda_available: bool
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None


class DetectorStatus(BaseModel):
    """Status of a single detector"""
    camera_id: int
    is_initialized: bool
    frame_count: int
    detection_types: List[str]
    last_detection_time: Optional[str] = None


class ServiceStatus(BaseModel):
    """Overall service status"""
    status: str  # healthy, degraded, unhealthy
    uptime_seconds: float
    gpu_info: GPUInfo
    active_detectors: int
    detectors: List[DetectorStatus]
    models_loaded: Dict[str, bool]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    service: str
    version: str


# ==================== Generic Responses ====================

class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    camera_id: Optional[int] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
