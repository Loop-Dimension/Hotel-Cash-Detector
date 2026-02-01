"""
API Routes for ML Detection Service
"""
import time
import base64
import tempfile
import os
from typing import List
from datetime import datetime

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from app.config import settings, get_gpu_info
from app.models.schemas import (
    DetectorConfig,
    DetectionRequest,
    DetectionResponse,
    BatchDetectionRequest,
    BatchDetectionResponse,
    ValidationRequest,
    VideoValidationRequest,
    ValidationResponse,
    ServiceStatus,
    DetectorStatus,
    DetectionResult,
    BoundingBox,
    GPUInfo,
    MessageResponse,
    ZoneUpdate,
)
from app.services.detector_manager import detector_manager

router = APIRouter()


# ==================== Utility Functions ====================

def decode_frame(image_base64: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        image_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image")
        return frame
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")


def encode_frame(frame: np.ndarray, quality: int = 85) -> str:
    """Encode numpy array to base64 JPEG"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def convert_detection(det: dict, frame_number: int) -> DetectionResult:
    """Convert detector output to response schema"""
    bbox = None
    if det.get('bbox'):
        b = det['bbox']
        if isinstance(b, (list, tuple)) and len(b) == 4:
            bbox = BoundingBox(x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]))

    return DetectionResult(
        label=det.get('label', 'UNKNOWN'),
        confidence=float(det.get('confidence', 0.0)),
        bbox=bbox,
        timestamp=det.get('timestamp', datetime.now().isoformat()),
        frame_number=det.get('frame_number', frame_number),
        metadata=det.get('metadata', {}),
    )


# ==================== Detector Management ====================

@router.post("/detectors/{camera_id}/initialize", response_model=MessageResponse)
async def initialize_detector(camera_id: int, config: DetectorConfig):
    """Initialize or update detector for a camera"""
    config.camera_id = camera_id

    try:
        detector = detector_manager.get_or_create_detector(config)
        return MessageResponse(
            message="Detector initialized successfully",
            camera_id=camera_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize detector: {str(e)}")


@router.delete("/detectors/{camera_id}", response_model=MessageResponse)
async def remove_detector(camera_id: int):
    """Remove detector for camera"""
    if detector_manager.remove_detector(camera_id):
        return MessageResponse(
            message=f"Detector for camera {camera_id} removed",
            camera_id=camera_id,
        )
    raise HTTPException(status_code=404, detail=f"Detector for camera {camera_id} not found")


@router.get("/detectors")
async def list_detectors():
    """List all active detectors"""
    return detector_manager.list_detectors()


@router.get("/detectors/{camera_id}/status", response_model=DetectorStatus)
async def get_detector_status(camera_id: int):
    """Get status of a specific detector"""
    status = detector_manager.get_detector_status(camera_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Detector for camera {camera_id} not found")
    return status


@router.get("/status", response_model=ServiceStatus)
async def get_service_status():
    """Get overall service status with GPU info"""
    status = detector_manager.get_status()
    detectors = detector_manager.list_detectors()

    detector_statuses = []
    for camera_id in detectors.keys():
        det_status = detector_manager.get_detector_status(camera_id)
        if det_status:
            detector_statuses.append(det_status)

    gpu = status.get('gpu_info', {})

    return ServiceStatus(
        status="healthy",
        uptime_seconds=status.get('uptime_seconds', 0),
        gpu_info=GPUInfo(
            cuda_available=gpu.get('cuda_available', False),
            cuda_version=gpu.get('cuda_version'),
            gpu_name=gpu.get('gpu_name'),
            gpu_count=gpu.get('gpu_count', 0),
            memory_used_mb=gpu.get('memory_used_mb'),
            memory_total_mb=gpu.get('memory_total_mb'),
        ),
        active_detectors=status.get('active_detectors', 0),
        detectors=detector_statuses,
        models_loaded={},
    )


# ==================== Frame Processing ====================

@router.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    """Process a single frame for detection"""
    start_time = time.time()

    # Get or create detector
    detector = detector_manager.get_detector(request.camera_id)
    if not detector:
        # Auto-create with default or provided config
        config = request.config or DetectorConfig(camera_id=request.camera_id)
        config.camera_id = request.camera_id
        detector = detector_manager.get_or_create_detector(config)

    try:
        # Decode frame
        frame = decode_frame(request.frame.image_base64)

        # Process frame
        result = detector.process_frame(frame, draw_overlay=request.draw_overlay)

        # Convert detections
        detections = [convert_detection(d, result.get('frame_number', 0))
                      for d in result.get('detections', [])]
        alerts = [convert_detection(a, result.get('frame_number', 0))
                  for a in result.get('alerts', [])]

        # Record detection if any
        if detections:
            detector_manager.record_detection(request.camera_id)

        # Encode processed frame if requested
        processed_frame = None
        if request.return_frame and result.get('frame') is not None:
            processed_frame = encode_frame(result['frame'])

        processing_time = int((time.time() - start_time) * 1000)

        return DetectionResponse(
            camera_id=request.camera_id,
            frame_number=result.get('frame_number', 0),
            detections=detections,
            alerts=alerts,
            processed_frame_base64=processed_frame,
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@router.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(request: BatchDetectionRequest):
    """Process multiple frames in batch"""
    start_time = time.time()

    results = []
    for frame_data in request.frames:
        single_request = DetectionRequest(
            camera_id=request.camera_id,
            frame=frame_data,
            draw_overlay=request.draw_overlay,
            return_frame=request.return_frames,
        )
        try:
            result = await detect(single_request)
            results.append(result)
        except HTTPException:
            # Continue with other frames even if one fails
            continue

    total_time = int((time.time() - start_time) * 1000)

    return BatchDetectionResponse(
        camera_id=request.camera_id,
        results=results,
        total_processing_time_ms=total_time,
    )


# ==================== Gemini Validation ====================

@router.post("/validate", response_model=ValidationResponse)
async def validate_detection(request: ValidationRequest):
    """Validate a detection using Gemini AI"""
    start_time = time.time()

    try:
        from detectors.gemini_validator import GeminiValidator

        # Decode frame
        frame = decode_frame(request.frame.image_base64)

        # Create validator
        validator = GeminiValidator(
            api_key=settings.GEMINI_API_KEY,
            camera_id=request.camera_id,
        )

        if request.custom_prompt:
            validator.set_custom_prompts({'unified': request.custom_prompt})

        # Validate
        is_valid, confidence, reason, corrected_type = validator.validate_event(
            frame, request.event_type
        )

        processing_time = int((time.time() - start_time) * 1000)

        return ValidationResponse(
            is_valid=is_valid,
            confidence=confidence,
            reason=reason,
            corrected_event_type=corrected_type,
            processing_time_ms=processing_time,
        )

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Gemini validator not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@router.post("/validate/video", response_model=ValidationResponse)
async def validate_detection_video(request: VideoValidationRequest):
    """Validate detection using video clip with Gemini AI"""
    start_time = time.time()

    try:
        from detectors.gemini_validator import GeminiValidator

        # Decode video to temp file
        video_data = base64.b64decode(request.video_base64)

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_data)
            video_path = f.name

        try:
            # Create validator
            validator = GeminiValidator(
                api_key=settings.GEMINI_API_KEY,
                camera_id=request.camera_id,
            )

            if request.custom_prompt:
                validator.set_custom_prompts({'unified': request.custom_prompt})

            # Validate with video
            is_valid, confidence, reason, corrected_type = validator.validate_event_video(
                video_path, request.event_type
            )

            processing_time = int((time.time() - start_time) * 1000)

            return ValidationResponse(
                is_valid=is_valid,
                confidence=confidence,
                reason=reason,
                corrected_event_type=corrected_type,
                processing_time_ms=processing_time,
            )

        finally:
            # Cleanup temp file
            if os.path.exists(video_path):
                os.unlink(video_path)

    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Gemini validator not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video validation error: {str(e)}")


# ==================== Configuration Updates ====================

@router.post("/detectors/{camera_id}/zones", response_model=MessageResponse)
async def update_zones(camera_id: int, zones: ZoneUpdate):
    """Update zone polygons for a detector"""
    success = detector_manager.update_zones(
        camera_id,
        cashier_zone=zones.cashier_zone_polygon,
        cash_drawer_zone=zones.cash_drawer_zone_polygon,
    )

    if not success:
        raise HTTPException(status_code=404, detail=f"Detector for camera {camera_id} not found")

    return MessageResponse(
        message="Zones updated successfully",
        camera_id=camera_id,
    )


@router.post("/detectors/{camera_id}/config", response_model=MessageResponse)
async def update_config(camera_id: int, config: DetectorConfig):
    """Update detector configuration"""
    config.camera_id = camera_id

    try:
        detector_manager.get_or_create_detector(config)
        return MessageResponse(
            message="Configuration updated successfully",
            camera_id=camera_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")
