"""
Gemini AI Validation Service for Detection Events

This module provides AI-powered validation of detected events using Google's Gemini API.
It acts as a secondary validation layer to reduce false positives by analyzing images
before events are stored in the database.

Usage:
    validator = GeminiValidator(api_key="your-api-key")
    is_valid, confidence, reason, corrected_event_type = validator.validate_event(frame, "cash")
"""

import cv2
import json
import os
import time
from typing import Tuple, Optional, Dict
from pathlib import Path
from google import genai
from google.genai import types


# ============================================================================
# GLOBAL GEMINI PROMPT - Edit this to change AI validation behavior
# ============================================================================
DEFAULT_UNIFIED_PROMPT = """You are an AI security analyst reviewing CCTV footage. Analyze this image for: {event_type}

YOUR TASK:
1. Look at the image carefully
2. Determine what event (if any) is actually happening
3. Classify into ONE of 3 categories: CASH, VIOLENCE, or FIRE
4. Respond with what you SEE, not what you're told to find

IMPORTANT: The system detects "potential_cash" as early warning, but YOU must classify ANY payment-related activity as "cash".

EVENT TYPE DEFINITIONS:
======================

CASH / PAYMENT TRANSACTIONS (Classify as: "cash")
CRITICAL RULE: Must see CASHIER + CUSTOMER + PAYMENT ITEMS

VALID if ALL conditions met:
   1. Person(s) behind counter (cashier area)
   2. Person(s) in front of counter (customer area)
   3. PAYMENT ITEMS visible: cash bills, coins, credit/debit cards, mobile payment
   4. Transaction happening: hands exchanging PAYMENT items, opening cash drawer
   
   EXAMPLES - ALL VALID:
   - Cashier receiving cash bills or coins from customer
   - Customer handing credit card / debit card / visa / payment card to cashier
   - Cashier opening cash drawer while handling money
   - Coins or bills visible on counter during transaction
   - Customer using mobile payment (phone near terminal)
   - Hand holding cash/card reaching to cashier
   - ANY payment with visible cash, coins, or cards

REJECT immediately if:
   1. Empty counter with no people
   2. Person walking by without stopping/interacting
   3. No PAYMENT ITEMS visible (no cash, cards, coins)
   4. Exchanging NON-PAYMENT items (keys, envelopes, documents, packages, food)
   5. Just talking or standing (no payment visible)
   6. Giving room keys, letters, or other hotel items (NOT payment)

REMEMBER: ONLY classify as "cash" if you see ACTUAL PAYMENT ITEMS (cash, coins, cards)

VIOLENCE/ALTERCATION (event_type = "violence")
VALID if you see:
   - People in fighting poses (fists raised, defensive stance)
   - Physical aggression (punching, pushing, grabbing)
   - Person on ground from being attacked
   - Multiple people surrounding one person aggressively
   - Clear hostile body language

REJECT if:
   - Normal standing/walking
   - Friendly handshake or conversation
   - People just standing close together
   - Normal interaction

FIRE/SMOKE (event_type = "fire")
VALID if you see:
   - Visible flames (orange/red/yellow fire)
   - Smoke clouds (white, gray, or black)
   - Unusual bright lighting from fire
   - Objects actively burning

REJECT if:
   - Normal lighting or sunset colors
   - Red/orange objects (not fire)
   - Steam from cooking
   - Screen reflections

RESPONSE FORMAT (JSON ONLY):
{
    "is_valid": true/false,
    "event_type_detected": "cash" | "violence" | "fire" | "none",
    "confidence": 0.0-1.0,
    "reason": "brief 1-sentence explanation"
}

IMPORTANT RULES:
1. is_valid = true ONLY if you SEE the event clearly
2. event_type_detected = MUST be one of: "cash", "violence", "fire", or "none"
3. If YOLO says "potential_cash", classify as "cash" if you see ANY payment activity
4. If YOLO says "violence" but you see payment, set event_type_detected = "cash"
5. If you see NOTHING suspicious, set is_valid = false, event_type_detected = "none"
6. Be LENIENT for payment detection - accept early stage transactions
7. NEVER return "potential_cash" - always use "cash" for any payment activity
"""
# ============================================================================


class GeminiValidator:
    """
    Validates detection events using Google Gemini Vision API.
    
    This acts as a filter layer - only events confirmed by Gemini are stored.
    Supports custom prompts per camera and logging of all validations.
    """
    
    # Gemini API - use gemini-2.5-flash-lite (cheapest, FREE standard tier)
    # Best for: high volume, cost-efficient image validation
    # Pricing: FREE (standard) | $0.10/1M input + $0.40/1M output (paid)
    # https://ai.google.dev/gemini-api/docs/pricing
    MODEL_NAME = "gemini-2.5-flash-lite"
    
    # Legacy prompts (for backward compatibility)
    PROMPTS = {
        'cash': """Analyze this CCTV image from a cash register area. 
Determine if there is a CASH TRANSACTION happening.

Look for these signs of a cash transaction:
1. A cashier behind a counter/register
2. A customer in front of the counter
3. Hands exchanging money, cards, or items
4. Cash register or POS terminal visible
5. Hand reaching into cash drawer

Respond in JSON format ONLY:
{
    "is_cash_transaction": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation",
    "details": {
        "cashier_visible": true/false,
        "customer_visible": true/false,
        "cash_exchange_visible": true/false,
        "register_visible": true/false
    }
}""",
        
        'violence': """Analyze this CCTV image for VIOLENCE or PHYSICAL ALTERCATION.

Look for these signs of violence:
1. People in fighting poses
2. Physical contact between people (punching, pushing, grabbing)
3. Aggressive body language
4. People on the ground from being pushed/hit
5. Multiple people surrounding one person aggressively

Do NOT flag as violence:
- Normal standing or walking
- Friendly interaction or handshakes
- People simply close together

Respond in JSON format ONLY:
{
    "is_violence": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation",
    "details": {
        "fighting_pose": true/false,
        "physical_contact": true/false,
        "aggressive_behavior": true/false,
        "people_count": number
    }
}""",
        
        'fire': """Analyze this CCTV image for FIRE or SMOKE.

Look for these signs of fire:
1. Visible flames (orange/red/yellow)
2. Smoke (white, gray, or black)
3. Unusual lighting that could indicate fire
4. Fire on objects, walls, or floor

Do NOT flag as fire:
- Normal lighting
- Red/orange colored objects
- Steam from cooking
- Sunlight reflections

Respond in JSON format ONLY:
{
    "is_fire": true/false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation",
    "details": {
        "flames_visible": true/false,
        "smoke_visible": true/false,
        "fire_location": "description or null"
    }
}"""
    }
    
    def __init__(self, api_key: str = None, enabled: bool = True, camera_id: int = None):
        """
        Initialize the Gemini validator.
        
        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            enabled: If False, all validations return True (bypass mode).
            camera_id: Camera ID for logging and custom prompts.
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        self.enabled = enabled and bool(self.api_key)
        self.client = None
        self.camera_id = camera_id
        self.custom_prompts = {}  # Custom prompts per event type
        self.last_validation_log = None  # Store last validation for debugging
        
        if self.enabled:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print(f"[GeminiValidator] Initialized with model: {self.MODEL_NAME}")
            except Exception as e:
                print(f"[GeminiValidator] Failed to initialize: {e}")
                self.enabled = False
        
        if not self.api_key and enabled:
            print("[GeminiValidator] Warning: No API key provided, validation disabled")
    
    def set_custom_prompts(self, prompts: Dict[str, str]):
        """Set custom prompts for event types (supports unified prompt)"""
        self.custom_prompts = prompts
    
    def get_prompt(self, event_type: str) -> str:
        """Get prompt for event type - uses unified prompt with {event_type} placeholder"""
        # Check for unified prompt first (stored in 'cash' key for compatibility)
        unified_prompt = self.custom_prompts.get('cash', '')
        
        # If the prompt contains {event_type}, it's a unified prompt
        if unified_prompt and '{event_type}' in unified_prompt:
            return unified_prompt.replace('{event_type}', event_type)
        
        # Legacy: check for specific event type prompt
        if self.custom_prompts.get(event_type):
            return self.custom_prompts.get(event_type)
        
        # Default: use global unified prompt
        return DEFAULT_UNIFIED_PROMPT.replace('{event_type}', event_type)
    
    def _encode_image(self, frame):
        """Convert OpenCV frame to bytes for Gemini API."""
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    
    def _save_validation_image(self, frame, event_type: str) -> Optional[str]:
        """Save validation image for debugging/logging"""
        try:
            from datetime import datetime
            from django.conf import settings
            
            # Create gemini_logs directory
            log_dir = Path(settings.MEDIA_ROOT) / 'gemini_logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{event_type}_{self.camera_id or 'unknown'}_{timestamp}.jpg"
            filepath = log_dir / filename
            
            cv2.imwrite(str(filepath), frame)
            return str(filepath.relative_to(settings.MEDIA_ROOT))
        except Exception as e:
            print(f"[GeminiValidator] Failed to save validation image: {e}")
            return None
    
    def _log_validation(
        self,
        camera_id: int,
        event_type: str,
        is_valid: bool, 
        confidence: float,
        reason: str,
        prompt: str,
        response_raw: str,
        image_path: str,
        processing_time_ms: int
    ):
        """Log validation result to database"""
        print(f"[GeminiValidator] _log_validation called: camera_id={camera_id}, event_type={event_type}")
        try:
            from cctv.models import GeminiLog, Camera
            
            print(f"[GeminiValidator] Attempting to get Camera {camera_id}")
            camera = Camera.objects.get(id=camera_id) if camera_id else None
            if camera:
                print(f"[GeminiValidator] Creating GeminiLog entry...")
                log = GeminiLog.objects.create(
                    camera=camera,
                    event_type=event_type,
                    is_validated=is_valid,
                    confidence=confidence,
                    reason=reason,
                    prompt_used=prompt,
                    response_raw=response_raw,
                    image_path=image_path or '',
                    processing_time_ms=processing_time_ms
                )
                print(
                    f"[GeminiValidator] ✅ Successfully logged validation ID {log.id} "
                    f"for camera {camera_id}, event_type={event_type}, is_validated={is_valid}"
                )
            else:
                print(f"[GeminiValidator] ❌ No camera found with id={camera_id}")
        except Exception as e:
            import traceback
            print(f"[GeminiValidator] ❌ Failed to log validation: {e}")
            traceback.print_exc()
    
    def _call_gemini_api(self, image_bytes: bytes, prompt: str) -> dict:
        """
        Call Gemini API with image and prompt using official SDK.
        
        Returns:
            dict: Parsed JSON response or error dict
        """
        if not self.client:
            return {"error": "Client not initialized"}
        
        try:
            response = self.client.models.generate_content(
                model=self.MODEL_NAME,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type="image/jpeg"
                            )
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_k=1,
                    top_p=1.0,
                    max_output_tokens=500,
                    response_mime_type="application/json"
                )
            )
            
            # Extract text from response
            if response.text:
                text = response.text.strip()
                
                # Parse JSON from response (handle markdown code blocks if any)
                if text.startswith('```json'):
                    text = text[7:]
                if text.startswith('```'):
                    text = text[3:]
                if text.endswith('```'):
                    text = text[:-3]
                text = text.strip()
                
                return json.loads(text)
            else:
                return {"error": "No response from Gemini"}
                
        except json.JSONDecodeError as e:
            print(f"[GeminiValidator] JSON parse error: {e}")
            print(f"[GeminiValidator] Response text: {response.text[:200] if response else 'None'}")
            return {"error": "Invalid JSON response"}
        except Exception as e:
            print(f"[GeminiValidator] API error: {e}")
            return {"error": str(e)}
    
    def validate_event(
        self, frame, event_type: str, save_image: bool = True
    ) -> Tuple[bool, float, str, str]:
        """
        Validate a detection event using Gemini AI.
        
        Args:
            frame: OpenCV frame (numpy array) to analyze
            event_type: Type of event ('cash', 'violence', 'fire')
            save_image: Whether to save the validation image for logging
            
        Returns:
            Tuple of (is_valid, confidence, reason, corrected_event_type)
            - is_valid: True if Gemini confirms the event
            - confidence: Gemini's confidence score (0.0-1.0)
            - reason: Gemini's explanation
            - corrected_event_type: The actual event type Gemini detected (may differ from input)
        """
        start_time = time.time()
        image_path = None
        prompt = ""
        response_raw = ""
        
        # If disabled or no API key, bypass validation
        if not self.enabled:
            return True, 1.0, "Validation bypassed (no API key)", event_type
        
        # Get prompt (custom or default)
        prompt = self.get_prompt(event_type)
        if not prompt:
            print(f"[GeminiValidator] Unknown event type: {event_type}")
            return True, 1.0, f"Unknown event type: {event_type}", event_type
        
        # Check frame validity
        if frame is None or frame.size == 0:
            return False, 0.0, "Invalid frame", event_type
        
        try:
            # Save image for logging if enabled
            if save_image and self.camera_id:
                image_path = self._save_validation_image(frame, event_type)
            
            # Encode image
            image_bytes = self._encode_image(frame)
            
            # Call Gemini API
            result = self._call_gemini_api(image_bytes, prompt)
            response_raw = json.dumps(result)
            
            # Check for errors
            if 'error' in result:
                # On API error, allow the event (don't block on API issues)
                print(f"[GeminiValidator] API error, allowing event: {result['error']}")
                return True, 1.0, f"API error: {result['error']}", event_type
            
            # Get Gemini's validation result
            is_valid = result.get('is_valid', False)
            confidence = result.get('confidence', 0.0)
            reason = result.get('reason', 'No reason provided')
            
            # Check if Gemini detected a DIFFERENT event type (correction)
            detected_type = result.get('event_type_detected', event_type)
            corrected_event_type = event_type  # Default to original
            
            # === 핵심 정책 변경 ===
            if is_valid and detected_type != 'none' and detected_type != event_type:
                # 규칙 1: violence → cash 로 교정되는 경우는 무시 (중복 cash 방지)
                if event_type == "violence" and detected_type == "cash":
                    is_valid = False
                    corrected_event_type = event_type  # 여전히 violence
                    reason = (
                        "Gemini classified this VIOLENCE event as CASH, "
                        "but to avoid duplicate CASH events from nearby frames, "
                        "this VIOLENCE event is ignored. "
                        + reason
                    )
                else:
                    # 그 외 타입 변경은 기존처럼 교정 허용
                    corrected_event_type = detected_type
                    reason = f"Corrected: {event_type.upper()} → {corrected_event_type.upper()}. {reason}"
            elif is_valid and detected_type == event_type:
                # Gemini confirmed the original detection
                pass  # corrected_event_type stays as event_type
            elif not is_valid:
                # Gemini rejected the detection
                # Keep original event_type but mark as invalid
                pass
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Store for debugging
            self.last_validation_log = {
                'event_type': event_type,
                'is_valid': is_valid,
                'confidence': confidence,
                'reason': reason,
                'prompt': prompt,
                'response': result,
                'image_path': image_path,
                'processing_time_ms': processing_time_ms
            }
            
            # Log to database
            print(f"[GeminiValidator] DEBUG: camera_id={self.camera_id}, will_log={bool(self.camera_id)}")
            if self.camera_id:
                self._log_validation(
                    self.camera_id, event_type, is_valid, confidence, 
                    reason, prompt, response_raw, image_path, processing_time_ms
                )
            else:
                print(f"[GeminiValidator] ⚠️ Skipping database log - no camera_id set!")
            
            print(
                f"[GeminiValidator] {event_type}: valid={is_valid}, "
                f"conf={confidence:.2f}, corrected={corrected_event_type}, "
                f"reason={reason[:100]}"
            )
            
            return is_valid, confidence, reason, corrected_event_type
            
        except Exception as e:
            print(f"[GeminiValidator] Exception: {e}")
            # On error, allow the event (don't block on validation errors)
            return True, 1.0, f"Validation error: {e}", event_type
    
    def validate_cash_transaction(self, frame) -> Tuple[bool, float, str]:
        """Convenience method for cash transaction validation."""
        # NOTE: 여기서는 4개를 리턴하지만, 호출 측에서 3개만 받으면 파이썬 언패킹으로 조절 가능
        return self.validate_event(frame, 'cash')
    
    def validate_violence(self, frame) -> Tuple[bool, float, str]:
        """Convenience method for violence validation."""
        return self.validate_event(frame, 'violence')
    
    def validate_fire(self, frame) -> Tuple[bool, float, str]:
        """Convenience method for fire validation."""
        return self.validate_event(frame, 'fire')


# Singleton instance for global access
_validator_instance = None


def get_validator(api_key: str = None) -> GeminiValidator:
    """
    Get or create the global GeminiValidator instance.
    
    Args:
        api_key: Optional API key (uses env var if not provided)
        
    Returns:
        GeminiValidator instance
    """
    global _validator_instance
    
    if _validator_instance is None:
        from django.conf import settings
        key = api_key or getattr(settings, 'GEMINI_API_KEY', None) or os.environ.get('GEMINI_API_KEY', '')
        _validator_instance = GeminiValidator(api_key=key)
    
    return _validator_instance


def validate_detection(frame, event_type: str, api_key: str = None) -> Tuple[bool, float, str, str]:
    """
    Convenience function to validate a detection event.
    
    Args:
        frame: OpenCV frame to analyze
        event_type: Type of event ('cash', 'violence', 'fire')
        api_key: Optional API key
        
    Returns:
        Tuple of (is_valid, confidence, reason, corrected_event_type)
    """
    validator = get_validator(api_key)
    return validator.validate_event(frame, event_type)
