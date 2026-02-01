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
# GLOBAL UNIFIED PROMPT (SOFT SCORING: CASH != CARD)
# ============================================================================
DEFAULT_UNIFIED_PROMPT = r"""
You are an AI Retail Security & Safety Analyst.

You will receive:
1) A CCTV image OR a short CCTV video clip (5–10 seconds)
2) An upstream detection_type string: "{event_type}"
3) Optional detection metadata (YOLO/camera context)

Your job is to visually analyze ONLY what is visible in the clip and choose ONE policy:
- CASH_TRANSACTION
- THREAT_TO_CASHIER
- FIRE_ALERT
- STAFF_CASH_THEFT_SUSPECT
- NONE

=====================================================================
GLOBAL OUTPUT FORMAT (MANDATORY)
=====================================================================

Return ONLY one JSON object with this exact structure:

{{
  "event_policy": "CASH_TRANSACTION | THREAT_TO_CASHIER | FIRE_ALERT | STAFF_CASH_THEFT_SUSPECT | NONE",
  "event_type_detected": "cash | violence | fire | staff_cash_theft | none",
  "is_valid_event": true | false,
  "decision": "TRUE_POSITIVE | FALSE_POSITIVE | NOT_APPLICABLE",
  "severity_label": "none | low | medium | high | critical",
  "confidence": 0.0-1.0,
  "policy_scores": {{}},
  "reason_bullets": [
    "- Factual bullet point",
    "- Each bullet must describe a concrete visual fact",
    "- Do not speculate beyond what is visible"
  ]
}}

Rules:
- Output JSON ONLY (no extra text).
- Fill ALL top-level fields.
- reason_bullets must be a list of strings starting with "- ".


=====================================================================
POLICY 1) CASH_TRANSACTION  (CASH/COINS only; cards should score low)
=====================================================================

Goal:
Detect REAL CASH payment between customer and cashier (banknotes or coins).
Cards/phones/QR/receipts are NOT cash.

MANDATORY CONTEXT (for cash transaction consideration):
- A person behind the counter (cashier area)
- A person in front of counter (customer area) or clearly acting as customer
- Interaction around the payment area

SOFT SCORING (IMPORTANT):
Even if there is a payment interaction, classify as CASH_TRANSACTION only if
the evidence of CASH/COINS is strong enough to pass the score threshold.
If the exchanged object is card-like or unclear, reduce points so it likely fails threshold.

policy_scores for CASH_TRANSACTION MUST be:

{{
  "money_likelihood": 0-40,
  "hand_to_hand": 0-40,
  "safe_drawer": 0-40,
  "counting_wallet": 0-10,
  "gaze_context": 0-10,
  "card_like_penalty": -40-0,
  "total_score": 0-100
}}

Scoring guidance:
- money_likelihood (0–40):
  - 35–40: clearly visible cash bills/coins (shape, texture, coin shine, banknote edges)
  - 20–34: partially occluded or brief but still plausible cash/coins
  - 5–19: ambiguous rectangular item (could be card or folded bill)
  - 0–4: clearly NOT cash (card/phone/receipt/paper)

- hand_to_hand (0–40):
  - 30–40: hand-to-hand exchange with visible cash/coins
  - 10–25: exchange motion occurs, but cash/coins not clearly visible
  - 0–5: exchanged object looks card-like/phone-like/receipt-like

- counting_wallet (0–10):
  - 8–10: visible cash counting or visible bills pulled out
  - 1–4: wallet motion but cash not visible
  - 0: no wallet/cash cues

- safe_drawer (0–40):
  - 30–40: clear register/cash storage interaction consistent with payment
  - 10–25: hand moves toward register area but drawer not visible
  - 0–5: no register-area interaction

- gaze_context (0–10):
  - 8–10: both focus on payment/POS area with transaction posture
  - 1–4: weak context
  - 0: unrelated

- card_like_penalty (-40 to 0):
  Apply negative points if evidence suggests CARD/PHONE/QR/RECEIPT:
  - -30 to -40: clearly card-like object / tapping / phone / QR visible
  - -10 to -25: strongly rigid rectangular object but not fully clear
  - 0: no card-like evidence

Total score:
- total_score = money_likelihood + hand_to_hand + safe_drawer + counting_wallet + gaze_context + card_like_penalty
- Clamp total_score to 0–100

Decision:
- If total_score >= 60 and cash/coin evidence is consistent -> CASH_TRANSACTION (severity_label="low")
- Otherwise -> NONE (even if it looks like a payment but appears to be card/phone)

=====================================================================
POLICY 2) THREAT_TO_CASHIER
=====================================================================

Goal: threats/violence toward cashier/staff.

Valid cues:
- aggressive reach across counter
- punching/pushing/grabbing/throwing objects
- weapon visible

policy_scores:
{{
  "mandatory_score": 0,
  "supporting_score": 0,
  "negative_score": 0,
  "total_score": 0,
  "threat_level": 0-4,
  "threat_label": "CLEAR | TENSE | INTIMIDATION | PHYSICAL | WEAPON"
}}

Severity mapping:
CLEAR->none, TENSE->low, INTIMIDATION->medium, PHYSICAL->high, WEAPON->critical

=====================================================================
POLICY 3) FIRE_ALERT
=====================================================================

Valid if you SEE:
- flames or smoke (not reflections/steam)

policy_scores:
{{
  "fire_confidence": 0.0-1.0,
  "smoke_confidence": 0.0-1.0
}}

Severity guideline:
none/low/medium/high/critical based on visible scale and risk.

=====================================================================
POLICY 4) STAFF_CASH_THEFT_SUSPECT
=====================================================================

Goal: suspicious cash removal by staff without valid customer transaction.

If metadata has has_cash_box_roi=true and cash_box_bboxes exist:
- cash_box access can be used as strong hint.
Otherwise use behavior:
- cash-like object appears in staff hand
- moved toward personal area (pocket/bag/inside clothes)
- nervous look-around / hiding

policy_scores:
{{
  "suspicion_level": 0-3,
  "suspicion_label": "none | low | medium | high",
  "cash_box_access": true/false,
  "looks_around": true/false,
  "moves_cash_to_personal_area": true/false,
  "customer_present": true/false,
  "paperwork_or_reconciliation": true/false
}}

Severity guideline:
none/low/medium/high (critical only if extremely obvious and severe)

FINAL POLICY PRIORITY:
1) FIRE_ALERT
2) THREAT_TO_CASHIER
3) CASH_TRANSACTION
4) STAFF_CASH_THEFT_SUSPECT
5) NONE

Always justify using reason_bullets with factual observations.
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
    MODEL_NAME = "gemini-2.5-flash"
    
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
        """Save validation image for debugging/logging.

        NOTE: In ML service mode, image saving is handled by the Django backend.
        This method is a no-op in the microservice context.
        """
        # In ML service, we don't save images locally
        # The Django backend handles storage after receiving the response
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
        processing_time_ms: int,
        validation_type: str = 'image',
        video_path: str = None
    ):
        """Log validation result.

        NOTE: In ML service mode, database logging is handled by the Django backend.
        This method only prints to console for debugging.
        """
        print(
            f"[GeminiValidator] Validation complete: camera_id={camera_id}, "
            f"event_type={event_type}, is_valid={is_valid}, "
            f"confidence={confidence:.2f}, type={validation_type}"
        )
    
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
                    max_output_tokens=1500,
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
                
                result = json.loads(text)
                print(f"[GeminiValidator] API response: {json.dumps(result, indent=2)[:500]}")
                return result
            else:
                return {"error": "No response from Gemini"}
                
        except json.JSONDecodeError as e:
            print(f"[GeminiValidator] JSON parse error: {e}")
            print(f"[GeminiValidator] Response text: {response.text[:500] if response else 'None'}")
            return {"error": "Invalid JSON response"}
        except Exception as e:
            print(f"[GeminiValidator] API error: {e}")
            return {"error": str(e)}
    
    def _parse_new_response_format(self, result: dict, original_event_type: str) -> Tuple[bool, float, str, str]:
        """
        Parse the new Gemini response format (soft scoring with policies).
        
        New format fields:
        - event_policy: CASH_TRANSACTION | THREAT_TO_CASHIER | FIRE_ALERT | STAFF_CASH_THEFT_SUSPECT | NONE
        - event_type_detected: cash | violence | fire | staff_cash_theft | none
        - is_valid_event: true/false
        - decision: TRUE_POSITIVE | FALSE_POSITIVE | NOT_APPLICABLE
        - severity_label: none | low | medium | high | critical
        - confidence: 0.0-1.0
        - policy_scores: dict with scoring details
        - reason_bullets: list of strings
        
        Returns:
            Tuple of (is_valid, confidence, reason, corrected_event_type)
        """
        # Check for new format fields
        event_policy = result.get('event_policy', '')
        is_valid_event = result.get('is_valid_event')
        event_type_detected = result.get('event_type_detected', 'none')
        confidence = result.get('confidence', 0.0)
        decision = result.get('decision', '')
        severity_label = result.get('severity_label', 'none')
        policy_scores = result.get('policy_scores', {})
        reason_bullets = result.get('reason_bullets', [])
        
        # Also support legacy format for backwards compatibility
        legacy_is_valid = result.get('is_valid')
        legacy_reason = result.get('reason', '')
        
        # Determine is_valid - prefer new format
        if is_valid_event is not None:
            is_valid = is_valid_event
        elif legacy_is_valid is not None:
            is_valid = legacy_is_valid
        else:
            # Infer from event_policy
            is_valid = event_policy not in ['NONE', '', None]
        
        # Build reason string from bullets or use legacy
        if reason_bullets and isinstance(reason_bullets, list):
            reason = ' '.join([b.strip() for b in reason_bullets])
        elif legacy_reason:
            reason = legacy_reason
        else:
            reason = f"Policy: {event_policy}, Decision: {decision}, Severity: {severity_label}"
        
        # Add policy scores to reason if available
        if policy_scores and isinstance(policy_scores, dict):
            total_score = policy_scores.get('total_score', 'N/A')
            reason = f"[Score: {total_score}] {reason}"
        
        # Map event_policy to event_type_detected if not set
        if not event_type_detected or event_type_detected == 'none':
            policy_to_type = {
                'CASH_TRANSACTION': 'cash',
                'THREAT_TO_CASHIER': 'violence',
                'FIRE_ALERT': 'fire',
                'STAFF_CASH_THEFT_SUSPECT': 'staff_cash_theft',
                'NONE': 'none'
            }
            event_type_detected = policy_to_type.get(event_policy, 'none')
        
        # Determine corrected event type
        corrected_event_type = original_event_type
        
        if is_valid and event_type_detected != 'none' and event_type_detected != original_event_type:
            # Handle violence -> cash correction (avoid duplicates)
            if original_event_type == "violence" and event_type_detected == "cash":
                is_valid = False
                corrected_event_type = original_event_type
                reason = f"[BLOCKED] Violence->Cash correction blocked. {reason}"
            else:
                corrected_event_type = event_type_detected
                reason = f"[CORRECTED: {original_event_type}→{corrected_event_type}] {reason}"
        
        return is_valid, confidence, reason, corrected_event_type

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
            
            # Parse response using new format parser (handles both old and new formats)
            is_valid, confidence, reason, corrected_event_type = self._parse_new_response_format(result, event_type)
            
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
                    reason, prompt, response_raw, image_path, processing_time_ms,
                    validation_type='image'
                )
            else:
                print(f"[GeminiValidator] ⚠️ Skipping database log - no camera_id set!")
            
            print(
                f"[GeminiValidator] IMAGE {event_type}: valid={is_valid}, "
                f"conf={confidence:.2f}, corrected={corrected_event_type}, "
                f"reason={reason[:150]}"
            )
            
            return is_valid, confidence, reason, corrected_event_type
            
        except Exception as e:
            print(f"[GeminiValidator] Exception: {e}")
            # On error, allow the event (don't block on validation errors)
            return True, 1.0, f"Validation error: {e}", event_type
    
    def validate_event_video(self, video_path: str, event_type: str) -> Tuple[bool, float, str, str]:
        """
        Validate a detection event using a 3-second video clip instead of a single frame.
        Provides better context for Gemini to analyze motion and behavior.
        
        Args:
            video_path: Path to the 3-second validation video clip
            event_type: Type of event to validate ('cash', 'violence', 'fire', 'potential_cash')
            
        Returns:
            Tuple of (is_valid, confidence, reason, corrected_event_type)
        """
        if not self.enabled or not self.client:
            return True, 1.0, "Validation disabled", event_type
        
        try:
            import time
            start_time = time.time()
            
            # Read video file
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            
            # Get appropriate prompt
            prompt = self.get_prompt(event_type)
            prompt = f"{prompt}\n\nAnalyze this short video clip (5-10 seconds) showing the detected event. Consider motion, behavior, and context over time."
            
            # Call Gemini API with video
            result = self._call_gemini_api_video(video_bytes, prompt)
            response_raw = json.dumps(result) if isinstance(result, dict) else str(result)
            
            # Parse response
            if 'error' in result:
                print(f"[GeminiValidator] Video API error: {result['error']}")
                return True, 1.0, f"API error: {result['error']}", event_type
            
            # Check if response is empty
            if not result:
                print(f"[GeminiValidator] Empty response from Gemini API")
                return True, 1.0, "Empty API response", event_type
            
            # Parse response using new format parser (handles both old and new formats)
            is_valid, confidence, reason, corrected_event_type = self._parse_new_response_format(result, event_type)
            
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
                'response_raw': response_raw,
                'video_path': video_path,  # Local temp path - will be updated by caller
                'processing_time_ms': processing_time_ms
            }
            
            # NOTE: Database logging is now handled by the caller (validate_detection in utils.py)
            # after it uploads the video to S3 and has the correct URL.
            # This allows us to store S3 URLs instead of temp file paths.
            
            print(f"[GeminiValidator] VIDEO {event_type}: valid={is_valid}, conf={confidence:.2f}, corrected={corrected_event_type}, reason={reason[:150]}")
            
            return is_valid, confidence, reason, corrected_event_type
            
        except Exception as e:
            print(f"[GeminiValidator] Video validation exception: {e}")
            import traceback
            traceback.print_exc()
            # On error, allow the event (don't block on validation errors)
            return True, 1.0, f"Video validation error: {e}", event_type
    
    def _call_gemini_api_video(self, video_bytes: bytes, prompt: str) -> dict:
        """
        Call Gemini API with video and prompt using official SDK.
        
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
                                data=video_bytes,
                                mime_type="video/mp4"
                            )
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_k=1,
                    top_p=1.0,
                    max_output_tokens=1500,
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
                
                # Parse JSON
                result = json.loads(text)
                print(f"[GeminiValidator] Video API response: {result}")
                return result
            else:
                print(f"[GeminiValidator] Video API returned no text. Full response: {response}")
                return {"error": "No response text"}
                
        except json.JSONDecodeError as e:
            print(f"[GeminiValidator] JSON parse error for video: {e}")
            print(f"Response text: {response.text if response else 'None'}")
            return {"error": f"JSON parse error: {e}"}
        except Exception as e:
            print(f"[GeminiValidator] API video call error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
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
        # In ML service, use environment variable directly
        key = api_key or os.environ.get('GEMINI_API_KEY', '')
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
