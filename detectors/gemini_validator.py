"""
Gemini AI Validation Service for Detection Events

This module provides AI-powered validation of detected events using Google's Gemini API.
It acts as a secondary validation layer to reduce false positives by analyzing images
before events are stored in the database.

Usage:
    validator = GeminiValidator(api_key="your-api-key")
    is_valid, confidence, reason = validator.validate_event(frame, "cash")
"""

import cv2
import json
import os
from typing import Tuple, Optional
from google import genai
from google.genai import types


class GeminiValidator:
    """
    Validates detection events using Google Gemini Vision API.
    
    This acts as a filter layer - only events confirmed by Gemini are stored.
    """
    
    # Gemini API - use gemini-2.5-flash-lite (cheapest, FREE standard tier)
    # Best for: high volume, cost-efficient image validation
    # Pricing: FREE (standard) | $0.10/1M input + $0.40/1M output (paid)
    # https://ai.google.dev/gemini-api/docs/pricing
    MODEL_NAME = "gemini-2.5-flash-lite"
    
    # Validation prompts for each event type
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
    
    def __init__(self, api_key: str = None, enabled: bool = True):
        """
        Initialize the Gemini validator.
        
        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            enabled: If False, all validations return True (bypass mode).
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        self.enabled = enabled and bool(self.api_key)
        self.client = None
        
        if self.enabled:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print(f"[GeminiValidator] Initialized with model: {self.MODEL_NAME}")
            except Exception as e:
                print(f"[GeminiValidator] Failed to initialize: {e}")
                self.enabled = False
        
        if not self.api_key and enabled:
            print("[GeminiValidator] Warning: No API key provided, validation disabled")
    
    def _encode_image(self, frame):
        """Convert OpenCV frame to bytes for Gemini API."""
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    
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
    
    def validate_event(self, frame, event_type: str) -> Tuple[bool, float, str]:
        """
        Validate a detection event using Gemini AI.
        
        Args:
            frame: OpenCV frame (numpy array) to analyze
            event_type: Type of event ('cash', 'violence', 'fire')
            
        Returns:
            Tuple of (is_valid, confidence, reason)
            - is_valid: True if Gemini confirms the event
            - confidence: Gemini's confidence score (0.0-1.0)
            - reason: Explanation from Gemini
        """
        # If disabled or no API key, bypass validation
        if not self.enabled:
            return True, 1.0, "Validation bypassed (no API key)"
        
        # Check for valid event type
        if event_type not in self.PROMPTS:
            print(f"[GeminiValidator] Unknown event type: {event_type}")
            return True, 1.0, f"Unknown event type: {event_type}"
        
        # Check frame validity
        if frame is None or frame.size == 0:
            return False, 0.0, "Invalid frame"
        
        try:
            # Encode image
            image_bytes = self._encode_image(frame)
            
            # Get prompt for event type
            prompt = self.PROMPTS[event_type]
            
            # Call Gemini API
            result = self._call_gemini_api(image_bytes, prompt)
            
            # Check for errors
            if 'error' in result:
                # On API error, allow the event (don't block on API issues)
                print(f"[GeminiValidator] API error, allowing event: {result['error']}")
                return True, 1.0, f"API error: {result['error']}"
            
            # Parse result based on event type
            if event_type == 'cash':
                is_valid = result.get('is_cash_transaction', False)
            elif event_type == 'violence':
                is_valid = result.get('is_violence', False)
            elif event_type == 'fire':
                is_valid = result.get('is_fire', False)
            else:
                is_valid = False
            
            confidence = result.get('confidence', 0.0)
            reason = result.get('reason', 'No reason provided')
            
            print(f"[GeminiValidator] {event_type}: valid={is_valid}, conf={confidence:.2f}, reason={reason}")
            
            return is_valid, confidence, reason
            
        except Exception as e:
            print(f"[GeminiValidator] Exception: {e}")
            # On error, allow the event (don't block on validation errors)
            return True, 1.0, f"Validation error: {e}"
    
    def validate_cash_transaction(self, frame) -> Tuple[bool, float, str]:
        """Convenience method for cash transaction validation."""
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


def validate_detection(frame, event_type: str, api_key: str = None) -> Tuple[bool, float, str]:
    """
    Convenience function to validate a detection event.
    
    Args:
        frame: OpenCV frame to analyze
        event_type: Type of event ('cash', 'violence', 'fire')
        api_key: Optional API key
        
    Returns:
        Tuple of (is_valid, confidence, reason)
    """
    validator = get_validator(api_key)
    return validator.validate_event(frame, event_type)
