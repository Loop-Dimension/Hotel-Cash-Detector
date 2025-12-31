"""
PMS Authentication Backend for CCTV System

This module handles authentication against the central PMS server.
Users are managed in PMS, and CCTV validates tokens with PMS on each request.
"""
import requests
import logging
from django.conf import settings
from django.contrib.auth.backends import BaseBackend
from django.contrib.auth import get_user_model

logger = logging.getLogger(__name__)

User = get_user_model()

# Role mapping from PMS to CCTV
PMS_TO_CCTV_ROLE = {
    "Super Admin": "admin",
    "super_admin": "admin",
    "Master": "admin",
    "master": "admin",
    "Team Leader": "project_manager",
    "team_leader": "project_manager",
    "Manager": "project_manager",
    "manager": "project_manager",
    "CLIENT": "project_manager",
    "client": "project_manager",
    "Project": "project_manager",
    "project": "project_manager",
}


def get_cctv_role(pms_role: str) -> str:
    """Map PMS role to CCTV role"""
    return PMS_TO_CCTV_ROLE.get(pms_role, "project_manager")


def get_pms_auth_url():
    """Get PMS auth URL from settings"""
    return getattr(settings, 'PMS_AUTH_URL', 'http://localhost:8000')


class PMSAuthBackend(BaseBackend):
    """
    Authentication backend that validates credentials against PMS.
    
    Flow:
    1. User enters credentials
    2. Backend calls PMS /api/v1/auth/login
    3. PMS returns JWT with role and allowed_systems
    4. Backend checks if 'cctv' is in allowed_systems
    5. Backend creates/updates local user with mapped role
    6. Returns user for Django session
    """
    
    def authenticate(self, request, username=None, password=None, **kwargs):
        """Authenticate user against PMS"""
        if not username or not password:
            return None
        
        pms_url = get_pms_auth_url()
        
        try:
            # Call PMS login API
            response = requests.post(
                f"{pms_url}/api/v1/auth/login",
                data={
                    "username": username,  # PMS expects email in username field
                    "password": password,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
            
            if response.status_code != 200:
                logger.warning(f"PMS auth failed for {username}: {response.status_code}")
                return None
            
            data = response.json()
            pms_user = data.get("user", {})
            access_token = str(data.get("access_token", ""))
            
            # Extract and convert all values to simple types
            user_email = str(pms_user.get("email", ""))
            user_id = str(pms_user.get("id", ""))
            
            # Role can be either a dict (with 'name' key) or a string
            pms_role_value = pms_user.get("role", "")
            if isinstance(pms_role_value, dict):
                pms_role = str(pms_role_value.get("name", ""))
            else:
                pms_role = str(pms_role_value)
            
            is_active = bool(pms_user.get("is_active", True))
            allowed_systems = pms_user.get("allowed_systems", [])
            
            # Ensure allowed_systems is a list
            if not isinstance(allowed_systems, list):
                allowed_systems = []
            
            # Check if user can access CCTV
            if "cctv" not in allowed_systems:
                logger.warning(f"User {username} not allowed to access CCTV. Allowed: {allowed_systems}")
                return None
            
            # Map PMS role to CCTV role
            cctv_role = get_cctv_role(pms_role)
            
            # Get or create local user (for Django session)
            user, created = User.objects.get_or_create(
                username=user_email,
                defaults={
                    "email": user_email,
                    "role": cctv_role,
                    "is_active": is_active,
                }
            )
            
            # Update user info from PMS
            if not created:
                user.role = cctv_role
                user.is_active = is_active
                user.save(update_fields=["role", "is_active"])
            
            # Store PMS token in session for subsequent requests
            if request:
                request.session["pms_token"] = access_token
                request.session["pms_user_id"] = user_id
                request.session["pms_role"] = pms_role
                request.session["allowed_systems"] = [str(s) for s in allowed_systems]
            
            logger.info(f"User {username} authenticated via PMS with role {cctv_role}")
            return user
            
        except requests.exceptions.RequestException as e:
            logger.error(f"PMS connection error: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"PMS auth error: {e}", exc_info=True)
            return None
    
    def get_user(self, user_id):
        """Get user by ID"""
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None


def verify_pms_token(request):
    """
    Verify the stored PMS token is still valid.
    Call this on each protected request.
    
    Returns: (is_valid, user_info)
    """
    token = request.session.get("pms_token")
    if not token:
        return False, None
    
    pms_url = get_pms_auth_url()
    
    try:
        response = requests.get(
            f"{pms_url}/api/v1/auth/verify",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        
        if response.status_code != 200:
            return False, None
        
        data = response.json()
        if not data.get("valid"):
            return False, None
        
        user_info = data.get("user", {})
        
        # Re-check if user still has CCTV access
        if "cctv" not in user_info.get("allowed_systems", []):
            return False, None
        
        return True, user_info
        
    except Exception as e:
        logger.error(f"PMS token verification error: {e}")
        return False, None


def pms_login_required(view_func):
    """
    Decorator that validates PMS token on each request.
    Use this instead of Django's login_required for PMS-authenticated views.
    """
    from functools import wraps
    from django.shortcuts import redirect
    from django.contrib.auth import logout
    
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect(settings.LOGIN_URL)
        
        # Verify token with PMS on each request
        is_valid, user_info = verify_pms_token(request)
        
        if not is_valid:
            # Token invalid or user lost CCTV access
            logout(request)
            return redirect(settings.LOGIN_URL)
        
        # Update session with latest user info
        if user_info:
            pms_role = user_info.get("role", "")
            request.session["pms_role"] = str(pms_role)
            request.session["allowed_systems"] = list(user_info.get("allowed_systems", []))
            
            # Update local user role if changed
            cctv_role = get_cctv_role(pms_role)
            if request.user.role != cctv_role:
                request.user.role = cctv_role
                request.user.save(update_fields=["role"])
        
        return view_func(request, *args, **kwargs)
    
    return wrapper
