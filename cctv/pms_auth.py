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
# Note: Kiosk and Call Test roles should NOT have CCTV access
PMS_TO_CCTV_ROLE = {
    "Super Admin": "admin",
    "super_admin": "admin",
    "Master": "admin",  # Legacy support
    "master": "admin",  # Legacy support
    "Team Leader": "project_manager",
    "team_leader": "project_manager",
    "Manager": "project_manager",
    "manager": "project_manager",
    "CLIENT": "project_manager",
    "client": "project_manager",
    "Client": "project_manager",
    "Project": "project_manager",
    "project": "project_manager",
    # Kiosk and Call Test are explicitly NOT included - they have no CCTV access
}


def get_cctv_role(pms_role: str) -> str:
    """
    Map PMS role to CCTV role
    Returns None for roles that shouldn't access CCTV (Kiosk, Call Test)
    """
    return PMS_TO_CCTV_ROLE.get(pms_role, None)


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
            
            # Get user's projects from PMS
            user_projects = pms_user.get("projects", [])
            if not isinstance(user_projects, list):
                user_projects = []
            
            # Extract project IDs
            project_ids = []
            for proj in user_projects:
                if isinstance(proj, dict) and "id" in proj:
                    project_ids.append(str(proj["id"]))
            
            # Also check for single project_id (for non-Team Leaders)
            single_project_id = pms_user.get("project_id")
            if single_project_id and single_project_id not in project_ids:
                project_ids.append(str(single_project_id))
            
            # Check if user can access CCTV
            if "cctv" not in allowed_systems:
                logger.warning(f"User {username} not allowed to access CCTV. Allowed: {allowed_systems}")
                return None
            
            # Map PMS role to CCTV role
            cctv_role = get_cctv_role(pms_role)
            
            # Reject Kiosk and Call Test roles (they should only access kiosk)
            if cctv_role is None:
                logger.warning(f"User {username} with role {pms_role} cannot access CCTV (kiosk-only role)")
                return None
            
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
                request.session["project_ids"] = project_ids  # Store project IDs instead of regions
                
                # Debug logging
                print(f"[PMS AUTH] User {user_id} authenticated")
                print(f"[PMS AUTH] Role: {pms_role}")
                print(f"[PMS AUTH] Projects from PMS: {user_projects}")
                print(f"[PMS AUTH] Single project_id: {single_project_id}")
                print(f"[PMS AUTH] Final project_ids stored in session: {project_ids}")
            
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
    Can check both session token (for web) and Authorization header (for API).
    
    Returns: (is_valid, user_info)
    """
    # Try Authorization header first (for API calls)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
    else:
        # Fall back to session token (for web pages)
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
    Supports both session-based auth (web pages) and JWT header auth (API calls).
    Use this instead of Django's login_required for PMS-authenticated views.
    """
    from functools import wraps
    from django.shortcuts import redirect
    from django.contrib.auth import logout
    from django.http import JsonResponse
    
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # Check if this is an API request (has Authorization header or is JSON request)
        is_api_request = (
            request.headers.get("Authorization", "").startswith("Bearer ") or
            request.path.startswith("/api/") or
            request.content_type == "application/json"
        )
        
        # For API requests, verify token from header
        if is_api_request:
            is_valid, user_info = verify_pms_token(request)
            
            if not is_valid:
                return JsonResponse(
                    {"error": "Unauthorized", "detail": "Invalid or expired token"},
                    status=401
                )
            
            # Create temporary user object for the request (don't need Django session for API)
            if user_info:
                user_email = str(user_info.get("email", ""))
                pms_role = user_info.get("role", "")
                if isinstance(pms_role, dict):
                    pms_role = str(pms_role.get("name", ""))
                else:
                    pms_role = str(pms_role)
                
                cctv_role = get_cctv_role(pms_role)
                
                # Get or create user
                user, _ = User.objects.get_or_create(
                    username=user_email,
                    defaults={
                        "email": user_email,
                        "role": cctv_role,
                        "is_active": True,
                    }
                )
                
                # Attach user to request
                request.user = user
                request.pms_user_info = user_info
            
            return view_func(request, *args, **kwargs)
        
        # For web pages, use session-based authentication
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
            
            # Extract project IDs from PMS user info
            project_ids = []
            user_projects = user_info.get("projects", [])
            for proj in user_projects:
                if isinstance(proj, dict) and "id" in proj:
                    project_ids.append(str(proj["id"]))
            single_project_id = user_info.get("project_id")
            if single_project_id and single_project_id not in project_ids:
                project_ids.append(str(single_project_id))
            request.session["project_ids"] = project_ids
            
            # Update local user role if changed
            cctv_role = get_cctv_role(pms_role)
            if request.user.role != cctv_role:
                request.user.role = cctv_role
                request.user.save(update_fields=["role"])
        
        return view_func(request, *args, **kwargs)
    
    return wrapper


def get_user_project_ids(request):
    """
    Get list of project IDs the current user can access.
    Returns empty list for admin users (meaning all projects).
    """
    if not request.user.is_authenticated:
        return []
    
    # Admin users can access all projects (empty list = all)
    if request.user.is_admin():
        return []
    
    return request.session.get("project_ids", [])


def get_accessible_regions(request):
    """
    DEPRECATED: Kept for backwards compatibility.
    Returns empty queryset since regions are no longer used.
    """
    from cctv.models import Region
    return Region.objects.none()


def filter_branches_by_region(queryset, request):
    """
    DEPRECATED: Use filter_branches_by_project instead.
    Kept for backwards compatibility - now filters by project.
    """
    return filter_branches_by_project(queryset, request)


def filter_branches_by_project(queryset, request):
    """
    Filter Branch queryset by user's assigned projects.
    Admin users see all branches.
    """
    project_ids = get_user_project_ids(request)
    
    # Debug logging
    print(f"[FILTER BRANCHES] User: {request.user.username if request.user.is_authenticated else 'Anonymous'}")
    print(f"[FILTER BRANCHES] project_ids from session: {project_ids}")
    print(f"[FILTER BRANCHES] Is admin: {request.user.is_admin() if request.user.is_authenticated else False}")
    
    # Empty list means admin - no filtering
    if not project_ids:
        total_count = queryset.count()
        print(f"[FILTER BRANCHES] Admin user - returning all {total_count} branches")
        return queryset
    
    # Convert project_ids to UUID objects for database comparison
    from uuid import UUID
    uuid_project_ids = []
    for pid in project_ids:
        try:
            if isinstance(pid, str):
                uuid_project_ids.append(UUID(pid))
            elif isinstance(pid, UUID):
                uuid_project_ids.append(pid)
        except (ValueError, AttributeError):
            print(f"[FILTER BRANCHES] WARNING: Invalid project_id: {pid}")
    
    # Filter by PMS project ID
    filtered = queryset.filter(pms_project_id__in=uuid_project_ids)
    print(f"[FILTER BRANCHES] Filtering by UUID project_ids: {uuid_project_ids}")
    print(f"[FILTER BRANCHES] Filtered branches count: {filtered.count()}")
    
    # Debug: Show which branches exist in DB
    from cctv.models import Branch
    all_branches = Branch.objects.all().values_list('id', 'name', 'pms_project_id')
    print(f"[FILTER BRANCHES] All branches in DB: {list(all_branches)}")
    
    return filtered
