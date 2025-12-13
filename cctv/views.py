"""
Views for Hotel CCTV Monitoring System
"""
import json
import cv2
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from django.db.models import Count, Q
from django.conf import settings

from .models import User, Region, Branch, Camera, Event, VideoRecord, BranchAccount
from .translations import get_translation, t

# Add root directory to path for detector imports
sys.path.insert(0, str(settings.BASE_DIR))

# Try to import detectors
try:
    from detectors import UnifiedDetector
    DETECTOR_AVAILABLE = True
except ImportError as e:
    DETECTOR_AVAILABLE = False
    print(f"Warning: Detectors not available - {e}")

import threading
import time

# Global detector instances per camera
camera_detectors = {}

# Global background worker state - using threading for Gunicorn compatibility
# Multiprocessing has issues with Gunicorn workers and semaphore limits
background_workers = {}  # camera_id -> BackgroundCameraWorker (threading)
background_worker_lock = threading.Lock()


def get_user_branches(user):
    """Get branches accessible by the user"""
    if user.is_admin():
        return Branch.objects.all()
    return user.managed_branches.all()


# ==================== AUTHENTICATION ====================

def login_view(request):
    """Login page"""
    if request.user.is_authenticated:
        return redirect('cctv:home')
    
    error = None
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            next_url = request.GET.get('next', '/')
            return redirect(next_url)
        else:
            error = '아이디 또는 비밀번호가 올바르지 않습니다.'
    
    return render(request, 'cctv/login.html', {'error': error})


def logout_view(request):
    """Logout"""
    logout(request)
    return redirect('cctv:login')


# ==================== MAIN PAGES ====================

@login_required
def home(request):
    """Dashboard home page"""
    user = request.user
    branches = get_user_branches(user)
    regions = Region.objects.all()
    
    context = {
        'user': user,
        'branches': branches,
        'regions': regions,
        'active_page': 'home',
    }
    return render(request, 'cctv/home.html', context)


@login_required
def monitor_all(request):
    """All branches monitoring page (Admin only)"""
    user = request.user
    if not user.is_admin():
        return redirect('cctv:monitor_local')
    
    branches = Branch.objects.all().select_related('region')
    regions = Region.objects.all()
    
    context = {
        'user': user,
        'branches': branches,
        'regions': regions,
        'active_page': 'monitor-all',
    }
    return render(request, 'cctv/monitor_all.html', context)


@login_required
def monitor_local(request, branch_id=None):
    """Local branch monitoring page"""
    user = request.user
    user_branches = get_user_branches(user)
    
    if branch_id:
        branch = get_object_or_404(Branch, id=branch_id)
        if not user.is_admin() and branch not in user_branches:
            return redirect('cctv:home')
    else:
        branch = user_branches.first()
    
    cameras = branch.cameras.all() if branch else []
    regions = Region.objects.all()
    
    context = {
        'user': user,
        'branch': branch,
        'branches': user_branches,
        'cameras': cameras,
        'regions': regions,
        'active_page': 'monitor-local',
    }
    return render(request, 'cctv/monitor_local.html', context)


@login_required
def video_logs(request):
    """Event logs page"""
    user = request.user
    user_branches = get_user_branches(user)
    regions = Region.objects.all()
    
    # Get filter parameters
    date_from = request.GET.get('from', '')
    date_to = request.GET.get('to', '')
    region_filter = request.GET.get('region', '')
    type_filter = request.GET.get('type', '')
    branch_filter = request.GET.get('branch', '')
    
    # Build query
    events = Event.objects.select_related('branch', 'camera', 'branch__region')
    
    if not user.is_admin():
        events = events.filter(branch__in=user_branches)
    
    # Date range filter
    if date_from:
        events = events.filter(created_at__date__gte=date_from)
    if date_to:
        events = events.filter(created_at__date__lte=date_to)
    
    # Region filter (by ID)
    if region_filter:
        try:
            events = events.filter(branch__region_id=int(region_filter))
        except ValueError:
            pass
    
    # Event type filter
    if type_filter:
        events = events.filter(event_type=type_filter)
    
    # Branch filter (by ID)
    if branch_filter:
        try:
            events = events.filter(branch_id=int(branch_filter))
        except ValueError:
            events = events.filter(branch__name__icontains=branch_filter)
    
    events = events.order_by('-created_at')[:100]
    
    # Get offline cameras
    offline_cameras = Camera.objects.filter(status='offline').select_related('branch')
    if not user.is_admin():
        offline_cameras = offline_cameras.filter(branch__in=user_branches)
    
    context = {
        'user': user,
        'events': events,
        'branches': user_branches,
        'offline_cameras': offline_cameras,
        'regions': regions,
        'active_page': 'video-logs',
        'filters': {
            'date_from': date_from,
            'date_to': date_to,
            'region': region_filter,
            'type': type_filter,
            'branch': branch_filter,
        }
    }
    return render(request, 'cctv/video_logs.html', context)


@login_required
def video_full(request):
    """Full videos page"""
    user = request.user
    user_branches = get_user_branches(user)
    regions = Region.objects.all()
    
    # Get filter parameters
    date_filter = request.GET.get('date', '')
    region_filter = request.GET.get('region', 'all')
    branch_filter = request.GET.get('branch', '')
    
    # Build query
    videos = VideoRecord.objects.select_related('branch', 'camera', 'branch__region')
    
    if not user.is_admin():
        videos = videos.filter(branch__in=user_branches)
    
    if date_filter:
        videos = videos.filter(recorded_date=date_filter)
    
    if region_filter != 'all':
        videos = videos.filter(branch__region__name=region_filter)
    
    if branch_filter:
        videos = videos.filter(branch__name__icontains=branch_filter)
    
    videos = videos.order_by('-recorded_date')[:50]
    
    context = {
        'user': user,
        'videos': videos,
        'branches': user_branches,
        'regions': regions,
        'active_page': 'video-full',
        'filters': {
            'date': date_filter,
            'region': region_filter,
            'branch': branch_filter,
        }
    }
    return render(request, 'cctv/video_full.html', context)


@login_required
def manage_branches(request):
    """Branch management page (Admin only)"""
    user = request.user
    if not user.is_admin():
        return redirect('cctv:home')
    
    branches = Branch.objects.all().select_related('region')
    regions = Region.objects.all()
    
    # Get filter parameters
    region_filter = request.GET.get('region', '전체')
    search_filter = request.GET.get('search', '')
    
    if region_filter != '전체':
        branches = branches.filter(region__name=region_filter)
    
    if search_filter:
        branches = branches.filter(name__icontains=search_filter)
    
    context = {
        'user': user,
        'branches': branches,
        'regions': regions,
        'active_page': 'manage-branches',
        'filters': {
            'region': region_filter,
            'search': search_filter,
        }
    }
    return render(request, 'cctv/manage_branches.html', context)


@login_required
def manage_branch_detail(request, branch_id):
    """Branch detail management page"""
    user = request.user
    branch = get_object_or_404(Branch, id=branch_id)
    
    if not user.is_admin() and branch not in get_user_branches(user):
        return redirect('cctv:home')
    
    accounts = branch.accounts.all()
    cameras = branch.cameras.all()
    regions = Region.objects.all()
    
    context = {
        'user': user,
        'branch': branch,
        'accounts': accounts,
        'cameras': cameras,
        'regions': regions,
        'active_page': 'manage-branch-detail',
    }
    return render(request, 'cctv/manage_branch_detail.html', context)


@login_required
def camera_settings(request, camera_id):
    """Camera settings page - configure zone and detection thresholds"""
    user = request.user
    camera = get_object_or_404(Camera, id=camera_id)
    
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return redirect('cctv:home')
    
    context = {
        'user': user,
        'camera': camera,
        'active_page': 'camera-settings',
    }
    return render(request, 'cctv/camera_settings.html', context)


@login_required
def reports(request):
    """Reports page (Admin only)"""
    user = request.user
    if not user.is_admin():
        return redirect('cctv:home')
    
    regions = Region.objects.all()
    branches = Branch.objects.all().select_related('region')
    
    context = {
        'user': user,
        'regions': regions,
        'branches': branches,
        'active_page': 'reports',
    }
    return render(request, 'cctv/reports.html', context)


# ==================== API ENDPOINTS ====================

@login_required
@require_http_methods(["GET", "POST"])
def api_branches(request):
    """API for branches"""
    user = request.user
    
    if request.method == 'GET':
        branches = get_user_branches(user).select_related('region')
        data = [{
            'id': b.id,
            'name': b.name,
            'region': b.region.name,
            'status': b.status,
            'status_display': b.get_status_display(),
            'camera_count': b.get_camera_count(),
            'event_count': b.get_today_event_count(),
        } for b in branches]
        return JsonResponse({'branches': data})
    
    elif request.method == 'POST':
        if not user.is_admin():
            return JsonResponse({'error': 'Permission denied'}, status=403)
        
        data = json.loads(request.body)
        
        # Support both region_id and region name
        if 'region_id' in data:
            region = get_object_or_404(Region, id=data['region_id'])
        else:
            region = get_object_or_404(Region, name=data.get('region'))
        
        branch = Branch.objects.create(
            name=data.get('name'),
            region=region,
            address=data.get('address', ''),
            status='pending'
        )
        
        return JsonResponse({
            'success': True,
            'branch': {
                'id': branch.id,
                'name': branch.name,
                'region': branch.region.name,
            }
        })


@login_required
@require_http_methods(["GET", "PUT", "DELETE"])
def api_branch_detail(request, branch_id):
    """API for single branch"""
    user = request.user
    branch = get_object_or_404(Branch, id=branch_id)
    
    if not user.is_admin() and branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    if request.method == 'GET':
        return JsonResponse({
            'id': branch.id,
            'name': branch.name,
            'region': branch.region.name,
            'region_id': branch.region.id,
            'status': branch.status,
            'status_display': branch.get_status_display(),
            'address': branch.address,
            'camera_count': branch.get_camera_count(),
        })
    
    elif request.method == 'PUT':
        if not user.is_admin():
            return JsonResponse({'error': 'Permission denied'}, status=403)
        
        data = json.loads(request.body)
        branch.name = data.get('name', branch.name)
        branch.address = data.get('address', branch.address)
        
        if 'region_id' in data:
            region = get_object_or_404(Region, id=data['region_id'])
            branch.region = region
        
        if 'status' in data:
            branch.status = data['status']
        
        branch.save()
        return JsonResponse({'success': True})
    
    elif request.method == 'DELETE':
        if not user.is_admin():
            return JsonResponse({'error': 'Permission denied'}, status=403)
        branch.delete()
        return JsonResponse({'success': True})
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)


@login_required
@require_http_methods(["GET", "POST"])
def api_branch_cameras(request, branch_id):
    """API for branch cameras"""
    user = request.user
    branch = get_object_or_404(Branch, id=branch_id)
    
    if not user.is_admin() and branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    if request.method == 'GET':
        cameras = branch.cameras.all()
        data = [{
            'id': c.id,
            'camera_id': c.camera_id,
            'name': c.name,
            'location': c.location,
            'status': c.status,
            'status_display': c.get_status_display(),
            'rtsp_url': c.rtsp_url,
        } for c in cameras]
        return JsonResponse({'cameras': data})
    
    elif request.method == 'POST':
        data = json.loads(request.body)
        
        camera = Camera.objects.create(
            branch=branch,
            camera_id=data.get('camera_id'),
            name=data.get('name'),
            location=data.get('location', ''),
            rtsp_url=data.get('rtsp_url'),
            status='offline'
        )
        
        return JsonResponse({
            'success': True,
            'camera': {
                'id': camera.id,
                'camera_id': camera.camera_id,
                'name': camera.name,
            }
        })


@login_required
@require_http_methods(["GET", "POST"])
def api_cameras(request):
    """API for all cameras"""
    user = request.user
    user_branches = get_user_branches(user)
    
    if request.method == 'GET':
        cameras = Camera.objects.filter(branch__in=user_branches).select_related('branch')
        data = [{
            'id': c.id,
            'camera_id': c.camera_id,
            'name': c.name,
            'branch': c.branch.name,
            'branch_id': c.branch.id,
            'location': c.location,
            'status': c.status,
            'status_display': c.get_status_display(),
        } for c in cameras]
        return JsonResponse({'cameras': data})
    
    elif request.method == 'POST':
        data = json.loads(request.body)
        
        # Get branch
        branch_id = data.get('branch_id')
        if not branch_id:
            return JsonResponse({'error': 'branch_id is required'}, status=400)
        
        branch = get_object_or_404(Branch, id=branch_id)
        
        if not user.is_admin() and branch not in user_branches:
            return JsonResponse({'error': 'Permission denied'}, status=403)
        
        # Generate camera_id if not provided
        camera_id_str = data.get('camera_id') or data.get('name', '').lower().replace(' ', '-')
        
        camera = Camera.objects.create(
            branch=branch,
            camera_id=camera_id_str,
            name=data.get('name'),
            location=data.get('location', ''),
            rtsp_url=data.get('rtsp_url', ''),
            status='online'  # Default to online
        )
        
        return JsonResponse({
            'success': True,
            'camera': {
                'id': camera.id,
                'camera_id': camera.camera_id,
                'name': camera.name,
            }
        })


@login_required
@require_http_methods(["GET", "PUT", "DELETE"])
def api_camera_detail(request, camera_id):
    """API for single camera"""
    user = request.user
    camera = get_object_or_404(Camera, id=camera_id)
    
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    if request.method == 'GET':
        return JsonResponse({
            'id': camera.id,
            'camera_id': camera.camera_id,
            'name': camera.name,
            'branch': camera.branch.name,
            'branch_id': camera.branch.id,
            'location': camera.location,
            'status': camera.status,
            'rtsp_url': camera.rtsp_url,
            # Cashier zone settings
            'cashier_zone': camera.get_cashier_zone(),
            # Detection toggles
            'detect_cash': camera.detect_cash,
            'detect_violence': camera.detect_violence,
            'detect_fire': camera.detect_fire,
            # Independent confidence thresholds
            'cash_confidence': camera.cash_confidence,
            'violence_confidence': camera.violence_confidence,
            'fire_confidence': camera.fire_confidence,
            # Full detection settings
            'detection_settings': camera.get_detection_settings(),
        })
    
    elif request.method == 'PUT':
        data = json.loads(request.body)
        
        # Basic info
        camera.camera_id = data.get('camera_id', camera.camera_id)
        camera.name = data.get('name', camera.name)
        camera.location = data.get('location', camera.location)
        camera.rtsp_url = data.get('rtsp_url', camera.rtsp_url)
        camera.status = data.get('status', camera.status)
        
        # Detection toggles
        if 'detect_cash' in data:
            camera.detect_cash = data['detect_cash']
        if 'detect_violence' in data:
            camera.detect_violence = data['detect_violence']
        if 'detect_fire' in data:
            camera.detect_fire = data['detect_fire']
        
        # Independent confidence thresholds
        if 'cash_confidence' in data:
            camera.cash_confidence = float(data['cash_confidence'])
        if 'violence_confidence' in data:
            camera.violence_confidence = float(data['violence_confidence'])
        if 'fire_confidence' in data:
            camera.fire_confidence = float(data['fire_confidence'])
        if 'pose_confidence' in data:
            camera.pose_confidence = max(0.1, min(0.9, float(data['pose_confidence'])))
        
        # Hand touch distance for cash detection
        if 'hand_touch_distance' in data:
            camera.hand_touch_distance = max(30, min(300, int(data['hand_touch_distance'])))
        
        # Cashier zone
        if 'cashier_zone' in data:
            zone = data['cashier_zone']
            camera.cashier_zone_x = int(zone.get('x', camera.cashier_zone_x))
            camera.cashier_zone_y = int(zone.get('y', camera.cashier_zone_y))
            camera.cashier_zone_width = int(zone.get('width', camera.cashier_zone_width))
            camera.cashier_zone_height = int(zone.get('height', camera.cashier_zone_height))
            camera.cashier_zone_enabled = zone.get('enabled', camera.cashier_zone_enabled)
        
        camera.save()
        return JsonResponse({'success': True, 'camera': camera.get_detection_settings()})
    
    elif request.method == 'DELETE':
        camera.delete()
        return JsonResponse({'success': True})


@login_required
def api_events(request):
    """API for events"""
    user = request.user
    user_branches = get_user_branches(user)
    
    # Get filter parameters
    date_filter = request.GET.get('date')
    region_filter = request.GET.get('region')
    type_filter = request.GET.get('type')
    branch_filter = request.GET.get('branch')
    limit = int(request.GET.get('limit', 50))
    
    events = Event.objects.filter(branch__in=user_branches).select_related('branch', 'camera', 'branch__region')
    
    if date_filter:
        events = events.filter(created_at__date=date_filter)
    
    if region_filter and region_filter != 'all':
        events = events.filter(branch__region__name=region_filter)
    
    if type_filter and type_filter != 'all':
        events = events.filter(event_type=type_filter)
    
    if branch_filter:
        events = events.filter(branch__name__icontains=branch_filter)
    
    events = events.order_by('-created_at')[:limit]
    
    data = [{
        'id': e.id,
        'branch': e.branch.name,
        'region': e.branch.region.name,
        'camera': e.camera.camera_id,
        'type': e.event_type,
        'type_display': e.get_event_type_display(),
        'status': e.status,
        'status_display': e.get_status_display(),
        'confidence': e.confidence,
        'date': e.created_at.strftime('%Y-%m-%d'),
        'time': e.created_at.strftime('%H:%M'),
        'created_at': e.created_at.isoformat(),
    } for e in events]
    
    return JsonResponse({'events': data})


@login_required
@require_http_methods(["GET", "PUT", "DELETE"])
def api_event_detail(request, event_id):
    """API for single event"""
    user = request.user
    event = get_object_or_404(Event, id=event_id)
    
    if not user.is_admin() and event.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    if request.method == 'GET':
        return JsonResponse({
            'id': event.id,
            'branch': event.branch.name,
            'branch_name': event.branch.name,
            'camera': event.camera.camera_id,
            'camera_id': event.camera.id,
            'camera_name': event.camera.name,
            'event_type': event.event_type,
            'type': event.event_type,
            'type_display': event.get_event_type_display(),
            'status': event.status,
            'status_display': event.get_status_display(),
            'confidence': event.confidence,
            'frame_number': event.frame_number,
            'bbox': event.get_bbox(),
            'clip_path': event.clip_path,
            'thumbnail_path': event.thumbnail_path if hasattr(event, 'thumbnail_path') else None,
            'created_at': event.created_at.isoformat(),
        })
    
    elif request.method == 'PUT':
        data = json.loads(request.body)
        event.status = data.get('status', event.status)
        event.notes = data.get('notes', event.notes)
        if data.get('status') in ['confirmed', 'reviewing']:
            event.reviewed_by = user
            event.reviewed_at = timezone.now()
        event.save()
        return JsonResponse({'success': True})
    
    elif request.method == 'DELETE':
        if not user.is_admin():
            return JsonResponse({'error': 'Permission denied'}, status=403)
        event.delete()
        return JsonResponse({'success': True})


@login_required
def api_videos(request):
    """API for video records"""
    user = request.user
    user_branches = get_user_branches(user)
    
    videos = VideoRecord.objects.filter(branch__in=user_branches).select_related('branch', 'camera', 'branch__region')
    
    date_filter = request.GET.get('date')
    region_filter = request.GET.get('region')
    branch_filter = request.GET.get('branch')
    
    if date_filter:
        videos = videos.filter(recorded_date=date_filter)
    
    if region_filter and region_filter != 'all':
        videos = videos.filter(branch__region__name=region_filter)
    
    if branch_filter:
        videos = videos.filter(branch__name__icontains=branch_filter)
    
    videos = videos.order_by('-recorded_date')[:50]
    
    data = [{
        'id': v.id,
        'branch': v.branch.name,
        'region': v.branch.region.name,
        'camera': v.camera.camera_id,
        'file_id': v.file_id,
        'date': v.recorded_date.isoformat(),
        'duration': v.duration,
        'file_size': v.file_size,
    } for v in videos]
    
    return JsonResponse({'videos': data})


@login_required
def api_home_stats(request):
    """API for home page statistics"""
    user = request.user
    user_branches = get_user_branches(user)
    today = timezone.now().date()
    
    # Calculate overall stats
    total_branches = user_branches.count()
    today_events = Event.objects.filter(branch__in=user_branches, created_at__date=today).count()
    online_cameras = Camera.objects.filter(branch__in=user_branches, status='online').count()
    pending_review = Event.objects.filter(branch__in=user_branches, status='pending').count()
    
    # Branch list with stats
    branches = []
    for branch in user_branches:
        event_count = branch.events.filter(created_at__date=today).count()
        confirmed = branch.events.filter(created_at__date=today, status='confirmed').count()
        pending = branch.events.filter(created_at__date=today, status='pending').count()
        
        if pending > 0:
            status = 'pending'
        elif confirmed == event_count and event_count > 0:
            status = 'confirmed'
        else:
            status = 'reviewing'
        
        branches.append({
            'id': branch.id,
            'name': branch.name,
            'event_count': event_count,
            'status': status,
        })
    
    return JsonResponse({
        'total_branches': total_branches,
        'today_events': today_events,
        'online_cameras': online_cameras,
        'pending_review': pending_review,
        'branches': branches,
    })


@login_required
def api_report_stats(request):
    """API for report statistics"""
    user = request.user
    if not user.is_admin():
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    user_branches = get_user_branches(user)
    today = timezone.now().date()
    month_start = today.replace(day=1)
    
    # Monthly stats
    monthly_events = Event.objects.filter(
        branch__in=user_branches,
        created_at__date__gte=month_start
    ).count()
    
    # Event type breakdown
    type_counts = Event.objects.filter(
        branch__in=user_branches,
        created_at__date__gte=month_start
    ).values('event_type').annotate(count=Count('id'))
    
    type_breakdown = {t['event_type']: t['count'] for t in type_counts}
    total = sum(type_breakdown.values()) or 1
    
    pie_data = [
        {'label': '현금', 'value': round(type_breakdown.get('cash', 0) / total * 100), 'color': '#1c1373'},
        {'label': '화재', 'value': round(type_breakdown.get('fire', 0) / total * 100), 'color': '#f28c28'},
        {'label': '난동', 'value': round(type_breakdown.get('violence', 0) / total * 100), 'color': '#3cb371'},
    ]
    
    # Daily stats for last 7 days
    daily_data = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = Event.objects.filter(
            branch__in=user_branches,
            created_at__date=day
        ).count()
        daily_data.append({
            'x': day.strftime('%m-%d'),
            'y': count
        })
    
    # Branch summary
    branch_summary = []
    for branch in user_branches[:5]:
        total_events = branch.events.filter(created_at__date__gte=month_start).count()
        branch_summary.append({
            'branch': branch.name,
            'total': total_events,
            'avg': '02:30',  # Placeholder
            'falseRate': '3.0%',  # Placeholder
        })
    
    return JsonResponse({
        'monthly_events': monthly_events,
        'pie_data': pie_data,
        'daily_data': daily_data,
        'branch_summary': branch_summary,
    })


@login_required
def api_reports(request):
    """API for reports page - returns all chart data"""
    user = request.user
    if not user.is_admin():
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    user_branches = get_user_branches(user)
    today = timezone.now().date()
    
    # Parse date range
    date_range = request.GET.get('date_range', 'week')
    region_id = request.GET.get('region_id')
    branch_id = request.GET.get('branch_id')
    
    if date_range == 'today':
        start_date = today
        end_date = today
    elif date_range == 'week':
        start_date = today - timedelta(days=7)
        end_date = today
    elif date_range == 'month':
        start_date = today.replace(day=1)
        end_date = today
    elif date_range == 'custom':
        start_date = request.GET.get('start_date', str(today - timedelta(days=7)))
        end_date = request.GET.get('end_date', str(today))
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date() if isinstance(start_date, str) else start_date
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date() if isinstance(end_date, str) else end_date
    else:
        start_date = today - timedelta(days=7)
        end_date = today
    
    # Base query
    events = Event.objects.filter(
        branch__in=user_branches,
        created_at__date__gte=start_date,
        created_at__date__lte=end_date
    )
    
    # Apply filters
    if region_id:
        events = events.filter(branch__region_id=region_id)
    if branch_id:
        events = events.filter(branch_id=branch_id)
    
    # Summary
    total_events = events.count()
    high_priority = events.filter(confidence__gte=0.8).count()
    pending = events.filter(status='pending').count()
    resolved = events.filter(status='confirmed').count()
    
    summary = {
        'total': total_events,
        'high_priority': high_priority,
        'pending': pending,
        'resolved': resolved,
    }
    
    # Events by type
    by_type = list(events.values('event_type').annotate(count=Count('id')))
    by_type = [{'type': t['event_type'].title(), 'count': t['count']} for t in by_type]
    
    # Events by branch
    by_branch = list(events.values('branch__name').annotate(count=Count('id')).order_by('-count')[:10])
    by_branch = [{'branch': b['branch__name'], 'count': b['count']} for b in by_branch]
    
    # Timeline - daily counts
    timeline = []
    current = start_date
    while current <= end_date:
        count = events.filter(created_at__date=current).count()
        timeline.append({
            'date': current.strftime('%m/%d'),
            'count': count
        })
        current += timedelta(days=1)
    
    # Events by hour
    by_hour = []
    for hour in range(24):
        count = events.filter(created_at__hour=hour).count()
        by_hour.append({
            'hour': hour,
            'count': count
        })
    
    # Detailed data for table
    details = []
    branch_type_groups = events.values('branch__name', 'event_type', 'created_at__date').annotate(
        count=Count('id'),
        high_priority=Count('id', filter=Q(confidence__gte=0.8)),
        resolved=Count('id', filter=Q(status='confirmed'))
    ).order_by('-created_at__date')[:50]
    
    for item in branch_type_groups:
        details.append({
            'date': item['created_at__date'].strftime('%Y-%m-%d') if item['created_at__date'] else '',
            'branch': item['branch__name'],
            'event_type': item['event_type'].title(),
            'count': item['count'],
            'high_priority': item['high_priority'],
            'resolved': item['resolved'],
        })
    
    return JsonResponse({
        'summary': summary,
        'by_type': by_type,
        'by_branch': by_branch,
        'timeline': timeline,
        'by_hour': by_hour,
        'details': details,
    })


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def api_set_cashier_zone(request, camera_id):
    """Set cashier zone for a camera"""
    user = request.user
    camera = get_object_or_404(Camera, id=camera_id)
    
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    data = json.loads(request.body)
    
    # Support both formats: dict or list
    if 'zone' in data:
        zone = data['zone']
        if isinstance(zone, list) and len(zone) == 4:
            camera.set_cashier_zone(zone[0], zone[1], zone[2], zone[3], True)
        elif isinstance(zone, dict):
            camera.set_cashier_zone(
                zone.get('x', 0),
                zone.get('y', 0),
                zone.get('width', 640),
                zone.get('height', 480),
                zone.get('enabled', True)
            )
    else:
        # Direct parameters
        camera.set_cashier_zone(
            data.get('x', camera.cashier_zone_x),
            data.get('y', camera.cashier_zone_y),
            data.get('width', camera.cashier_zone_width),
            data.get('height', camera.cashier_zone_height),
            data.get('enabled', True)
        )
    
    return JsonResponse({'success': True, 'cashier_zone': camera.get_cashier_zone()})


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def api_camera_settings(request, camera_id):
    """Update all camera detection settings at once"""
    user = request.user
    camera = get_object_or_404(Camera, id=camera_id)
    
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    data = json.loads(request.body)
    
    # Detection toggles
    if 'detect_cash' in data:
        camera.detect_cash = bool(data['detect_cash'])
    if 'detect_violence' in data:
        camera.detect_violence = bool(data['detect_violence'])
    if 'detect_fire' in data:
        camera.detect_fire = bool(data['detect_fire'])
    
    # Confidence thresholds (independent per camera)
    if 'cash_confidence' in data:
        camera.cash_confidence = max(0.0, min(1.0, float(data['cash_confidence'])))
    if 'violence_confidence' in data:
        camera.violence_confidence = max(0.0, min(1.0, float(data['violence_confidence'])))
    if 'fire_confidence' in data:
        camera.fire_confidence = max(0.0, min(1.0, float(data['fire_confidence'])))
    if 'pose_confidence' in data:
        camera.pose_confidence = max(0.1, min(0.9, float(data['pose_confidence'])))
    
    # Hand touch distance for cash detection
    if 'hand_touch_distance' in data:
        camera.hand_touch_distance = max(30, min(300, int(data['hand_touch_distance'])))
    
    # Cashier zone
    if 'cashier_zone' in data:
        zone = data['cashier_zone']
        camera.cashier_zone_x = int(zone.get('x', camera.cashier_zone_x))
        camera.cashier_zone_y = int(zone.get('y', camera.cashier_zone_y))
        camera.cashier_zone_width = int(zone.get('width', camera.cashier_zone_width))
        camera.cashier_zone_height = int(zone.get('height', camera.cashier_zone_height))
        camera.cashier_zone_enabled = bool(zone.get('enabled', camera.cashier_zone_enabled))
    
    camera.save()
    
    return JsonResponse({
        'success': True,
        'settings': camera.get_detection_settings()
    })


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def api_test_camera_connection(request, camera_id):
    """Test camera RTSP connection and update status"""
    user = request.user
    camera = get_object_or_404(Camera, id=camera_id)
    
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    try:
        import os
        # Use TCP transport to avoid RTP packet ordering issues
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000'
        
        # Try to connect to the RTSP stream
        cap = cv2.VideoCapture(camera.rtsp_url, cv2.CAP_FFMPEG)
        
        # Set timeout (5 seconds)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        
        if cap.isOpened():
            # Try to read one frame
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Connection successful
                camera.status = 'online'
                camera.last_connected = timezone.now()
                camera.save()
                return JsonResponse({
                    'success': True,
                    'online': True,
                    'message': 'Camera connected successfully'
                })
            else:
                camera.status = 'offline'
                camera.save()
                return JsonResponse({
                    'success': True,
                    'online': False,
                    'error': 'Connected but cannot read frames'
                })
        else:
            camera.status = 'offline'
            camera.save()
            return JsonResponse({
                'success': True,
                'online': False,
                'error': 'Cannot connect to RTSP stream'
            })
            
    except Exception as e:
        camera.status = 'offline'
        camera.save()
        return JsonResponse({
            'success': False,
            'online': False,
            'error': str(e)
        })

@login_required
@csrf_exempt
@require_http_methods(["POST"])
def api_toggle_detection(request, camera_id):
    """Toggle detection types for a camera"""
    user = request.user
    camera = get_object_or_404(Camera, id=camera_id)
    
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    data = json.loads(request.body)
    detection_type = data.get('type')
    enabled = data.get('enabled', True)
    
    if detection_type == 'cash':
        camera.detect_cash = enabled
    elif detection_type == 'violence':
        camera.detect_violence = enabled
    elif detection_type == 'fire':
        camera.detect_fire = enabled
    
    camera.save()
    
    return JsonResponse({
        'success': True,
        'detect_cash': camera.detect_cash,
        'detect_violence': camera.detect_violence,
        'detect_fire': camera.detect_fire,
    })


@login_required
@require_http_methods(["GET", "POST"])
def api_branch_accounts(request, branch_id):
    """API for branch accounts"""
    user = request.user
    branch = get_object_or_404(Branch, id=branch_id)
    
    if not user.is_admin() and branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    if request.method == 'GET':
        accounts = branch.accounts.all()
        data = [{
            'id': a.id,
            'name': a.name,
            'email': a.email,
            'role': a.role,
            'role_display': a.get_role_display(),
        } for a in accounts]
        return JsonResponse({'accounts': data})
    
    elif request.method == 'POST':
        data = json.loads(request.body)
        account = BranchAccount.objects.create(
            branch=branch,
            name=data.get('name'),
            email=data.get('email'),
            role=data.get('role', 'staff')
        )
        return JsonResponse({
            'success': True,
            'account': {
                'id': account.id,
                'name': account.name,
                'email': account.email,
                'role': account.role,
            }
        })


# ==================== VIDEO STREAMING ====================

def get_detector_for_camera(camera):
    """Get or create detector for a camera with camera-specific settings"""
    global camera_detectors
    
    if not DETECTOR_AVAILABLE:
        return None
    
    # Always update detector if settings changed
    cache_key = f"{camera.id}_{camera.updated_at.timestamp()}"
    
    if camera.id not in camera_detectors or camera_detectors.get(f'{camera.id}_key') != cache_key:
        # Use camera-specific confidence thresholds
        config = {
            'models_dir': str(settings.DETECTION_CONFIG['MODELS_DIR']),
            'cashier_zone': [
                camera.cashier_zone_x,
                camera.cashier_zone_y,
                camera.cashier_zone_width,
                camera.cashier_zone_height
            ],
            'cashier_zone_enabled': camera.cashier_zone_enabled,
            # Hide zone overlay by default (UI only - backend detection still works)
            'show_zone_overlay': False,
            # Camera-specific confidence thresholds
            'cash_confidence': camera.cash_confidence,
            'violence_confidence': camera.violence_confidence,
            'fire_confidence': camera.fire_confidence,
            # Detection toggles
            'detect_cash': camera.detect_cash,
            'detect_violence': camera.detect_violence,
            'detect_fire': camera.detect_fire,
            # Camera-specific hand touch distance (or use global default)
            'hand_touch_distance': getattr(camera, 'hand_touch_distance', 100),
            # Other settings from global config
            'pose_confidence': settings.DETECTION_CONFIG.get('POSE_CONFIDENCE', 0.5),
            'min_transaction_frames': settings.DETECTION_CONFIG.get('MIN_TRANSACTION_FRAMES', 1),
            'min_fire_frames': settings.DETECTION_CONFIG.get('MIN_FIRE_FRAMES', 15),
            'min_fire_area': settings.DETECTION_CONFIG.get('MIN_FIRE_AREA', 500),
            'violence_duration': settings.DETECTION_CONFIG.get('VIOLENCE_DURATION', 30),
        }
        camera_detectors[camera.id] = UnifiedDetector(config)
        camera_detectors[f'{camera.id}_key'] = cache_key
    
    return camera_detectors[camera.id]


def generate_frames(camera):
    """Generator for video frames with detection overlay.
    
    Optimization: If a background worker is already connected to this camera,
    use its frames instead of making a new connection (no delay!)
    """
    
    # Check if background worker is running for this camera
    worker = None
    with background_worker_lock:
        if camera.id in background_workers:
            worker = background_workers[camera.id]
            if not worker.running:
                worker = None
    
    # If worker exists, use its frames
    if worker is not None:
        print(f"[{camera.camera_id}] Using shared connection from background worker")
        
        # Pre-allocate encode params - quality 75 for faster encoding
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]
        last_frame_time = 0
        
        while worker.running:
            frame = worker.get_current_frame(with_overlay=True)
            if frame is not None:
                # Throttle to ~30fps
                current_time = time.time()
                if current_time - last_frame_time >= 0.033:
                    _, buffer = cv2.imencode('.jpg', frame, encode_params)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    last_frame_time = current_time
            time.sleep(0.015)  # ~66fps check rate for smoother streaming
        return
    
    # No background worker - make new connection (legacy behavior)
    # Use TCP transport to avoid RTP packet ordering issues (bad cseq errors)
    import os
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000'
    cap = cv2.VideoCapture(camera.rtsp_url, cv2.CAP_FFMPEG)
    
    # Set timeout for connection
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    if not cap.isOpened():
        # Only set offline if connection actually failed
        camera.status = 'offline'
        camera.save()
        # Return placeholder frame
        frame = create_placeholder_frame("Cannot connect to camera")
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    # Update camera status - connected successfully
    camera.status = 'online'
    camera.last_connected = timezone.now()
    camera.save()
    
    detector = get_detector_for_camera(camera)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip = settings.DETECTION_CONFIG['FRAME_SKIP']
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % frame_skip != 0:
                continue
            
            # Process frame with detector
            if detector:
                result = detector.process_frame(frame, draw_overlay=True)
                frame = result['frame']
                
                # Save detections to database
                for det in result.get('detections', []):
                    save_detection(camera, det, frame_count)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    finally:
        cap.release()
        # Don't set offline when stream ends normally
        # Camera stays online until connection actually fails


def create_placeholder_frame(text):
    """Create a placeholder frame with text"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def save_detection(camera, detection, frame_number):
    """Save detection to database"""
    Event.objects.create(
        branch=camera.branch,
        camera=camera,
        event_type=detection.get('label', 'unknown').lower(),
        confidence=detection.get('confidence', 0),
        frame_number=frame_number,
        bbox_x1=detection.get('bbox', [0, 0, 0, 0])[0],
        bbox_y1=detection.get('bbox', [0, 0, 0, 0])[1],
        bbox_x2=detection.get('bbox', [0, 0, 0, 0])[2],
        bbox_y2=detection.get('bbox', [0, 0, 0, 0])[3],
    )


@login_required
def video_feed(request, camera_id):
    """Video streaming endpoint"""
    camera = get_object_or_404(Camera, id=camera_id)
    
    user = request.user
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    return StreamingHttpResponse(
        generate_frames(camera),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


# ==================== DEVELOPER MODE / DEBUG ====================

# Store debug mode state per camera (session-like)
camera_debug_state = {}

DEVELOPER_PASSWORD = "00oo00oo"

# Cache for pose model in debug frames
_debug_pose_model = None

def get_debug_pose_model():
    """Get or create pose model for debug overlay"""
    global _debug_pose_model
    if _debug_pose_model is None:
        try:
            from ultralytics import YOLO
            from pathlib import Path
            models_dir = Path(settings.DETECTION_CONFIG['MODELS_DIR'])
            pose_model_name = settings.DETECTION_CONFIG.get('POSE_MODEL', 'yolov8s-pose.pt')
            pose_path = models_dir / pose_model_name
            if pose_path.exists():
                _debug_pose_model = YOLO(str(pose_path))
                print(f"[Debug] Loaded pose model: {pose_path}")
            else:
                _debug_pose_model = YOLO(pose_model_name)
                print(f"[Debug] Downloaded pose model: {pose_model_name}")
        except Exception as e:
            print(f"Failed to load debug pose model: {e}")
    return _debug_pose_model


def generate_debug_frames(camera):
    """Generator for video frames with debug overlay showing poses, distances, roles"""
    
    # Check if background worker is running
    worker = None
    with background_worker_lock:
        if camera.id in background_workers:
            worker = background_workers[camera.id]
            if not worker.running:
                worker = None
    
    # If background worker is running, use its frames
    if worker is not None:
        # Pre-load pose model for this debug session
        pose_model = get_debug_pose_model()
        
        while worker.running:
            # Get raw frame from worker
            frame = worker.get_current_frame(with_overlay=False)
            if frame is not None:
                # Draw our own debug overlay with poses, distances, labels
                debug_frame = draw_debug_frame(frame.copy(), camera, pose_model)
                _, buffer = cv2.imencode('.jpg', debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)  # ~30fps
        return
    
    # No background worker - make new connection with TCP transport
    import os
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000'
    cap = cv2.VideoCapture(camera.rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    
    if not cap.isOpened():
        frame = create_placeholder_frame("Cannot connect to camera")
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    pose_model = get_debug_pose_model()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            debug_frame = draw_debug_frame(frame.copy(), camera, pose_model)
            _, buffer = cv2.imencode('.jpg', debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()


def draw_debug_frame(frame, camera, pose_model=None):
    """Draw comprehensive debug overlay on frame with poses, distances, and roles"""
    h, w = frame.shape[:2]
    
    # Get cashier zone
    cashier_zone = [camera.cashier_zone_x, camera.cashier_zone_y, 
                    camera.cashier_zone_width, camera.cashier_zone_height]
    zx, zy, zw, zh = cashier_zone
    
    # Draw cashier zone with label
    cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), (0, 255, 255), 3)
    cv2.rectangle(frame, (zx, zy - 30), (zx + 150, zy), (0, 255, 255), -1)
    cv2.putText(frame, "CASHIER ZONE", (zx + 5, zy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Get pose model if not provided
    if pose_model is None:
        pose_model = get_debug_pose_model()
    
    if pose_model is None:
        cv2.putText(frame, "No pose model available", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        draw_debug_panel(frame, camera)
        return frame
    
    # Get pose confidence threshold from camera settings
    pose_conf = getattr(camera, 'pose_confidence', 0.3) or 0.3
    
    try:
        # Run pose estimation with camera's confidence threshold
        results = pose_model(frame, verbose=False, conf=pose_conf)
        
        if results and len(results) > 0:
            result = results[0]
            
            people_info = []
            
            if result.keypoints is not None and result.boxes is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()  # Get confidence scores
                
                for idx, (kpts, box, conf) in enumerate(zip(keypoints_data, boxes, confs)):
                    x1, y1, x2, y2 = map(int, box)
                    person_conf = float(conf)  # Person detection confidence
                    
                    # Determine if person is in cashier zone
                    person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    in_zone = (zx <= person_center[0] <= zx + zw and 
                               zy <= person_center[1] <= zy + zh)
                    
                    # Label as Cashier or Client with confidence
                    role = "CASHIER" if in_zone else "CLIENT"
                    color = (0, 255, 0) if in_zone else (0, 165, 255)  # Green for cashier, orange for client
                    
                    # Draw person bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw role label with confidence and black background
                    label_text = f"{role} {person_conf:.0%}"
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_w + 10, y1), (0, 0, 0), -1)
                    cv2.putText(frame, label_text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw skeleton/pose
                    draw_skeleton(frame, kpts, color)
                    
                    # Get hand positions (wrists)
                    LEFT_WRIST, RIGHT_WRIST = 9, 10
                    hands = {}
                    
                    if len(kpts) > LEFT_WRIST:
                        lw = kpts[LEFT_WRIST]
                        if len(lw) >= 3 and lw[2] >= 0.3:
                            hands['left'] = (int(lw[0]), int(lw[1]), float(lw[2]))
                            cv2.circle(frame, (int(lw[0]), int(lw[1])), 12, (255, 0, 255), -1)
                            cv2.circle(frame, (int(lw[0]), int(lw[1])), 12, (255, 255, 255), 2)
                    
                    if len(kpts) > RIGHT_WRIST:
                        rw = kpts[RIGHT_WRIST]
                        if len(rw) >= 3 and rw[2] >= 0.3:
                            hands['right'] = (int(rw[0]), int(rw[1]), float(rw[2]))
                            cv2.circle(frame, (int(rw[0]), int(rw[1])), 12, (255, 0, 255), -1)
                            cv2.circle(frame, (int(rw[0]), int(rw[1])), 12, (255, 255, 255), 2)
                    
                    people_info.append({
                        'idx': idx,
                        'role': role,
                        'in_zone': in_zone,
                        'hands': hands,
                        'bbox': (x1, y1, x2, y2)
                    })
            
            # Calculate and draw hand distances between people
            hand_touch_distance = camera.hand_touch_distance or 100
            
            for i, p1 in enumerate(people_info):
                for j, p2 in enumerate(people_info):
                    if i >= j:
                        continue
                    
                    for hand1_name, hand1_pos in p1.get('hands', {}).items():
                        for hand2_name, hand2_pos in p2.get('hands', {}).items():
                            # Calculate distance
                            dx = hand1_pos[0] - hand2_pos[0]
                            dy = hand1_pos[1] - hand2_pos[1]
                            distance = int(np.sqrt(dx*dx + dy*dy))
                            
                            # Draw line between hands
                            is_close = distance < hand_touch_distance
                            line_color = (0, 255, 0) if is_close else (0, 0, 255)
                            cv2.line(frame, (hand1_pos[0], hand1_pos[1]), 
                                    (hand2_pos[0], hand2_pos[1]), line_color, 3)
                            
                            # Draw distance label at midpoint with black background
                            mid_x = (hand1_pos[0] + hand2_pos[0]) // 2
                            mid_y = (hand1_pos[1] + hand2_pos[1]) // 2
                            
                            dist_text = f"{distance}px"
                            status = "CLOSE!" if is_close else "FAR"
                            full_text = f"{dist_text} {status}"
                            
                            # Black background for text
                            (text_w, text_h), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (mid_x - 5, mid_y - text_h - 10), 
                                         (mid_x + text_w + 10, mid_y + 5), (0, 0, 0), -1)
                            cv2.putText(frame, full_text, (mid_x, mid_y - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)
            
            # Show people count
            cv2.rectangle(frame, (w - 200, h - 40), (w - 5, h - 5), (0, 0, 0), -1)
            cv2.putText(frame, f"People: {len(people_info)}", (w - 190, h - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No people detected
            cv2.rectangle(frame, (w - 200, h - 40), (w - 5, h - 5), (0, 0, 0), -1)
            cv2.putText(frame, "No people detected", (w - 190, h - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    except Exception as e:
        cv2.putText(frame, f"Debug error: {str(e)[:50]}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw debug info panel
    draw_debug_panel(frame, camera)
    
    return frame


def draw_skeleton(frame, keypoints, color=(0, 255, 0)):
    """Draw pose skeleton on frame"""
    # COCO skeleton connections
    skeleton = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    for start_idx, end_idx in skeleton:
        if len(keypoints) > max(start_idx, end_idx):
            start = keypoints[start_idx]
            end = keypoints[end_idx]
            
            if len(start) >= 3 and len(end) >= 3 and start[2] >= 0.3 and end[2] >= 0.3:
                cv2.line(frame, (int(start[0]), int(start[1])), 
                        (int(end[0]), int(end[1])), color, 2)


def draw_debug_panel(frame, camera, cached_worker=None):
    """Draw debug info panel on frame"""
    h, w = frame.shape[:2]
    
    # Use cached worker if provided, otherwise try to get it (quick lock)
    worker = cached_worker
    if worker is None:
        with background_worker_lock:
            if camera.id in background_workers:
                worker = background_workers[camera.id]
    
    detector = worker.detector if worker else None
    
    # Panel background
    cv2.rectangle(frame, (5, 5), (400, 200), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (400, 200), (0, 255, 255), 2)
    
    y = 25
    cv2.putText(frame, "DEVELOPER DEBUG MODE", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    y += 25
    cv2.putText(frame, f"Camera: {camera.name} ({camera.camera_id})", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y += 20
    cv2.putText(frame, f"Hand Touch Distance: {camera.hand_touch_distance or 100}px", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Detector state
    if detector:
        y += 25
        cash_consec = getattr(detector.cash_detector, 'consecutive_detections', 0) if hasattr(detector, 'cash_detector') else 0
        cash_thresh = getattr(detector.cash_detector, 'min_transaction_frames', 3) if hasattr(detector, 'cash_detector') else 3
        cash_status = f"TRIGGERED!" if cash_consec >= cash_thresh else f"{cash_consec}/{cash_thresh}"
        cash_color = (0, 255, 0) if cash_consec >= cash_thresh else (255, 255, 255)
        cv2.putText(frame, f"Cash Detection: {cash_status}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cash_color, 1)
        
        y += 20
        violence_consec = getattr(detector.violence_detector, 'consecutive_violence', 0) if hasattr(detector, 'violence_detector') else 0
        violence_thresh = getattr(detector.violence_detector, 'min_violence_frames', 10) if hasattr(detector, 'violence_detector') else 10
        cv2.putText(frame, f"Violence Detection: {violence_consec}/{violence_thresh}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        fire_consec = getattr(detector.fire_detector, 'consecutive_fire', 0) if hasattr(detector, 'fire_detector') else 0
        fire_thresh = getattr(detector.fire_detector, 'min_fire_frames', 10) if hasattr(detector, 'fire_detector') else 10
        cv2.putText(frame, f"Fire Detection: {fire_consec}/{fire_thresh}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 25
        frame_count = getattr(detector, 'frame_count', 0)
        cv2.putText(frame, f"Frame: {frame_count}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        y += 20
        last_tx = getattr(detector.cash_detector, 'last_transaction_frame', 0) if hasattr(detector, 'cash_detector') else 0
        cooldown = getattr(detector.cash_detector, 'transaction_cooldown', 45) if hasattr(detector, 'cash_detector') else 45
        frames_since = frame_count - last_tx
        cv2.putText(frame, f"Frames since last event: {frames_since} (cooldown: {cooldown})", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    else:
        y += 25
        cv2.putText(frame, "No detector attached", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    # Legend
    cv2.rectangle(frame, (w - 180, 5), (w - 5, 85), (0, 0, 0), -1)
    cv2.putText(frame, "Legend:", (w - 170, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(frame, (w - 170, 35), (w - 155, 50), (0, 255, 0), -1)
    cv2.putText(frame, "Cashier", (w - 150, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(frame, (w - 170, 55), (w - 155, 70), (0, 165, 255), -1)
    cv2.putText(frame, "Client", (w - 150, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


@login_required
def video_feed_debug(request, camera_id):
    """Debug video streaming endpoint with pose/distance overlay"""
    camera = get_object_or_404(Camera, id=camera_id)
    
    user = request.user
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    # Check if debug mode is enabled for this camera
    if camera.id not in camera_debug_state or not camera_debug_state[camera.id].get('enabled'):
        return JsonResponse({'error': 'Developer mode not enabled'}, status=403)
    
    return StreamingHttpResponse(
        generate_debug_frames(camera),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def api_verify_dev_password(request, camera_id):
    """Verify developer password and enable debug mode"""
    camera = get_object_or_404(Camera, id=camera_id)
    
    user = request.user
    if not user.is_admin() and camera.branch not in get_user_branches(user):
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    data = json.loads(request.body)
    password = data.get('password', '')
    
    if password == DEVELOPER_PASSWORD:
        camera_debug_state[camera.id] = {
            'enabled': True,
            'unlocked_at': timezone.now().isoformat(),
            'unlocked_by': user.username
        }
        return JsonResponse({
            'success': True,
            'message': 'Developer mode unlocked'
        })
    else:
        return JsonResponse({
            'success': False,
            'error': 'Incorrect password'
        }, status=401)


@login_required
@csrf_exempt
@require_http_methods(["POST"])
def api_lock_dev_mode(request, camera_id):
    """Lock developer mode for a camera"""
    camera = get_object_or_404(Camera, id=camera_id)
    
    if camera.id in camera_debug_state:
        del camera_debug_state[camera.id]
    
    return JsonResponse({'success': True, 'message': 'Developer mode locked'})


@login_required
@require_http_methods(["GET"])
def api_dev_mode_status(request, camera_id):
    """Get developer mode status for a camera"""
    camera = get_object_or_404(Camera, id=camera_id)
    
    is_enabled = camera.id in camera_debug_state and camera_debug_state[camera.id].get('enabled', False)
    
    return JsonResponse({
        'enabled': is_enabled,
        'unlocked_at': camera_debug_state.get(camera.id, {}).get('unlocked_at'),
        'unlocked_by': camera_debug_state.get(camera.id, {}).get('unlocked_by')
    })


@login_required
@require_http_methods(["GET"])
def api_detection_debug_info(request, camera_id):
    """Get live detection debug information"""
    camera = get_object_or_404(Camera, id=camera_id)
    
    # Check if debug mode is enabled
    if camera.id not in camera_debug_state or not camera_debug_state[camera.id].get('enabled'):
        return JsonResponse({'error': 'Developer mode not enabled'}, status=403)
    
    # Get detector info
    detector = camera_detectors.get(camera.id)
    
    if not detector:
        return JsonResponse({
            'error': 'Detector not initialized',
            'detector_available': DETECTOR_AVAILABLE
        })
    
    # Collect debug info
    info = {
        'frame_count': getattr(detector, 'frame_count', 0),
        'debug_mode': getattr(detector, 'debug_mode', False),
        'cash_detection': {
            'enabled': detector.detect_cash,
            'consecutive_frames': getattr(detector.cash_detector, 'consecutive_detections', 0),
            'min_frames_required': getattr(detector.cash_detector, 'min_transaction_frames', 3),
            'last_transaction_frame': getattr(detector.cash_detector, 'last_transaction_frame', 0),
            'cooldown': getattr(detector.cash_detector, 'transaction_cooldown', 45),
            'hand_touch_distance': getattr(detector.cash_detector, 'hand_touch_distance', 100),
            'cashier_zone': detector.cash_detector.cashier_zone,
        },
        'violence_detection': {
            'enabled': detector.detect_violence,
            'consecutive_frames': getattr(detector.violence_detector, 'consecutive_violence', 0),
            'min_frames_required': getattr(detector.violence_detector, 'min_violence_frames', 10),
        },
        'fire_detection': {
            'enabled': detector.detect_fire,
            'consecutive_frames': getattr(detector.fire_detector, 'consecutive_fire', 0),
            'min_frames_required': getattr(detector.fire_detector, 'min_fire_frames', 10),
        },
        'recent_alerts': detector.alerts_history[-10:] if hasattr(detector, 'alerts_history') else [],
    }
    
    return JsonResponse(info)


# ==================== USER MANAGEMENT ====================

@login_required
@require_http_methods(["GET", "POST"])
def api_users(request):
    """API for users"""
    user = request.user
    if not user.is_admin():
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    if request.method == 'GET':
        users = User.objects.all()
        data = [{
            'id': u.id,
            'username': u.username,
            'email': u.email,
            'first_name': u.first_name,
            'last_name': u.last_name,
            'role': u.role,
            'is_active': u.is_active,
        } for u in users]
        return JsonResponse({'users': data})
    
    elif request.method == 'POST':
        data = json.loads(request.body)
        
        # Create user
        new_user = User.objects.create_user(
            username=data.get('username'),
            email=data.get('email', ''),
            password=data.get('password', 'password123'),
            first_name=data.get('first_name', ''),
            last_name=data.get('last_name', ''),
            role=data.get('role', 'staff'),
        )
        
        # Add to branch if specified
        if 'branch_id' in data:
            branch = get_object_or_404(Branch, id=data['branch_id'])
            branch.managers.add(new_user)
        
        return JsonResponse({
            'success': True,
            'user': {
                'id': new_user.id,
                'username': new_user.username,
            }
        })


@login_required
@require_http_methods(["GET", "PUT", "DELETE"])
def api_user_detail(request, user_id):
    """API for single user"""
    current_user = request.user
    if not current_user.is_admin():
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    target_user = get_object_or_404(User, id=user_id)
    
    if request.method == 'GET':
        return JsonResponse({
            'id': target_user.id,
            'username': target_user.username,
            'email': target_user.email,
            'first_name': target_user.first_name,
            'last_name': target_user.last_name,
            'role': target_user.role,
            'is_active': target_user.is_active,
            'branches': [{'id': b.id, 'name': b.name} for b in target_user.managed_branches.all()],
        })
    
    elif request.method == 'PUT':
        data = json.loads(request.body)
        
        target_user.email = data.get('email', target_user.email)
        target_user.first_name = data.get('first_name', target_user.first_name)
        target_user.last_name = data.get('last_name', target_user.last_name)
        target_user.role = data.get('role', target_user.role)
        target_user.is_active = data.get('is_active', target_user.is_active)
        
        if 'password' in data and data['password']:
            target_user.set_password(data['password'])
        
        target_user.save()
        return JsonResponse({'success': True})
    
    elif request.method == 'DELETE':
        target_user.delete()
        return JsonResponse({'success': True})


# ==================== REGION MANAGEMENT ====================

@login_required
@require_http_methods(["GET", "POST"])
def api_regions(request):
    """API for regions"""
    user = request.user
    
    if request.method == 'GET':
        regions = Region.objects.all()
        data = [{
            'id': r.id,
            'name': r.name,
            'code': r.code,
            'branch_count': r.branches.count(),
        } for r in regions]
        return JsonResponse({'regions': data})
    
    elif request.method == 'POST':
        if not user.is_admin():
            return JsonResponse({'error': 'Permission denied'}, status=403)
        
        data = json.loads(request.body)
        region = Region.objects.create(
            name=data.get('name'),
            code=data.get('code', ''),
        )
        
        return JsonResponse({
            'success': True,
            'region': {
                'id': region.id,
                'name': region.name,
            }
        })


@login_required
@require_http_methods(["GET", "PUT", "DELETE"])
def api_region_detail(request, region_id):
    """API for single region"""
    user = request.user
    region = get_object_or_404(Region, id=region_id)
    
    if request.method == 'GET':
        return JsonResponse({
            'id': region.id,
            'name': region.name,
            'code': region.code,
            'branch_count': region.branches.count(),
        })
    
    elif request.method == 'PUT':
        if not user.is_admin():
            return JsonResponse({'error': 'Permission denied'}, status=403)
        
        data = json.loads(request.body)
        region.name = data.get('name', region.name)
        region.code = data.get('code', region.code)
        region.save()
        return JsonResponse({'success': True})
    
    elif request.method == 'DELETE':
        if not user.is_admin():
            return JsonResponse({'error': 'Permission denied'}, status=403)
        region.delete()
        return JsonResponse({'success': True})


# ==================== BULK OPERATIONS ====================

@login_required
@require_http_methods(["POST"])
def api_bulk_delete_events(request):
    """API for bulk deleting events"""
    user = request.user
    if not user.is_admin():
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    data = json.loads(request.body)
    event_ids = data.get('event_ids', [])
    
    if not event_ids:
        # Delete by filter if no IDs provided
        filters = {}
        if 'branch_id' in data:
            filters['branch_id'] = data['branch_id']
        if 'event_type' in data:
            filters['event_type'] = data['event_type']
        if 'status' in data:
            filters['status'] = data['status']
        if 'before_date' in data:
            filters['created_at__lt'] = data['before_date']
        
        if filters:
            deleted, _ = Event.objects.filter(**filters).delete()
        else:
            return JsonResponse({'error': 'No filter or event IDs provided'}, status=400)
    else:
        deleted, _ = Event.objects.filter(id__in=event_ids).delete()
    
    return JsonResponse({'success': True, 'deleted_count': deleted})


@login_required
@require_http_methods(["POST"])
def api_bulk_update_events(request):
    """API for bulk updating event status"""
    user = request.user
    
    data = json.loads(request.body)
    event_ids = data.get('event_ids', [])
    new_status = data.get('status')
    
    if not event_ids or not new_status:
        return JsonResponse({'error': 'event_ids and status are required'}, status=400)
    
    user_branches = get_user_branches(user)
    events = Event.objects.filter(id__in=event_ids, branch__in=user_branches)
    
    updated = events.update(
        status=new_status,
        reviewed_by=user,
        reviewed_at=timezone.now()
    )
    
    return JsonResponse({'success': True, 'updated_count': updated})


# ==================== LANGUAGE ====================

def set_language(request):
    """Set the user's preferred language"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            lang = data.get('lang', 'ko')
        except:
            lang = request.POST.get('lang', 'ko')
        
        # Validate language
        if lang not in ['ko', 'en']:
            lang = 'ko'
        
        # Store in session
        request.session['lang'] = lang
        
        # Create response
        response = JsonResponse({'success': True, 'lang': lang})
        
        # Also set cookie
        response.set_cookie('lang', lang, max_age=365*24*60*60)  # 1 year
        
        return response
    
    return JsonResponse({'error': 'POST required'}, status=405)


def get_translations_api(request):
    """Get translations for current language"""
    lang = request.session.get('lang', request.COOKIES.get('lang', 'ko'))
    translations = get_translation(lang)
    return JsonResponse(translations)


# ==================== BACKGROUND WORKERS ====================

class BackgroundCameraWorker:
    """Background worker for continuous camera detection"""
    
    def __init__(self, camera, models_dir, output_dir):
        self.camera_id = camera.id
        self.camera_code = camera.camera_id
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.running = False
        self.thread = None
        self.detection_thread = None  # Separate detection thread
        self.clip_save_thread = None  # Separate clip saving thread
        self.detector = None
        self.last_event_time = {}
        self.event_cooldown = 15  # seconds between events (reduced from 30)
        
        # RAW frame buffer (every 2nd frame) for smooth clips
        self.raw_frame_buffer = []
        self.raw_buffer_lock = threading.Lock()
        self.clip_buffer_size = 450  # 30 seconds at 15fps (every 2nd frame of 30fps)
        self.clip_duration_sec = 30  # clip duration in seconds
        
        self.frame_count = 0
        self.status = 'stopped'
        self.last_error = None
        
        # Track when clips were saved to avoid duplicates
        self.last_clip_time = {}  # event_type -> timestamp
        self.last_clip_path = {}  # event_type -> clip_path
        
        # Pending clip save info (detection triggers this, clip thread saves later)
        self.pending_clip_event = None  # (event_type, confidence, bbox, trigger_frame, trigger_time)
        self.pending_clip_lock = threading.Lock()
        
        # Shared frame for live viewing (no need to reconnect!)
        self.current_frame = None
        self.current_frame_with_overlay = None
        self.frame_lock = threading.Lock()
        
        # Frame queue for detection processing (non-blocking)
        self.detection_queue = None  # Will be queue.Queue
        self.detection_queue_maxsize = 5  # Small queue to prevent memory buildup
        
        # Uptime tracking
        self.start_time = None
        self.events_detected = 0
        self.frames_processed = 0
        
        # Stream fps (detected from camera)
        self.stream_fps = 30  # Default, will be updated from camera
    
    def get_uptime(self):
        """Get worker uptime as formatted string"""
        if not self.start_time:
            return "Not started"
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_stats(self):
        """Get worker statistics"""
        return {
            'uptime': self.get_uptime(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'events_detected': self.events_detected,
            'frames_processed': self.frames_processed,
            'running': self.running,
        }
    
    def _create_rtsp_capture(self, rtsp_url):
        """Create RTSP capture with optimized settings for stability.
        
        Uses TCP transport to avoid RTP packet ordering issues (bad cseq errors).
        Sets buffer sizes and timeouts for better stream handling.
        """
        import os
        
        # Set FFmpeg options via environment for better RTSP handling
        # - rtsp_transport=tcp: Use TCP instead of UDP to avoid packet loss
        # - stimeout=60000000: 60 second socket timeout (in microseconds) - prevents premature timeout
        # - max_delay=1000000: Max delay 1 second
        # - fflags=nobuffer+discardcorrupt: Reduce buffering, discard corrupt frames
        # - analyzeduration=2000000: Analyze for 2 seconds max
        # - probesize=2000000: Probe size 2MB
        # - buffer_size=4096000: 4MB buffer for network stability
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;60000000|max_delay;1000000|fflags;nobuffer+discardcorrupt|analyzeduration;2000000|probesize;2000000|buffer_size;4096000'
        
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # Set additional capture properties - balanced timeouts
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)  # 30s connection timeout
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)  # 15s read timeout 
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)  # Slightly larger buffer for stability
        
        return cap
    
    def get_current_frame(self, with_overlay=True):
        """Get the current frame for live viewing (shared from background worker)
        
        Returns reference to frame - caller should not modify!
        """
        with self.frame_lock:
            if with_overlay and self.current_frame_with_overlay is not None:
                return self.current_frame_with_overlay
            elif self.current_frame is not None:
                return self.current_frame
            return None
    
    def get_camera(self):
        try:
            return Camera.objects.get(id=self.camera_id)
        except Camera.DoesNotExist:
            return None
    
    def create_detector(self, camera):
        if not DETECTOR_AVAILABLE:
            return None
        
        zone = camera.get_cashier_zone()
        config = {
            'models_dir': str(self.models_dir),
            # Model names from settings
            'pose_model': settings.DETECTION_CONFIG.get('POSE_MODEL', 'yolov8s-pose.pt'),
            'yolo_model': settings.DETECTION_CONFIG.get('YOLO_MODEL', 'yolov8s.pt'),
            'fire_model': settings.DETECTION_CONFIG.get('FIRE_MODEL', 'fire_smoke_yolov8.pt'),
            # Detection settings
            'cashier_zone': [zone['x'], zone['y'], zone['width'], zone['height']],
            'hand_touch_distance': camera.hand_touch_distance,
            'pose_confidence': 0.5,
            'min_transaction_frames': 1,  # Immediate cash detection
            'fire_confidence': camera.fire_confidence,
            'min_fire_frames': 3,
            'violence_confidence': camera.violence_confidence,
            'min_violence_frames': 10,
            'detect_cash': camera.detect_cash,
            'detect_violence': camera.detect_violence,
            'detect_fire': camera.detect_fire,
            'cash_confidence': camera.cash_confidence,
        }
        return UnifiedDetector(config)
    
    def save_event(self, camera, event_type, confidence, frame_number, bbox=None, clip_path=None, thumbnail_path=None, metadata=None):
        """Save event to database with detection metadata as JSON file."""
        import json
        from datetime import datetime
        try:
            # Build metadata JSON with detection parameters
            event_metadata = metadata or {}
            
            # Add timestamp
            timestamp = datetime.now()
            event_metadata['timestamp'] = timestamp.isoformat()
            
            # Add standard detection info
            event_metadata.update({
                'frame_number': frame_number,
                'confidence': round(confidence, 3),
                'bbox': bbox,
                'camera_id': camera.camera_id,
                'camera_name': camera.name,
                'event_type': event_type,
                'clip_path': clip_path,
                'thumbnail_path': thumbnail_path,
            })
            
            # Add detector-specific info if available
            if self.detector:
                if event_type == 'cash' and hasattr(self.detector, 'cash_detector'):
                    cd = self.detector.cash_detector
                    event_metadata['cash_detection'] = {
                        'hand_touch_distance_threshold': getattr(cd, 'hand_touch_distance', 100),
                        'cashier_zone': getattr(cd, 'cashier_zone', []),
                        'pose_confidence': getattr(cd, 'pose_confidence', 0.5),
                    }
                    # Include detection-specific metadata if available in main metadata
                    if metadata:
                        # Cashier info
                        if 'cashier' in metadata:
                            event_metadata['cashier'] = metadata['cashier']
                        # Customer info
                        if 'customer' in metadata:
                            event_metadata['customer'] = metadata['customer']
                        # Actual distance measured
                        if 'distance' in metadata:
                            event_metadata['measured_hand_distance'] = metadata['distance']
                        # Interaction point
                        if 'interaction_point' in metadata:
                            event_metadata['interaction_point'] = metadata['interaction_point']
                elif event_type == 'violence' and hasattr(self.detector, 'violence_detector'):
                    vd = self.detector.violence_detector
                    event_metadata['violence_detection'] = {
                        'min_violence_frames': getattr(vd, 'min_violence_frames', 20),
                        'violence_confidence': getattr(vd, 'violence_confidence', 0.5),
                    }
                elif event_type == 'fire' and hasattr(self.detector, 'fire_detector'):
                    fd = self.detector.fire_detector
                    event_metadata['fire_detection'] = {
                        'min_fire_frames': getattr(fd, 'min_fire_frames', 15),
                        'fire_confidence': getattr(fd, 'fire_confidence', 0.5),
                    }
            
            # Save JSON file to media/json folder
            json_dir = os.path.join(settings.MEDIA_ROOT, 'json')
            os.makedirs(json_dir, exist_ok=True)
            
            # Generate JSON filename matching clip pattern
            json_filename = f"{event_type}_{camera.camera_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            json_path = os.path.join(json_dir, json_filename)
            
            # Write JSON file with pretty formatting
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(event_metadata, f, indent=2, ensure_ascii=False)
            
            # Store relative path for database
            json_relative_path = f"json/{json_filename}"
            print(f"[JSON] Saved metadata: {json_relative_path}")
            
            event = Event.objects.create(
                branch=camera.branch,
                camera=camera,
                event_type=event_type,
                confidence=confidence,
                frame_number=frame_number,
                bbox_x1=bbox[0] if bbox else 0,
                bbox_y1=bbox[1] if bbox else 0,
                bbox_x2=bbox[2] if bbox else 0,
                bbox_y2=bbox[3] if bbox else 0,
                clip_path=clip_path,
                thumbnail_path=thumbnail_path,
                metadata=json_relative_path,  # Store path to JSON file
            )
            print(f"[DB] Saved event: {event_type} (id={event.id}) with JSON: {json_relative_path}")
            return event
        except Exception as e:
            print(f"[DB] Error saving event: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_clip(self, frames, camera, detection_type, fps=30):
        """Save video clip directly to media folder for web access.
        
        Uses H.264 encoding via ffmpeg for browser compatibility.
        """
        if not frames or len(frames) == 0:
            print(f"[Clip] No frames to save")
            return None
        
        import cv2
        import subprocess
        import uuid
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:6]  # Add unique ID to prevent conflicts
        
        # Save to media/clips for web access
        clip_dir = Path(settings.MEDIA_ROOT) / 'clips'
        clip_dir.mkdir(parents=True, exist_ok=True)
        
        thumb_dir = Path(settings.MEDIA_ROOT) / 'thumbnails'
        thumb_dir.mkdir(parents=True, exist_ok=True)
        
        temp_filename = f"{camera.camera_id}_{detection_type}_{timestamp}_{unique_id}_temp.avi"
        final_filename = f"{camera.camera_id}_{detection_type}_{timestamp}.mp4"
        
        temp_path = clip_dir / temp_filename
        final_path = clip_dir / final_filename
        
        height, width = frames[0].shape[:2]
        
        # Use MJPG codec for temp file (reliable, fast)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(temp_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"[Clip] Failed to open video writer")
            return None
        
        # Write frames with detection label
        frame_count = 0
        for frame in frames:
            if frame is None:
                continue
            # Add detection type label
            label = f"{detection_type.upper()} DETECTED"
            cv2.rectangle(frame, (10, 10), (250, 45), (0, 0, 0), -1)
            color = {'cash': (0, 255, 0), 'violence': (0, 0, 255), 'fire': (0, 165, 255)}.get(detection_type, (255, 255, 255))
            cv2.putText(frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            out.write(frame)
            frame_count += 1
        
        out.release()
        out = None  # Ensure handle is released
        
        print(f"[Clip] Wrote {frame_count} frames to temp file")
        
        # Convert to H.264 MP4 using ffmpeg
        ffmpeg_path = settings.DETECTION_CONFIG.get('FFMPEG_PATH', 'ffmpeg')
        try:
            result = subprocess.run([
                ffmpeg_path, '-y', '-i', str(temp_path),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-r', str(fps),
                str(final_path)
            ], capture_output=True, timeout=180)
            
            # Clean up temp file with retry for Windows file locking
            self._safe_delete(temp_path)
            
            if result.returncode == 0 and final_path.exists():
                print(f"[Clip] Saved: {final_path} ({final_path.stat().st_size / 1024:.1f} KB)")
            else:
                print(f"[Clip] FFmpeg error: {result.stderr.decode()[:300]}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"[Clip] FFmpeg timeout")
            self._safe_delete(temp_path)
            return None
        except FileNotFoundError:
            print(f"[Clip] FFmpeg not found")
            self._safe_delete(temp_path)
            return None
        except Exception as e:
            print(f"[Clip] Error: {e}")
            self._safe_delete(temp_path)
            return None
        
        # Save thumbnail from last frame
        thumb_filename = f"{camera.camera_id}_{detection_type}_{timestamp}.jpg"
        thumb_path = thumb_dir / thumb_filename
        
        thumb_frame = frames[-1].copy()
        label = f"{detection_type.upper()}"
        color = {'cash': (0, 255, 0), 'violence': (0, 0, 255), 'fire': (0, 165, 255)}.get(detection_type, (255, 255, 255))
        cv2.rectangle(thumb_frame, (10, 10), (150, 45), (0, 0, 0), -1)
        cv2.putText(thumb_frame, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imwrite(str(thumb_path), thumb_frame)
        
        print(f"[Clip] Thumbnail: {thumb_path}")
        
        return f'/media/clips/{final_filename}', f'/media/thumbnails/{thumb_filename}'
    
    def _safe_delete(self, file_path, retries=3, delay=0.5):
        """Safely delete a file with retries for Windows file locking issues."""
        import time as time_module
        for attempt in range(retries):
            try:
                if file_path.exists():
                    file_path.unlink()
                return True
            except PermissionError:
                if attempt < retries - 1:
                    time_module.sleep(delay)
                else:
                    # File will be deleted on next cleanup
                    print(f"[Clip] Note: Temp file will be cleaned up later: {file_path.name}")
                    return False
            except Exception:
                return False
        return False
    
    def _cleanup_temp_files(self):
        """Clean up any leftover temp files from previous runs."""
        try:
            clip_dir = Path(settings.MEDIA_ROOT) / 'clips'
            if clip_dir.exists():
                for temp_file in clip_dir.glob('*_temp.avi'):
                    try:
                        temp_file.unlink()
                        print(f"[Clip] Cleaned up old temp file: {temp_file.name}")
                    except Exception:
                        pass
        except Exception:
            pass
    
    def run(self):
        """Main frame reading loop - ONLY reads frames, never blocks on detection.
        
        Detection happens in a separate thread (run_detection) to prevent stream freezing.
        """
        import cv2
        import queue
        from django.db import connection
        connection.close()
        
        # Clean up any leftover temp files from previous runs
        self._cleanup_temp_files()
        
        camera = self.get_camera()
        if not camera:
            self.status = 'error'
            self.last_error = 'Camera not found'
            return
        
        # Set start time for uptime tracking
        self.start_time = datetime.now()
        
        self.status = 'starting'
        self.detector = self.create_detector(camera)
        if not self.detector:
            self.status = 'error'
            self.last_error = 'Detector not available'
            return
        
        # Initialize detection queue
        self.detection_queue = queue.Queue(maxsize=self.detection_queue_maxsize)
        
        # Start detection thread (separate from frame reading)
        self.detection_thread = threading.Thread(target=self.run_detection, daemon=True)
        self.detection_thread.start()
        print(f"[Worker] Started detection thread for camera {camera.camera_id}")
        
        # Start clip saving thread (separate from detection)
        self.clip_save_thread = threading.Thread(target=self.run_clip_saver, daemon=True)
        self.clip_save_thread.start()
        print(f"[Worker] Started clip saver thread for camera {camera.camera_id}")
        
        # Try to connect with retries
        cap = None
        max_connect_retries = 5
        for attempt in range(max_connect_retries):
            print(f"[Worker] Attempting to connect to stream (attempt {attempt + 1}/{max_connect_retries})...")
            cap = self._create_rtsp_capture(camera.rtsp_url)
            if cap.isOpened():
                # Try to read a test frame to confirm stream is working
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"[Worker] Successfully connected to stream: {camera.rtsp_url}")
                    break
                else:
                    print(f"[Worker] Stream opened but no frames available, retrying...")
                    cap.release()
                    cap = None
            else:
                print(f"[Worker] Failed to open stream, retrying in 5s...")
            time.sleep(5)
        
        if cap is None or not cap.isOpened():
            self.status = 'error'
            self.last_error = f'Cannot open stream after {max_connect_retries} attempts: {camera.rtsp_url}'
            camera.status = 'offline'
            camera.save()
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.stream_fps = fps  # Store for clip saver
        self.status = 'running'
        camera.status = 'online'
        camera.last_connected = timezone.now()
        camera.save()
        
        consecutive_failures = 0
        max_failures = 20  # Max consecutive read failures before reconnect (increased)
        frame_count = 0
        last_buffer_frame = 0  # Track when we last buffered a frame
        last_success_time = time.time()  # Track last successful frame read
        
        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                
                # Check if we've been failing for too long (30 seconds)
                time_since_success = time.time() - last_success_time
                
                if consecutive_failures >= max_failures or time_since_success > 30:
                    self.status = 'reconnecting'
                    self.last_error = f'Stream lost ({consecutive_failures} failures, {time_since_success:.0f}s), reconnecting...'
                    print(f"[Worker] {self.last_error}")
                    cap.release()
                    time.sleep(3)
                    cap = self._create_rtsp_capture(camera.rtsp_url)
                    if cap.isOpened():
                        # Test frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            self.status = 'running'
                            self.last_error = None
                            consecutive_failures = 0
                            last_success_time = time.time()
                            print(f"[Worker] Reconnected successfully")
                continue
            
            # Successful frame read
            consecutive_failures = 0
            last_success_time = time.time()
            frame_count += 1
            self.frame_count = frame_count
            
            # Update current frame for live viewing (minimal lock time)
            with self.frame_lock:
                self.current_frame = frame
                if self.current_frame_with_overlay is None:
                    self.current_frame_with_overlay = frame
            
            # Buffer every 2nd frame for clips (reduces memory, still smooth)
            if frame_count % 2 == 0:
                with self.raw_buffer_lock:
                    self.raw_frame_buffer.append(frame.copy())
                    if len(self.raw_frame_buffer) > self.clip_buffer_size:
                        self.raw_frame_buffer.pop(0)
            
            # Send every 4th frame to detection (reduces CPU load)
            if frame_count % 4 == 0:
                try:
                    self.detection_queue.put_nowait((frame, frame_count, fps))
                except:
                    pass  # Queue full, skip
        
        cap.release()
        
        # Stop detection thread
        if self.detection_queue:
            try:
                self.detection_queue.put_nowait(None)  # Signal to stop
            except:
                pass
        
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        
        self.status = 'stopped'
        camera = self.get_camera()
        if camera:
            camera.status = 'offline'
            camera.save()
    
    def run_detection(self):
        """Detection processing loop - runs in separate thread.
        
        Processes frames from queue without blocking frame reading.
        """
        from django.db import connection
        connection.close()
        
        camera = self.get_camera()
        if not camera:
            return
        
        last_settings_check = time.time()
        
        print(f"[Detection] Started detection loop for camera {camera.camera_id}")
        
        while self.running:
            try:
                # Wait for frame from queue (with timeout to check running flag)
                item = self.detection_queue.get(timeout=1.0)
                
                if item is None:  # Stop signal
                    break
                
                frame, frame_count, fps = item
                
            except Exception:
                # Timeout or empty queue - just continue
                continue
            
            # Reload camera settings periodically
            if time.time() - last_settings_check > 30:
                camera = self.get_camera()
                if camera and self.detector:
                    self.detector.detect_cash = camera.detect_cash
                    self.detector.detect_violence = camera.detect_violence
                    self.detector.detect_fire = camera.detect_fire
                last_settings_check = time.time()
            
            try:
                # Process frame WITH overlay
                result = self.detector.process_frame(frame.copy(), draw_overlay=True)
                frame_with_overlay = result.get('frame', frame)
                
                # Update overlay frame for live viewing
                with self.frame_lock:
                    self.current_frame_with_overlay = frame_with_overlay
                
                # Track frames processed
                self.frames_processed += 1
                
            except Exception as e:
                self.last_error = f"Detection error: {str(e)}"
                continue
            
            # Handle detections - trigger delayed clip save
            if result.get('detections'):
                for det in result['detections']:
                    det_label = det.get('label', '').lower()
                    confidence = det.get('confidence', 0)
                    bbox = det.get('bbox')
                    
                    if 'cash' in det_label:
                        event_type = 'cash'
                    elif 'violence' in det_label:
                        event_type = 'violence'
                    elif 'fire' in det_label:
                        event_type = 'fire'
                    else:
                        continue
                    
                    # Check cooldown - skip if we saved same event type recently
                    now = datetime.now()
                    last_time = self.last_event_time.get(event_type)
                    if last_time and (now - last_time).total_seconds() < self.event_cooldown:
                        # Still in cooldown, skip this detection (will merge with existing clip)
                        continue
                    
                    # Get full detection metadata for JSON output
                    det_metadata = det.get('metadata', {})
                    
                    # Schedule delayed clip save (wait for buffer to fill with 30s of frames)
                    with self.pending_clip_lock:
                        if self.pending_clip_event is None:
                            self.pending_clip_event = {
                                'event_type': event_type,
                                'confidence': confidence,
                                'bbox': bbox,
                                'trigger_time': now,
                                'fps': fps,
                                'distance': det_metadata.get('distance'),
                                'distance_threshold': det_metadata.get('distance_threshold'),
                                'cashier': det_metadata.get('cashier'),
                                'customer': det_metadata.get('customer'),
                                'interaction_point': det_metadata.get('interaction_point'),
                                'people_count': det_metadata.get('people_count'),
                            }
                            print(f"[Clip] Detection triggered - will save 30s clip after buffer fills")
                    
                    # Update last event time immediately
                    self.last_event_time[event_type] = now
        
        print(f"[Detection] Stopped detection loop for camera {camera.camera_id}")
    
    def run_clip_saver(self):
        """Clip saving loop - runs in separate thread.
        
        Waits for pending clip events and saves them after buffer has 30 seconds of frames.
        This ensures smooth video clips without blocking stream or detection.
        """
        from django.db import connection
        connection.close()
        
        camera = self.get_camera()
        if not camera:
            return
        
        print(f"[ClipSaver] Started for camera {camera.camera_id}")
        
        while self.running:
            # Check if there's a pending clip to save
            pending = None
            with self.pending_clip_lock:
                if self.pending_clip_event is not None:
                    pending = self.pending_clip_event.copy()
            
            if pending is None:
                # No pending clip, sleep and check again
                time.sleep(0.5)
                continue
            
            # Wait for buffer to have enough frames (30 seconds worth)
            # Check buffer size
            buffer_size = 0
            with self.raw_buffer_lock:
                buffer_size = len(self.raw_frame_buffer)
            
            # We buffer every 2nd frame, so effective fps is half of stream fps
            stream_fps = self.stream_fps if self.stream_fps > 0 else 30
            buffer_fps = stream_fps / 2  # Every 2nd frame
            target_frames = int(self.clip_duration_sec * buffer_fps)
            
            # If buffer has at least 10 seconds, save what we have
            min_frames = int(10 * buffer_fps)  # At least 10 seconds
            
            if buffer_size < min_frames:
                # Buffer too small, wait more
                time.sleep(1)
                continue
            
            # Use whatever frames we have (up to 30 seconds)
            frames_to_use = min(buffer_size, target_frames)
            
            # Copy frames to save
            frames_to_save = []
            with self.raw_buffer_lock:
                frames_to_save = self.raw_frame_buffer[-frames_to_use:].copy()
            
            print(f"[ClipSaver] Saving {len(frames_to_save)} frames ({len(frames_to_save)/buffer_fps:.1f}s) for {pending['event_type']}")
            
            # Verify frames are valid (not empty)
            valid_frames = [f for f in frames_to_save if f is not None and f.size > 0]
            if len(valid_frames) < min_frames:
                print(f"[ClipSaver] Not enough valid frames ({len(valid_frames)}/{min_frames}), skipping")
                with self.pending_clip_lock:
                    self.pending_clip_event = None
                continue
            
            # Save clip with correct fps (half of stream fps since we buffer every 2nd frame)
            try:
                paths = self.save_clip(valid_frames, camera, pending['event_type'], buffer_fps)
                
                # Only save event if clip was successfully created
                if paths and paths[0]:
                    clip_path = paths[0]
                    thumb_path = paths[1] if len(paths) > 1 else None
                    
                    # Build comprehensive metadata including cashier/customer positions
                    event_metadata = {
                        'distance': pending.get('distance'),
                        'distance_threshold': pending.get('distance_threshold'),
                        'trigger_time': pending['trigger_time'].isoformat(),
                        'frames_saved': len(valid_frames),
                        'duration_sec': round(len(valid_frames) / buffer_fps, 1),
                        'people_count': pending.get('people_count'),
                    }
                    
                    # Add cashier position if available
                    if pending.get('cashier'):
                        event_metadata['cashier'] = pending['cashier']
                    
                    # Add customer position if available  
                    if pending.get('customer'):
                        event_metadata['customer'] = pending['customer']
                    
                    # Add interaction point if available
                    if pending.get('interaction_point'):
                        event_metadata['interaction_point'] = pending['interaction_point']
                    
                    # Save event to database with metadata
                    self.save_event(
                        camera,
                        pending['event_type'],
                        pending['confidence'],
                        self.frame_count,
                        pending['bbox'],
                        clip_path,
                        thumb_path,
                        metadata=event_metadata
                    )
                    self.events_detected += 1
                    print(f"[ClipSaver] Saved clip: {clip_path}")
                else:
                    print(f"[ClipSaver] Clip save failed, not creating event")
                
            except Exception as e:
                print(f"[ClipSaver] Error saving clip: {e}")
                import traceback
                traceback.print_exc()
            
            # Clear pending clip
            with self.pending_clip_lock:
                self.pending_clip_event = None
        
        print(f"[ClipSaver] Stopped for camera {camera.camera_id}")
    
    def start(self):
        if self.running:
            return False
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return True
    
    def stop(self):
        self.running = False
        
        # Signal detection thread to stop
        if self.detection_queue:
            try:
                self.detection_queue.put_nowait(None)
            except:
                pass
        
        # Wait for detection thread
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        
        # Wait for clip saver thread
        if self.clip_save_thread:
            self.clip_save_thread.join(timeout=5)
        
        # Wait for main thread
        if self.thread:
            self.thread.join(timeout=5)
        return True


@login_required
@require_http_methods(['POST'])
def start_background_worker(request, camera_id):
    """Start background worker for a camera (threading-based for Gunicorn)"""
    if not request.user.is_admin():
        return JsonResponse({'error': 'Admin only'}, status=403)
    
    camera = get_object_or_404(Camera, id=camera_id)
    
    with background_worker_lock:
        # Check if worker already exists
        if camera_id in background_workers:
            worker = background_workers[camera_id]
            if worker.running:
                return JsonResponse({'error': 'Worker already running', 'status': 'running'})
        
        # Create new worker thread
        models_dir = settings.BASE_DIR / 'models'
        output_dir = settings.MEDIA_ROOT
        
        worker = BackgroundCameraWorker(camera, models_dir, output_dir)
        worker.start()
        background_workers[camera_id] = worker
    
    return JsonResponse({'success': True, 'message': f'Worker started for {camera.camera_id}'})


@login_required
@require_http_methods(['POST'])
def stop_background_worker(request, camera_id):
    """Stop background worker for a camera (threading-based)"""
    if not request.user.is_admin():
        return JsonResponse({'error': 'Admin only'}, status=403)
    
    with background_worker_lock:
        if camera_id not in background_workers:
            return JsonResponse({'error': 'Worker not found', 'status': 'stopped'})
        
        worker = background_workers[camera_id]
        worker.stop()
        del background_workers[camera_id]
    
    return JsonResponse({'success': True, 'message': 'Worker stopped'})


@login_required
def get_background_worker_status(request):
    """Get status of all background workers (threading-based)"""
    if not request.user.is_admin():
        return JsonResponse({'error': 'Admin only'}, status=403)
    
    statuses = {}
    with background_worker_lock:
        for camera_id, worker in list(background_workers.items()):
            camera = Camera.objects.filter(id=camera_id).first()
            
            statuses[camera_id] = {
                'camera_id': worker.camera_code if hasattr(worker, 'camera_code') else (camera.camera_id if camera else f'cam-{camera_id}'),
                'camera_name': camera.name if camera else 'Unknown',
                'status': worker.status,
                'running': worker.running,
                'frame_count': worker.frame_count,
                'last_error': worker.last_error,
                'uptime': worker.get_uptime(),
                'events_detected': worker.events_detected,
                'frames_processed': worker.frames_processed,
                'start_time': worker.start_time.isoformat() if worker.start_time else None,
            }
    
    # Also include cameras without workers
    all_cameras = Camera.objects.all()
    for camera in all_cameras:
        if camera.id not in statuses:
            statuses[camera.id] = {
                'camera_id': camera.camera_id,
                'camera_name': camera.name,
                'status': 'stopped',
                'running': False,
                'frame_count': 0,
                'last_error': None,
                'uptime': 'Not running',
                'events_detected': 0,
                'frames_processed': 0,
                'start_time': None,
            }
    
    return JsonResponse({'workers': statuses})


def start_all_background_workers_internal():
    """Start background workers for all cameras (threading-based for Gunicorn)
    
    This is called automatically when Django starts.
    """
    cameras = Camera.objects.filter(status__in=['online', 'offline'])
    started = []
    failed = []
    
    models_dir = settings.BASE_DIR / 'models'
    output_dir = settings.MEDIA_ROOT
    
    with background_worker_lock:
        for camera in cameras:
            # Check if worker already exists
            if camera.id in background_workers:
                worker = background_workers[camera.id]
                if worker.running:
                    print(f"  ⏭️  Worker already running: {camera.camera_id}")
                    continue
                else:
                    # Dead worker, remove it
                    del background_workers[camera.id]
            
            try:
                worker = BackgroundCameraWorker(camera, models_dir, output_dir)
                worker.start()
                background_workers[camera.id] = worker
                started.append(camera.camera_id)
                print(f"  ▶️  Started worker thread: {camera.camera_id} ({camera.name})")
            except Exception as e:
                failed.append((camera.camera_id, str(e)))
                print(f"  ✗ Failed to start {camera.camera_id}: {e}")
    
    if failed:
        print(f"  ⚠️  {len(failed)} workers failed to start")
    
    return started


@login_required
@require_http_methods(['POST'])
def start_all_background_workers(request):
    """Start background workers for all cameras"""
    if not request.user.is_admin():
        return JsonResponse({'error': 'Admin only'}, status=403)
    
    started = start_all_background_workers_internal()
    return JsonResponse({'success': True, 'started': started, 'count': len(started)})


@login_required
@require_http_methods(['POST'])
def stop_all_background_workers(request):
    """Stop all background workers (threading-based)"""
    if not request.user.is_admin():
        return JsonResponse({'error': 'Admin only'}, status=403)
    
    stopped = []
    with background_worker_lock:
        for camera_id, worker in list(background_workers.items()):
            camera = Camera.objects.filter(id=camera_id).first()
            camera_code = camera.camera_id if camera else f'cam-{camera_id}'
            worker.stop()
            stopped.append(camera_code)
        background_workers.clear()
    
    return JsonResponse({'success': True, 'stopped': stopped, 'count': len(stopped)})
