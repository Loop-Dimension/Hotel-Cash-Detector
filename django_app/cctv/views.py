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

# Add flask directory to path for detector imports
FLASK_DIR = Path(settings.BASE_DIR).parent / 'flask'
sys.path.insert(0, str(FLASK_DIR))

# Try to import detectors
try:
    from detectors import UnifiedDetector
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    print("Warning: Detectors not available")

# Global detector instances per camera
camera_detectors = {}


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
    date_filter = request.GET.get('date', timezone.now().date().isoformat())
    region_filter = request.GET.get('region', 'all')
    type_filter = request.GET.get('type', 'all')
    branch_filter = request.GET.get('branch', '')
    
    # Build query
    events = Event.objects.select_related('branch', 'camera', 'branch__region')
    
    if not user.is_admin():
        events = events.filter(branch__in=user_branches)
    
    if date_filter:
        events = events.filter(created_at__date=date_filter)
    
    if region_filter != 'all':
        events = events.filter(branch__region__name=region_filter)
    
    if type_filter != 'all':
        events = events.filter(event_type=type_filter)
    
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
            'date': date_filter,
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
        # Try to connect to the RTSP stream
        cap = cv2.VideoCapture(camera.rtsp_url)
        
        # Set timeout (5 seconds)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        
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
            # Camera-specific confidence thresholds
            'cash_confidence': camera.cash_confidence,
            'violence_confidence': camera.violence_confidence,
            'fire_confidence': camera.fire_confidence,
            # Detection toggles
            'detect_cash': camera.detect_cash,
            'detect_violence': camera.detect_violence,
            'detect_fire': camera.detect_fire,
            # Other settings from global config
            'hand_touch_distance': settings.DETECTION_CONFIG.get('HAND_TOUCH_DISTANCE', 100),
            'pose_confidence': settings.DETECTION_CONFIG.get('POSE_CONFIDENCE', 0.5),
            'min_transaction_frames': settings.DETECTION_CONFIG.get('MIN_TRANSACTION_FRAMES', 30),
            'min_fire_frames': settings.DETECTION_CONFIG.get('MIN_FIRE_FRAMES', 15),
            'min_fire_area': settings.DETECTION_CONFIG.get('MIN_FIRE_AREA', 500),
            'violence_duration': settings.DETECTION_CONFIG.get('VIOLENCE_DURATION', 30),
        }
        camera_detectors[camera.id] = UnifiedDetector(config)
        camera_detectors[f'{camera.id}_key'] = cache_key
    
    return camera_detectors[camera.id]


def generate_frames(camera):
    """Generator for video frames with detection"""
    cap = cv2.VideoCapture(camera.rtsp_url)
    
    # Set timeout for connection
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    
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
