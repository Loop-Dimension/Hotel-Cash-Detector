"""
URL patterns for CCTV app
"""
from django.urls import path
from . import views

app_name = 'cctv'

urlpatterns = [
    # Authentication
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Main pages
    path('', views.home, name='home'),
    path('monitor/all/', views.monitor_all, name='monitor_all'),
    path('monitor/local/', views.monitor_local, name='monitor_local'),
    path('monitor/local/<int:branch_id>/', views.monitor_local, name='monitor_local_branch'),
    
    # Video management
    path('video/logs/', views.video_logs, name='video_logs'),
    path('video/full/', views.video_full, name='video_full'),
    
    # Branch management
    path('manage/branches/', views.manage_branches, name='manage_branches'),
    path('manage/branches/<int:branch_id>/', views.manage_branch_detail, name='manage_branch_detail'),
    path('camera/<int:camera_id>/settings/', views.camera_settings, name='camera_settings'),
    
    # Reports
    path('reports/', views.reports, name='reports'),
    
    # Language
    path('api/set-language/', views.set_language, name='set_language'),
    path('api/translations/', views.get_translations_api, name='get_translations'),
    
    # API endpoints - Branches
    path('api/branches/', views.api_branches, name='api_branches'),
    path('api/branches/<int:branch_id>/', views.api_branch_detail, name='api_branch_detail'),
    path('api/branches/<int:branch_id>/cameras/', views.api_branch_cameras, name='api_branch_cameras'),
    path('api/branches/<int:branch_id>/accounts/', views.api_branch_accounts, name='api_branch_accounts'),
    
    # API endpoints - Cameras
    path('api/cameras/', views.api_cameras, name='api_cameras'),
    path('api/cameras/<int:camera_id>/', views.api_camera_detail, name='api_camera_detail'),
    path('api/cameras/<int:camera_id>/set-zone/', views.api_set_cashier_zone, name='api_set_cashier_zone'),
    path('api/cameras/<int:camera_id>/toggle-detection/', views.api_toggle_detection, name='api_toggle_detection'),
    path('api/cameras/<int:camera_id>/settings/', views.api_camera_settings, name='api_camera_settings'),
    path('api/cameras/<int:camera_id>/test-connection/', views.api_test_camera_connection, name='api_test_camera_connection'),
    
    # API endpoints - Events
    path('api/events/', views.api_events, name='api_events'),
    path('api/events/<int:event_id>/', views.api_event_detail, name='api_event_detail'),
    path('api/events/bulk-delete/', views.api_bulk_delete_events, name='api_bulk_delete_events'),
    path('api/events/bulk-update/', views.api_bulk_update_events, name='api_bulk_update_events'),
    
    # API endpoints - Videos
    path('api/videos/', views.api_videos, name='api_videos'),
    
    # API endpoints - Users
    path('api/users/', views.api_users, name='api_users'),
    path('api/users/<int:user_id>/', views.api_user_detail, name='api_user_detail'),
    
    # API endpoints - Regions
    path('api/regions/', views.api_regions, name='api_regions'),
    path('api/regions/<int:region_id>/', views.api_region_detail, name='api_region_detail'),
    
    # API endpoints - Stats & Reports
    path('api/home-stats/', views.api_home_stats, name='api_home_stats'),
    path('api/report-stats/', views.api_report_stats, name='api_report_stats'),
    path('api/reports/', views.api_reports, name='api_reports'),
    
    # Video streaming
    path('video-feed/<int:camera_id>/', views.video_feed, name='video_feed'),
    path('video-feed-debug/<int:camera_id>/', views.video_feed_debug, name='video_feed_debug'),
    
    # Developer Mode API
    path('api/cameras/<int:camera_id>/dev-mode/verify/', views.api_verify_dev_password, name='api_verify_dev_password'),
    path('api/cameras/<int:camera_id>/dev-mode/lock/', views.api_lock_dev_mode, name='api_lock_dev_mode'),
    path('api/cameras/<int:camera_id>/dev-mode/status/', views.api_dev_mode_status, name='api_dev_mode_status'),
    path('api/cameras/<int:camera_id>/dev-mode/debug-info/', views.api_detection_debug_info, name='api_detection_debug_info'),
    
    # Background workers API
    path('api/workers/status/', views.get_background_worker_status, name='api_workers_status'),
    path('api/workers/start-all/', views.start_all_background_workers, name='api_workers_start_all'),
    path('api/workers/stop-all/', views.stop_all_background_workers, name='api_workers_stop_all'),
    path('api/workers/<int:camera_id>/start/', views.start_background_worker, name='api_worker_start'),
    path('api/workers/<int:camera_id>/stop/', views.stop_background_worker, name='api_worker_stop'),
]
