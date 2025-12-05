"""
Django Admin configuration for CCTV app
"""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, Region, Branch, Camera, Event, VideoRecord, BranchAccount


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['username', 'email', 'role', 'is_staff', 'is_active']
    list_filter = ['role', 'is_staff', 'is_active']
    fieldsets = BaseUserAdmin.fieldsets + (
        ('Role', {'fields': ('role', 'phone')}),
    )
    add_fieldsets = BaseUserAdmin.add_fieldsets + (
        ('Role', {'fields': ('role', 'phone')}),
    )


@admin.register(Region)
class RegionAdmin(admin.ModelAdmin):
    list_display = ['name', 'code']
    search_fields = ['name', 'code']


@admin.register(Branch)
class BranchAdmin(admin.ModelAdmin):
    list_display = ['name', 'region', 'status', 'created_at']
    list_filter = ['region', 'status']
    search_fields = ['name']
    filter_horizontal = ['managers']


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ['camera_id', 'name', 'branch', 'status', 'location', 'detect_cash', 'detect_violence', 'detect_fire']
    list_filter = ['branch', 'status', 'detect_cash', 'detect_violence', 'detect_fire']
    search_fields = ['camera_id', 'name']
    list_editable = ['status', 'detect_cash', 'detect_violence', 'detect_fire']
    fieldsets = (
        ('Basic Info', {
            'fields': ('camera_id', 'name', 'branch', 'location', 'rtsp_url', 'status')
        }),
        ('Detection Settings', {
            'fields': ('detect_cash', 'detect_violence', 'detect_fire')
        }),
        ('Confidence Thresholds', {
            'fields': ('cash_confidence', 'violence_confidence', 'fire_confidence'),
            'description': 'Set confidence threshold (0.0 - 1.0) for each detection type'
        }),
        ('Cashier Zone', {
            'fields': ('cashier_zone_enabled', 'cashier_zone_x', 'cashier_zone_y', 'cashier_zone_width', 'cashier_zone_height'),
            'classes': ('collapse',)
        }),
        ('Custom Models', {
            'fields': ('custom_yolo_model', 'custom_pose_model', 'custom_fire_model'),
            'classes': ('collapse',)
        }),
    )


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ['id', 'branch', 'camera', 'event_type', 'confidence_percent', 'status', 'created_at', 'reviewed_by']
    list_filter = ['event_type', 'status', 'branch', 'created_at']
    search_fields = ['camera__camera_id', 'branch__name']
    date_hierarchy = 'created_at'
    list_editable = ['status']
    readonly_fields = ['confidence', 'frame_number', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'created_at']
    
    fieldsets = (
        ('Event Info', {
            'fields': ('branch', 'camera', 'event_type', 'status')
        }),
        ('Detection Details', {
            'fields': ('confidence', 'frame_number'),
        }),
        ('Bounding Box', {
            'fields': ('bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'),
            'classes': ('collapse',)
        }),
        ('Review', {
            'fields': ('reviewed_by', 'reviewed_at', 'notes')
        }),
        ('Files', {
            'fields': ('clip_path', 'thumbnail_path'),
            'classes': ('collapse',)
        }),
    )
    
    def confidence_percent(self, obj):
        return f"{obj.confidence * 100:.1f}%"
    confidence_percent.short_description = 'Confidence'
    confidence_percent.admin_order_field = 'confidence'


@admin.register(VideoRecord)
class VideoRecordAdmin(admin.ModelAdmin):
    list_display = ['file_id', 'branch', 'camera', 'recorded_date']
    list_filter = ['branch', 'recorded_date']
    search_fields = ['file_id', 'branch__name']


@admin.register(BranchAccount)
class BranchAccountAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'branch', 'role']
    list_filter = ['branch', 'role']
    search_fields = ['name', 'email']
