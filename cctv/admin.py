"""
Django Admin configuration for CCTV app
"""
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.utils import timezone
from django.utils.html import format_html
from .models import User, Region, Branch, Camera, Event, VideoRecord, BranchAccount, GeminiPrompts, GeminiLog


# Add server time to admin header (Korean timezone)
def get_admin_header():
    """Generate admin header with current Korean time"""
    now = timezone.localtime()
    return format_html(
        'Hotel CCTV Admin <span style="font-size: 12px; margin-left: 20px; color: #666;">서버 시간: {} KST</span>',
        now.strftime('%Y-%m-%d %H:%M:%S')
    )

admin.site.site_header = get_admin_header()


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
    list_display = ['camera_id', 'name', 'branch', 'status', 'location', 'detect_cash', 'detect_violence', 'detect_fire', 'cashier_zone_enabled']
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
            'fields': ('cash_confidence', 'violence_confidence', 'fire_confidence', 'pose_confidence'),
            'description': 'Set confidence threshold (0.0 - 1.0) for each detection type'
        }),
        ('Hand Detection Settings', {
            'fields': ('hand_touch_distance', 'hand_tracking_duration'),
            'description': 'Distance threshold for hand proximity detection and tracking duration'
        }),
        ('Polygon Zones', {
            'fields': ('cashier_zone_enabled', 'cashier_zone_polygon', 'cash_drawer_zone_polygon'),
            'description': 'Define detection zones using polygon points [[x1,y1], [x2,y2], ...]'
        }),
        ('Custom Models (Optional)', {
            'fields': ('custom_yolo_model', 'custom_pose_model', 'custom_fire_model'),
            'classes': ('collapse',),
            'description': 'Override global model settings for this camera'
        }),
    )


@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ['id', 'branch', 'camera', 'event_type', 'confidence_percent', 'status', 'created_at', 'reviewed_by']
    list_filter = ['event_type', 'status', 'branch', 'created_at']
    search_fields = ['camera__camera_id', 'branch__name']
    date_hierarchy = 'created_at'
    list_editable = ['status']
    readonly_fields = ['confidence', 'frame_number', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'created_at', 'clip_url', 'thumbnail_url']
    
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
            'fields': ('clip_path', 'clip_url', 'thumbnail_path', 'thumbnail_url'),
        }),
    )
    
    def confidence_percent(self, obj):
        return f"{obj.confidence * 100:.1f}%"
    confidence_percent.short_description = 'Confidence'
    confidence_percent.admin_order_field = 'confidence'
    
    def clip_url(self, obj):
        if obj.clip_path:
            if obj.clip_path.startswith('http'):
                return format_html('<a href="{}" target="_blank">{}</a>', obj.clip_path, obj.clip_path)
            return obj.clip_path
        return '-'
    clip_url.short_description = 'Clip URL (clickable)'
    
    def thumbnail_url(self, obj):
        if obj.thumbnail_path:
            if obj.thumbnail_path.startswith('http'):
                return format_html('<a href="{}" target="_blank"><img src="{}" width="100" /> View</a>', obj.thumbnail_path, obj.thumbnail_path)
            return obj.thumbnail_path
        return '-'
    thumbnail_url.short_description = 'Thumbnail Preview'


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


@admin.register(GeminiPrompts)
class GeminiPromptsAdmin(admin.ModelAdmin):
    list_display = ['id', 'updated_at', 'created_at']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Global Gemini AI Prompts', {
            'fields': ('unified_prompt',),
            'description': 'Configure the unified prompt used for all detection types (cash, violence, fire) across all cameras'
        }),
        ('Metadata', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def has_add_permission(self, request):
        # Only allow one instance
        return not GeminiPrompts.objects.exists()
    
    def has_delete_permission(self, request, obj=None):
        # Don't allow deletion
        return False


@admin.register(GeminiLog)
class GeminiLogAdmin(admin.ModelAdmin):
    list_display = ['id', 'camera', 'event_type', 'validation_type', 'is_validated', 'confidence_percent', 'processing_time', 'created_at']
    list_filter = ['event_type', 'validation_type', 'is_validated', 'camera', 'created_at']
    search_fields = ['camera__camera_id', 'camera__name', 'reason']
    date_hierarchy = 'created_at'
    readonly_fields = ['camera', 'event_type', 'validation_type', 'is_validated', 'confidence', 'reason', 
                       'prompt_used', 'response_raw', 'image_url', 'video_url', 'processing_time_ms', 'created_at']
    
    fieldsets = (
        ('Validation Info', {
            'fields': ('camera', 'event_type', 'validation_type', 'is_validated', 'confidence', 'processing_time_ms', 'created_at')
        }),
        ('Result', {
            'fields': ('reason',),
        }),
        ('Media', {
            'fields': ('image_path', 'image_url', 'video_path', 'video_url'),
        }),
        ('Prompt & Response', {
            'fields': ('prompt_used', 'response_raw'),
            'classes': ('collapse',)
        }),
    )
    
    def confidence_percent(self, obj):
        return f"{obj.confidence * 100:.1f}%"
    confidence_percent.short_description = 'Confidence'
    confidence_percent.admin_order_field = 'confidence'
    
    def processing_time(self, obj):
        return f"{obj.processing_time_ms}ms"
    processing_time.short_description = 'Time'
    processing_time.admin_order_field = 'processing_time_ms'
    
    def image_url(self, obj):
        if obj.image_path:
            if obj.image_path.startswith('http'):
                return format_html('<a href="{}" target="_blank"><img src="{}" width="200" /></a>', obj.image_path, obj.image_path)
            # Check if it's a relative path - construct S3 URL
            from django.conf import settings
            if getattr(settings, 'USE_S3', False):
                url = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{obj.image_path}"
                return format_html('<a href="{}" target="_blank"><img src="{}" width="200" /></a>', url, url)
            return obj.image_path
        return '-'
    image_url.short_description = 'Image Preview'
    
    def video_url(self, obj):
        if obj.video_path:
            if obj.video_path.startswith('http'):
                return format_html('<a href="{}" target="_blank">View Video</a>', obj.video_path)
            return obj.video_path
        return '-'
    video_url.short_description = 'Video URL'
    
    def has_add_permission(self, request):
        return False
    
    def has_change_permission(self, request, obj=None):
        return False

