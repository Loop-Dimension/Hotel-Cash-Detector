"""
Models for Hotel CCTV Monitoring System
"""
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone


class User(AbstractUser):
    """
    Custom User model with two account types:
    - ADMIN: Master account, can see everything
    - PROJECT_MANAGER: Can only see their assigned projects/hotels
    """
    ROLE_CHOICES = [
        ('admin', 'Admin (Master)'),
        ('project_manager', 'Project Manager'),
    ]
    
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='project_manager')
    phone = models.CharField(max_length=20, blank=True, null=True)
    
    class Meta:
        db_table = 'users'
    
    def is_admin(self):
        return self.role == 'admin'
    
    def is_project_manager(self):
        return self.role == 'project_manager'
    
    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"


class Region(models.Model):
    """Regions for organizing branches"""
    name = models.CharField(max_length=50, unique=True)
    code = models.CharField(max_length=10, unique=True)
    
    class Meta:
        db_table = 'regions'
        ordering = ['name']
    
    def __str__(self):
        return self.name


class Branch(models.Model):
    """Hotel/Project branches - synced from PMS projects"""
    STATUS_CHOICES = [
        ('confirmed', '확인완료'),
        ('reviewing', '확인중'),
        ('pending', '미확인'),
    ]
    
    name = models.CharField(max_length=100)
    # Region is now optional - kept for backwards compatibility but not used in UI
    region = models.ForeignKey(Region, on_delete=models.SET_NULL, related_name='branches', null=True, blank=True)
    address = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # PMS Integration - Link to PMS Project
    pms_project_id = models.CharField(max_length=36, blank=True, null=True, unique=True,
                                       help_text='UUID from PMS project for synchronization')
    pms_project_type = models.CharField(max_length=50, blank=True, null=True)
    
    # Project managers assigned to this branch
    managers = models.ManyToManyField(User, related_name='managed_branches', blank=True)
    
    class Meta:
        db_table = 'branches'
        ordering = ['name']
        verbose_name_plural = 'Branches'
    
    def __str__(self):
        return self.name
    
    def get_camera_count(self):
        return self.cameras.count()
    
    def get_online_camera_count(self):
        return self.cameras.filter(status='online').count()
    
    def get_today_event_count(self):
        today = timezone.now().date()
        return self.events.filter(created_at__date=today).count()


class Camera(models.Model):
    """CCTV Cameras"""
    STATUS_CHOICES = [
        ('online', '활성'),
        ('offline', '오프라인'),
        ('maintenance', '점검중'),
    ]
    
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE, related_name='cameras')
    camera_id = models.CharField(max_length=50)  # e.g., CAM-SEO-01
    name = models.CharField(max_length=100)  # e.g., 로비 카메라 1
    location = models.CharField(max_length=100, blank=True)  # e.g., 출입구, 카운터
    rtsp_url = models.CharField(max_length=500)  # RTSP stream URL
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='online')
    
    # Detection zone toggles
    cashier_zone_enabled = models.BooleanField(default=False)
    cash_drawer_zone_enabled = models.BooleanField(default=False)
    
    # Hand tracking duration (frames) - how long to track cashier's hand after touch
    hand_tracking_duration = models.IntegerField(default=90)  # ~3 seconds at 30fps
    
    # Polygon zones (always enabled for precise detection)
    cashier_zone_polygon = models.TextField(blank=True, null=True, help_text='JSON array of polygon points [[x1,y1],[x2,y2],...] for cashier zone')
    cash_drawer_zone_polygon = models.TextField(blank=True, null=True, help_text='JSON array of polygon points [[x1,y1],[x2,y2],...] for cash drawer zone')
    
    # Independent confidence thresholds per camera
    cash_confidence = models.FloatField(default=0.5)
    violence_confidence = models.FloatField(default=0.6)
    fire_confidence = models.FloatField(default=0.5)
    pose_confidence = models.FloatField(default=0.3)  # Person detection confidence for debug overlay
    
    # Hand touch distance threshold (pixels) for cash detection
    hand_touch_distance = models.IntegerField(default=100)
    
    # Detection toggles
    detect_cash = models.BooleanField(default=True)
    detect_violence = models.BooleanField(default=True)
    detect_fire = models.BooleanField(default=True)
    
    # Model paths (optional override per camera, uses global if empty)
    custom_yolo_model = models.CharField(max_length=500, blank=True, null=True)
    custom_pose_model = models.CharField(max_length=500, blank=True, null=True)
    custom_fire_model = models.CharField(max_length=500, blank=True, null=True)
    
    last_connected = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'cameras'
        ordering = ['branch', 'camera_id']
        unique_together = ['branch', 'camera_id']
    
    def __str__(self):
        return f"{self.camera_id} - {self.name}"
    
    def get_cashier_zone_polygon_points(self):
        """Get cashier zone as polygon points list"""
        import json
        if self.cashier_zone_polygon:
            try:
                return json.loads(self.cashier_zone_polygon)
            except json.JSONDecodeError:
                pass
        # Return None if no polygon exists
        return None
    
    def set_cashier_zone_polygon(self, points, enabled=True):
        """Set cashier zone from polygon points [[x1,y1], [x2,y2], ...]"""
        import json
        self.cashier_zone_polygon = json.dumps(points)
        self.cashier_zone_enabled = enabled
        self.save()
    
    def get_cash_drawer_zone_polygon_points(self):
        """Get cash drawer zone as polygon points list"""
        import json
        if self.cash_drawer_zone_polygon:
            try:
                return json.loads(self.cash_drawer_zone_polygon)
            except json.JSONDecodeError:
                pass
        # Return None if no polygon exists
        return None
    
    def set_cash_drawer_zone_polygon(self, points, enabled=True):
        """Set cash drawer zone from polygon points [[x1,y1], [x2,y2], ...]"""
        import json
        self.cash_drawer_zone_polygon = json.dumps(points)
        self.cash_drawer_zone_enabled = enabled
        self.save()
    
    def get_confidence_thresholds(self):
        """Get all confidence thresholds"""
        return {
            'cash': self.cash_confidence,
            'violence': self.violence_confidence,
            'fire': self.fire_confidence
        }
    
    def get_detection_settings(self):
        """Get full detection settings for this camera"""
        return {
            'detect_cash': self.detect_cash,
            'detect_violence': self.detect_violence,
            'detect_fire': self.detect_fire,
            'cash_confidence': self.cash_confidence,
            'violence_confidence': self.violence_confidence,
            'fire_confidence': self.fire_confidence,
            'hand_tracking_duration': self.hand_tracking_duration,
            'cashier_zone_polygon': self.get_cashier_zone_polygon_points(),
            'cash_drawer_zone_polygon': self.get_cash_drawer_zone_polygon_points(),
        }


class Event(models.Model):
    """Detection events"""
    TYPE_CHOICES = [
        ('cash', '현금'),
        ('potential_cash', '잠재 현금'),
        ('fire', '화재'),
        ('violence', '난동'),
    ]
    
    STATUS_CHOICES = [
        ('confirmed', '확인완료'),
        ('reviewing', '확인중'),
        ('pending', '미확인'),
    ]
    
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE, related_name='events')
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='events')
    event_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    confidence = models.FloatField(default=0.0)
    frame_number = models.IntegerField(default=0)
    
    # Bounding box
    bbox_x1 = models.IntegerField(default=0)
    bbox_y1 = models.IntegerField(default=0)
    bbox_x2 = models.IntegerField(default=0)
    bbox_y2 = models.IntegerField(default=0)
    
    # Clip path if exported
    clip_path = models.CharField(max_length=500, blank=True, null=True)
    thumbnail_path = models.CharField(max_length=500, blank=True, null=True)
    
    # JSON metadata with detection parameters
    metadata = models.TextField(blank=True, null=True, help_text='JSON with detection parameters')
    
    notes = models.TextField(blank=True, null=True)
    reviewed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='reviewed_events')
    reviewed_at = models.DateTimeField(blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'events'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.get_event_type_display()} - {self.camera.camera_id} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
    
    def get_bbox(self):
        return [self.bbox_x1, self.bbox_y1, self.bbox_x2, self.bbox_y2]


class GeminiPrompts(models.Model):
    """Global Gemini AI prompts for all cameras"""
    unified_prompt = models.TextField(
        blank=True, 
        null=True,
        help_text='Unified prompt for all detection types (cash, violence, fire)'
    )
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'gemini_prompts'
        verbose_name = 'Gemini AI Prompt'
        verbose_name_plural = 'Gemini AI Prompts'
    
    def __str__(self):
        return f"Gemini Prompts (Updated: {self.updated_at})"
    
    @classmethod
    def get_prompts(cls):
        """Get global prompts or create default"""
        from detectors.gemini_validator import DEFAULT_UNIFIED_PROMPT
        instance, created = cls.objects.get_or_create(pk=1)
        
        # Return unified prompt or default
        unified = instance.unified_prompt or DEFAULT_UNIFIED_PROMPT
        return {
            'unified': unified,
            'cash': unified,
            'violence': unified,
            'fire': unified,
        }
    
    @classmethod
    def get_default_prompt(cls):
        """Get the default unified prompt"""
        from detectors.gemini_validator import DEFAULT_UNIFIED_PROMPT
        return DEFAULT_UNIFIED_PROMPT
    
    @classmethod
    def set_unified_prompt(cls, prompt: str):
        """Set unified prompt for all detection types"""
        instance, created = cls.objects.get_or_create(pk=1)
        instance.unified_prompt = prompt if prompt else None
        instance.save()
        return instance


class VideoRecord(models.Model):
    """Full video recordings"""
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE, related_name='videos')
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='videos')
    file_id = models.CharField(max_length=50)
    file_path = models.CharField(max_length=500)
    file_size = models.BigIntegerField(default=0)  # in bytes
    duration = models.IntegerField(default=0)  # in seconds
    
    recorded_date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'video_records'
        ordering = ['-recorded_date']
    
    def __str__(self):
        return f"{self.file_id} - {self.branch.name} ({self.recorded_date})"


class BranchAccount(models.Model):
    """Accounts assigned to branches (for branch detail management)"""
    ROLE_CHOICES = [
        ('manager', '지점장'),
        ('staff', '스태프'),
        ('control', '관제'),
    ]
    
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE, related_name='accounts')
    name = models.CharField(max_length=100)
    email = models.EmailField()
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='staff')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'branch_accounts'
        ordering = ['branch', 'name']
    
    def __str__(self):
        return f"{self.name} - {self.branch.name} ({self.get_role_display()})"


class GeminiLog(models.Model):
    """Logs for Gemini AI validation events"""
    VALIDATION_TYPE_CHOICES = [
        ('image', 'Image (Screenshot)'),
        ('video', 'Video (3 sec clip)'),
    ]
    
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name='gemini_logs')
    event_type = models.CharField(max_length=20)  # cash, violence, fire
    validation_type = models.CharField(max_length=10, choices=VALIDATION_TYPE_CHOICES, default='image', help_text='Whether validation was done via image or video')
    is_validated = models.BooleanField(default=False)
    confidence = models.FloatField(default=0.0)
    reason = models.TextField(blank=True)
    prompt_used = models.TextField(blank=True)
    response_raw = models.TextField(blank=True, help_text='Raw JSON response from Gemini')
    image_path = models.CharField(max_length=500, blank=True, null=True)
    video_path = models.CharField(max_length=500, blank=True, null=True, help_text='Path to validation video clip if used')
    processing_time_ms = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'gemini_logs'
        ordering = ['-created_at']
    
    def __str__(self):
        status = "✓" if self.is_validated else "✗"
        return f"[{status}] {self.event_type} - {self.camera.camera_id} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
