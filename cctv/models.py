"""
Models for Hotel CCTV Monitoring System
"""
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
from datetime import datetime


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
    """Hotel/Project branches"""
    STATUS_CHOICES = [
        ('confirmed', '확인완료'),
        ('reviewing', '확인중'),
        ('pending', '미확인'),
    ]
    
    name = models.CharField(max_length=100)
    region = models.ForeignKey(Region, on_delete=models.CASCADE, related_name='branches')
    address = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Project managers assigned to this branch
    managers = models.ManyToManyField(User, related_name='managed_branches', blank=True)
    
    class Meta:
        db_table = 'branches'
        ordering = ['name']
        verbose_name_plural = 'Branches'
    
    def __str__(self):
        return f"{self.name} ({self.region.name})"
    
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
    
    # Cashier zone for cash detection [x, y, width, height] - independent per camera
    cashier_zone_x = models.IntegerField(default=0)
    cashier_zone_y = models.IntegerField(default=0)
    cashier_zone_width = models.IntegerField(default=640)
    cashier_zone_height = models.IntegerField(default=480)
    cashier_zone_enabled = models.BooleanField(default=False)
    
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
    
    def get_cashier_zone(self):
        """Get cashier zone as dict"""
        return {
            'x': self.cashier_zone_x,
            'y': self.cashier_zone_y,
            'width': self.cashier_zone_width,
            'height': self.cashier_zone_height,
            'enabled': self.cashier_zone_enabled
        }
    
    def set_cashier_zone(self, x, y, width, height, enabled=True):
        """Set cashier zone from coordinates"""
        self.cashier_zone_x = int(x)
        self.cashier_zone_y = int(y)
        self.cashier_zone_width = int(width)
        self.cashier_zone_height = int(height)
        self.cashier_zone_enabled = enabled
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
            'cashier_zone': self.get_cashier_zone()
        }


class Event(models.Model):
    """Detection events"""
    TYPE_CHOICES = [
        ('cash', '현금'),
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


class WorkerState(models.Model):
    """Track background worker state across multiple Gunicorn processes"""
    
    camera_id = models.IntegerField(unique=True, db_index=True)
    camera_code = models.CharField(max_length=50)
    
    # Worker state
    running = models.BooleanField(default=False)
    status = models.CharField(max_length=20, default='stopped')  # starting, running, reconnecting, stopped, error
    
    # Process info
    process_id = models.IntegerField(null=True)  # PID of worker process
    thread_id = models.CharField(max_length=50, null=True)  # Thread identifier
    
    # Statistics
    frame_count = models.IntegerField(default=0)
    events_detected = models.IntegerField(default=0)
    frames_processed = models.IntegerField(default=0)
    
    # Timestamps
    start_time = models.DateTimeField(null=True)
    last_heartbeat = models.DateTimeField(auto_now=True)  # Auto-update on save
    last_error = models.TextField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'worker_states'
        ordering = ['camera_id']
    
    def __str__(self):
        return f"Worker[{self.camera_code}] - {self.status}"
    
    def get_uptime(self):
        """Get worker uptime as formatted string"""
        if not self.start_time:
            return "Not started"
        now = datetime.now()
        if self.start_time.tzinfo:
            now = timezone.now()
        elapsed = (now - self.start_time).total_seconds()
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def is_alive(self, timeout=30):
        """Check if worker is alive based on heartbeat"""
        if not self.last_heartbeat:
            return False
        now = datetime.now()
        if self.last_heartbeat.tzinfo:
            now = timezone.now()
        elapsed = (now - self.last_heartbeat).total_seconds()
        return elapsed < timeout
    
    def heartbeat(self, frame_count=None, events_detected=None, frames_processed=None, status=None):
        """Update heartbeat and optionally other fields"""
        if frame_count is not None:
            self.frame_count = frame_count
        if events_detected is not None:
            self.events_detected = events_detected
        if frames_processed is not None:
            self.frames_processed = frames_processed
        if status is not None:
            self.status = status
        self.save(update_fields=['frame_count', 'events_detected', 'frames_processed', 'status', 'last_heartbeat', 'updated_at'])
    
    @classmethod
    def cleanup_dead_workers(cls, timeout=60):
        """Remove state for workers that haven't sent heartbeat in timeout seconds"""
        cutoff = timezone.now() - timezone.timedelta(seconds=timeout)
        dead_workers = cls.objects.filter(running=True, last_heartbeat__lt=cutoff)
        count = dead_workers.count()
        if count > 0:
            dead_workers.update(running=False, status='error', last_error='Heartbeat timeout')
        return count
    
    @classmethod
    def get_or_create_for_camera(cls, camera):
        """Get or create worker state for a camera"""
        state, created = cls.objects.get_or_create(
            camera_id=camera.id,
            defaults={
                'camera_code': camera.camera_id,
                'running': False,
                'status': 'stopped',
            }
        )
        if not created and state.camera_code != camera.camera_id:
            # Update camera_code if changed
            state.camera_code = camera.camera_id
            state.save(update_fields=['camera_code'])
        return state
    
    @classmethod
    def is_worker_running(cls, camera_id):
        """Check if worker is running for camera"""
        try:
            state = cls.objects.get(camera_id=camera_id)
            return state.running and state.is_alive()
        except cls.DoesNotExist:
            return False
    
    @classmethod
    def get_all_states(cls):
        """Get all worker states with alive check"""
        states = {}
        for state in cls.objects.all():
            states[state.camera_id] = {
                'camera_id': state.camera_id,
                'camera_code': state.camera_code,
                'status': state.status,
                'running': state.running and state.is_alive(),
                'frame_count': state.frame_count,
                'events_detected': state.events_detected,
                'frames_processed': state.frames_processed,
                'last_error': state.last_error,
                'uptime': state.get_uptime(),
                'start_time': state.start_time.isoformat() if state.start_time else None,
                'is_alive': state.is_alive(),
            }
        return states
