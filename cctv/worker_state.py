"""
Worker state management using SQLite for multi-process synchronization.

This allows multiple Gunicorn workers to share background worker state
without duplicate threads or inconsistent status.
"""
import os
import time
from datetime import datetime
from django.conf import settings
from django.db import models


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
        elapsed = (datetime.now().replace(tzinfo=self.start_time.tzinfo) - self.start_time).total_seconds()
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def is_alive(self, timeout=30):
        """Check if worker is alive based on heartbeat"""
        if not self.last_heartbeat:
            return False
        elapsed = (datetime.now().replace(tzinfo=self.last_heartbeat.tzinfo) - self.last_heartbeat).total_seconds()
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
        from django.utils import timezone
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
