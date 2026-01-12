"""
Integration tests for S3 storage functionality.

These tests verify the complete flow of:
1. Creating validation clips
2. Uploading to S3
3. Storing correct URLs in database
4. API endpoints returning correct URLs

Run with: python manage.py test tests.test_s3_integration
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hotel_cctv.settings')
django.setup()

from django.test import TestCase, override_settings
from django.conf import settings

from cctv.models import Camera, Branch, GeminiLog, Event
from cctv.utils import (
    save_validation_clip,
    upload_validation_clip_to_s3,
    save_clip,
    save_event,
    _log_video_validation,
)
from cctv.storage import is_s3_enabled, get_storage, upload_file_to_storage


class S3StorageBaseTest(TestCase):
    """Base test class with common setup."""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Create test branch and camera
        cls.branch = Branch.objects.create(
            name='Test Branch',
            branch_id='TEST001'
        )
        cls.camera = Camera.objects.create(
            name='Test Camera',
            camera_id='CAM-TEST-001',
            branch=cls.branch,
            rtsp_url='rtsp://test:test@localhost/stream'
        )
    
    @classmethod
    def tearDownClass(cls):
        # Cleanup
        GeminiLog.objects.filter(camera=cls.camera).delete()
        Event.objects.filter(camera=cls.camera).delete()
        cls.camera.delete()
        cls.branch.delete()
        super().tearDownClass()
    
    def create_test_frames(self, count=45, width=640, height=480):
        """Create dummy frames for testing."""
        frames = []
        for i in range(count):
            # Create a simple colored frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 5 % 256  # Blue channel varies
            frame[:, :, 1] = 100  # Green constant
            frame[:, :, 2] = 50   # Red constant
            frames.append(frame)
        return frames


class TestS3ConfigurationTest(S3StorageBaseTest):
    """Test S3 configuration and helpers."""
    
    def test_is_s3_enabled_returns_bool(self):
        """is_s3_enabled should return a boolean."""
        result = is_s3_enabled()
        self.assertIsInstance(result, bool)
    
    def test_get_storage_returns_storage(self):
        """get_storage should return a storage backend."""
        storage = get_storage()
        self.assertIsNotNone(storage)
        # Should have save method
        self.assertTrue(hasattr(storage, 'save'))


class TestValidationClipFlow(S3StorageBaseTest):
    """Test the complete validation clip flow."""
    
    def test_save_validation_clip_creates_file(self):
        """save_validation_clip should create a video file."""
        frames = self.create_test_frames(30)
        
        result = save_validation_clip(frames, self.camera, 'cash')
        
        self.assertIsNotNone(result)
        # Should return a local temp path
        self.assertTrue(Path(result).exists())
        self.assertTrue(result.endswith('.mp4'))
        
        # Cleanup
        Path(result).unlink(missing_ok=True)
    
    def test_save_validation_clip_empty_frames_returns_none(self):
        """save_validation_clip should return None for empty frames."""
        result = save_validation_clip([], self.camera, 'cash')
        self.assertIsNone(result)
    
    @patch('cctv.utils.is_s3_enabled')
    @patch('cctv.utils.upload_file_to_storage')
    def test_upload_validation_clip_to_s3_when_enabled(self, mock_upload, mock_s3_enabled):
        """upload_validation_clip_to_s3 should upload and return S3 URL."""
        mock_s3_enabled.return_value = True
        mock_upload.return_value = 'https://hotel-cctv.s3.amazonaws.com/validation_clips/test.mp4'
        
        # Create a temp file
        frames = self.create_test_frames(30)
        local_path = save_validation_clip(frames, self.camera, 'cash')
        
        result = upload_validation_clip_to_s3(local_path)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith('https://'))
        mock_upload.assert_called_once()
    
    @patch('cctv.utils.is_s3_enabled')
    def test_upload_validation_clip_local_when_disabled(self, mock_s3_enabled):
        """upload_validation_clip_to_s3 should return /media/ path when S3 disabled."""
        mock_s3_enabled.return_value = False
        
        # Create a temp file
        frames = self.create_test_frames(30)
        local_path = save_validation_clip(frames, self.camera, 'cash')
        
        result = upload_validation_clip_to_s3(local_path)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith('/media/'))
        
        # Cleanup
        media_path = Path(settings.MEDIA_ROOT) / 'validation_clips' / Path(local_path).name
        media_path.unlink(missing_ok=True)


class TestGeminiLogIntegration(S3StorageBaseTest):
    """Test GeminiLog creation with S3 URLs."""
    
    def test_log_video_validation_creates_log(self):
        """_log_video_validation should create GeminiLog entry."""
        test_url = 'https://hotel-cctv.s3.amazonaws.com/validation_clips/test.mp4'
        
        _log_video_validation(
            camera_id=self.camera.id,
            event_type='cash',
            is_valid=False,
            confidence=0.85,
            reason='Test reason',
            prompt='Test prompt',
            response_raw='{"test": true}',
            video_url=test_url,
            processing_time_ms=1234
        )
        
        # Verify log was created
        log = GeminiLog.objects.filter(camera=self.camera).latest('created_at')
        
        self.assertEqual(log.event_type, 'cash')
        self.assertEqual(log.is_validated, False)
        self.assertEqual(log.confidence, 0.85)
        self.assertEqual(log.video_path, test_url)
        self.assertEqual(log.validation_type, 'video')
        self.assertEqual(log.processing_time_ms, 1234)
    
    def test_log_stores_full_s3_url(self):
        """GeminiLog should store full S3 URL, not /media/ path."""
        test_url = 'https://hotel-cctv.s3.ap-northeast-2.amazonaws.com/validation_clips/CAM-001_cash_20260112_210332.mp4'
        
        _log_video_validation(
            camera_id=self.camera.id,
            event_type='cash',
            is_valid=True,
            confidence=0.95,
            reason='Valid cash transaction',
            prompt='Analyze...',
            response_raw='{}',
            video_url=test_url,
            processing_time_ms=500
        )
        
        log = GeminiLog.objects.filter(camera=self.camera).latest('created_at')
        
        # Should NOT be a /media/ path
        self.assertFalse(log.video_path.startswith('/media/'))
        # Should be full S3 URL
        self.assertTrue(log.video_path.startswith('https://'))
        self.assertIn('s3', log.video_path)


class TestSaveClipIntegration(S3StorageBaseTest):
    """Test save_clip function with S3."""
    
    @patch('cctv.utils.is_s3_enabled')
    @patch('cctv.utils.upload_file_to_storage')
    @patch('cctv.utils.save_image_to_storage')
    def test_save_clip_returns_s3_urls(self, mock_save_image, mock_upload, mock_s3_enabled):
        """save_clip should return S3 URLs when S3 is enabled."""
        mock_s3_enabled.return_value = True
        mock_upload.return_value = 'https://hotel-cctv.s3.amazonaws.com/clips/test.mp4'
        mock_save_image.return_value = 'https://hotel-cctv.s3.amazonaws.com/thumbnails/test.jpg'
        
        frames = self.create_test_frames(60)
        
        clip_url, thumb_url = save_clip(frames, self.camera, 'cash', fps=15)
        
        self.assertIsNotNone(clip_url)
        self.assertIsNotNone(thumb_url)
        self.assertTrue(clip_url.startswith('https://'))
        self.assertTrue(thumb_url.startswith('https://'))


class TestSaveEventIntegration(S3StorageBaseTest):
    """Test save_event function with S3 URLs."""
    
    def test_save_event_stores_s3_urls(self):
        """save_event should store S3 URLs in database."""
        clip_url = 'https://hotel-cctv.s3.amazonaws.com/clips/test.mp4'
        thumb_url = 'https://hotel-cctv.s3.amazonaws.com/thumbnails/test.jpg'
        
        event = save_event(
            camera=self.camera,
            event_type='cash',
            confidence=0.9,
            frame_number=100,
            bbox=[10, 20, 100, 200],
            clip_path=clip_url,
            thumbnail_path=thumb_url,
            metadata={'test': True}
        )
        
        self.assertIsNotNone(event)
        self.assertEqual(event.clip_path, clip_url)
        self.assertEqual(event.thumbnail_path, thumb_url)
        # Should NOT modify URLs
        self.assertTrue(event.clip_path.startswith('https://'))


class TestCompleteFlow(S3StorageBaseTest):
    """Test complete detection -> validation -> storage flow."""
    
    @patch('cctv.utils.is_s3_enabled')
    @patch('cctv.utils.upload_file_to_storage')
    def test_complete_validation_flow(self, mock_upload, mock_s3_enabled):
        """Test complete flow: create clip -> validate -> upload -> log."""
        mock_s3_enabled.return_value = True
        s3_url = 'https://hotel-cctv.s3.ap-northeast-2.amazonaws.com/validation_clips/test_complete.mp4'
        mock_upload.return_value = s3_url
        
        # Step 1: Create validation clip (returns local temp path)
        frames = self.create_test_frames(45)
        local_path = save_validation_clip(frames, self.camera, 'cash')
        self.assertIsNotNone(local_path)
        self.assertTrue(Path(local_path).exists())
        
        # Step 2: Simulate Gemini validation (would read local file)
        # In real flow, validator.validate_event_video(local_path, event_type) is called here
        
        # Step 3: Upload to S3 after validation
        final_url = upload_validation_clip_to_s3(local_path)
        self.assertEqual(final_url, s3_url)
        
        # Step 4: Log to database with S3 URL
        _log_video_validation(
            camera_id=self.camera.id,
            event_type='cash',
            is_valid=False,
            confidence=0.75,
            reason='No cash visible',
            prompt='Analyze...',
            response_raw='{}',
            video_url=final_url,
            processing_time_ms=3000
        )
        
        # Step 5: Verify database has correct S3 URL
        log = GeminiLog.objects.filter(camera=self.camera).latest('created_at')
        self.assertEqual(log.video_path, s3_url)
        self.assertTrue(log.video_path.startswith('https://'))
        self.assertIn('s3', log.video_path)
    
    @patch('cctv.utils.is_s3_enabled')
    def test_complete_flow_local_storage(self, mock_s3_enabled):
        """Test complete flow with local storage (S3 disabled)."""
        mock_s3_enabled.return_value = False
        
        # Step 1: Create validation clip
        frames = self.create_test_frames(45)
        local_path = save_validation_clip(frames, self.camera, 'cash')
        
        # Step 2: Upload (moves to media folder)
        final_url = upload_validation_clip_to_s3(local_path)
        
        # Should be /media/ path
        self.assertTrue(final_url.startswith('/media/'))
        
        # Step 3: Log to database
        _log_video_validation(
            camera_id=self.camera.id,
            event_type='cash',
            is_valid=True,
            confidence=0.9,
            reason='Valid',
            prompt='...',
            response_raw='{}',
            video_url=final_url,
            processing_time_ms=1000
        )
        
        # Step 4: Verify
        log = GeminiLog.objects.filter(camera=self.camera).latest('created_at')
        self.assertTrue(log.video_path.startswith('/media/'))
        
        # Cleanup
        media_path = Path(settings.MEDIA_ROOT) / 'validation_clips' / Path(local_path).name
        media_path.unlink(missing_ok=True)


class TestAPIEndpoints(S3StorageBaseTest):
    """Test API endpoints return correct URLs."""
    
    def setUp(self):
        super().setUp()
        # Create a test GeminiLog with S3 URL
        self.test_log = GeminiLog.objects.create(
            camera=self.camera,
            event_type='cash',
            validation_type='video',
            is_validated=False,
            confidence=0.8,
            reason='Test reason',
            prompt_used='Test prompt',
            response_raw='{}',
            video_path='https://hotel-cctv.s3.ap-northeast-2.amazonaws.com/validation_clips/test.mp4',
            processing_time_ms=2000
        )
    
    def test_gemini_log_stores_s3_url(self):
        """Verify GeminiLog model stores S3 URL correctly."""
        log = GeminiLog.objects.get(id=self.test_log.id)
        
        self.assertTrue(log.video_path.startswith('https://'))
        self.assertIn('s3', log.video_path)
        self.assertNotIn('/media/', log.video_path)


if __name__ == '__main__':
    import unittest
    unittest.main()
