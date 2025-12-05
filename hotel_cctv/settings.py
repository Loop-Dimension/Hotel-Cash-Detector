"""
Django settings for Hotel CCTV Monitoring System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-fallback-key-change-in-production')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv('DEBUG', 'True').lower() in ('true', '1', 'yes')

ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'cctv',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'hotel_cctv.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'cctv.context_processors.language_context',
                'cctv.context_processors.app_context',
            ],
        },
    },
]

WSGI_APPLICATION = 'hotel_cctv.wsgi.application'

# Database - SQLite
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Custom User Model
AUTH_USER_MODEL = 'cctv.User'

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = os.getenv('DEFAULT_LANGUAGE', 'ko')
if LANGUAGE_CODE == 'ko':
    LANGUAGE_CODE = 'ko-kr'
elif LANGUAGE_CODE == 'en':
    LANGUAGE_CODE = 'en-us'
    
TIME_ZONE = 'Asia/Seoul'
USE_I18N = True
USE_TZ = True

# Available languages
LANGUAGES = [
    ('ko', '한국어'),
    ('en', 'English'),
]

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Upload settings
DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024 * 1024  # 1GB
FILE_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024 * 1024  # 1GB

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Login settings
LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/login/'

# Detection settings
DETECTION_CONFIG = {
    # Models are now inside django_app/models/
    'MODELS_DIR': BASE_DIR / 'models',
    'INPUT_DIR': BASE_DIR / 'input',
    'OUTPUT_DIR': BASE_DIR / os.getenv('OUTPUT_DIR', 'output'),
    'UPLOAD_DIR': BASE_DIR / os.getenv('UPLOAD_DIR', 'uploads'),
    
    # Model paths from environment (relative to MODELS_DIR)
    'YOLO_MODEL': os.getenv('YOLO_MODEL_PATH', 'models/yolov8s.pt'),
    'POSE_MODEL': os.getenv('POSE_MODEL_PATH', 'models/yolov8s-pose.pt'),
    'FIRE_MODEL': os.getenv('FIRE_MODEL_PATH', 'models/fire_smoke_yolov8.pt'),
    
    # Default detection thresholds (can be overridden per camera)
    'CONFIDENCE_THRESHOLD': float(os.getenv('CASH_DETECTION_CONFIDENCE', '0.5')),
    'POSE_CONFIDENCE': 0.5,
    'HAND_TOUCH_DISTANCE': 80,
    'MIN_TRANSACTION_FRAMES': 1,
    
    # Violence detection defaults
    'VIOLENCE_CONFIDENCE': float(os.getenv('VIOLENCE_DETECTION_CONFIDENCE', '0.6')),
    'VIOLENCE_DURATION': 5,
    
    # Fire detection defaults
    'FIRE_CONFIDENCE': float(os.getenv('FIRE_DETECTION_CONFIDENCE', '0.5')),
    'MIN_FIRE_FRAMES': 4,
    'MIN_FIRE_AREA': 100,
    
    # Cashier zone defaults (can be set per camera)
    'DEFAULT_CASHIER_ZONE': [0, 0, 640, 480],
    
    # Processing
    'FRAME_SKIP': 2,
    'ALERT_COOLDOWN': 30,
    
    # RTSP settings
    'RTSP_TIMEOUT': int(os.getenv('RTSP_TIMEOUT', '10')),
    'RTSP_BUFFER_SIZE': int(os.getenv('RTSP_BUFFER_SIZE', '10')),
}

# Admin credentials from environment (for seed command)
ADMIN_CREDENTIALS = {
    'username': os.getenv('ADMIN_USERNAME', 'admin'),
    'email': os.getenv('ADMIN_EMAIL', 'admin@hotel.com'),
    'password': os.getenv('ADMIN_PASSWORD', 'admin123'),
}
