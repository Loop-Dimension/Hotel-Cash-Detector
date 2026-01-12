"""
Context processors for Hotel CCTV Monitoring System
"""
from django.conf import settings
from django.utils import timezone
from .translations import get_translation


def language_context(request):
    """Add language and translations to template context"""
    # Get language from session, cookie, or default
    lang = request.session.get('lang', request.COOKIES.get('lang', 'ko'))
    
    # Validate language
    if lang not in ['ko', 'en']:
        lang = 'ko'
    
    # Get translations
    translations = get_translation(lang)
    
    return {
        'current_lang': lang,
        'translations': translations,
        't': translations,  # Shorthand alias
        'available_langs': [
            {'code': 'ko', 'name': '한국어'},
            {'code': 'en', 'name': 'English'},
        ],
    }


def server_time_context(request):
    """Add current server time to template context (Korean timezone)"""
    now = timezone.now()
    local_now = timezone.localtime(now)
    
    return {
        'server_time': local_now,
        'server_time_str': local_now.strftime('%Y-%m-%d %H:%M:%S'),
        'server_time_korean': local_now.strftime('%Y년 %m월 %d일 %H시 %M분'),
        'server_timezone': 'KST (UTC+9)',
        'server_timezone_full': str(timezone.get_current_timezone()),
    }


def app_context(request):
    """Add app-wide context variables"""
    return {
        'app_version': '1.0.0',
        'debug_mode': settings.DEBUG,
    }
