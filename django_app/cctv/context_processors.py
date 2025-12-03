"""
Context processors for Hotel CCTV Monitoring System
"""
from django.conf import settings
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


def app_context(request):
    """Add app-wide context variables"""
    return {
        'app_version': '1.0.0',
        'debug_mode': settings.DEBUG,
    }
