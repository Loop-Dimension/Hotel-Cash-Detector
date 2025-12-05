"""
URL configuration for Hotel CCTV Monitoring System
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('cctv.urls')),
]

# Serve media files (for development and simple production setups)
# For proper production, configure your web server (nginx/apache) to serve /media/
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
