"""
URL configuration for uploadproj project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('upload.urls')), # 루트 경로로 접속 시 upload 앱에서 처리
]


# 프로젝트 (uploadproj/urls.py) 최하단에 추가
from django.conf import settings
from django.conf.urls.static import static

# 업로드 폴더 위치를 설정한 파일의 위치
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)