# upload/urls.py

from django.urls import path
from .views import UploadedImageListCreateView, index, UploadedImageDestroyView

urlpatterns = [
    path('images/', UploadedImageListCreateView.as_view(), name='image-list-create'),
    path('images/<int:pk>/', UploadedImageDestroyView.as_view(), name='image-delete'),
    path('', index, name='index'),
]