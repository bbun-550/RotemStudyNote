from django.urls import path
from . import views

urlpatterns = [
    path('', views.post_list, name='post_list'),
    path('write/', views.post_create, name='post_create'),
    path('view/<int:post_id>/', views.post_detail, name='post_detail'),
]
