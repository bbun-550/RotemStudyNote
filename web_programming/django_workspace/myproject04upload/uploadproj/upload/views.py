from django.shortcuts import render

# Create your views here.
# upload/views.py

from rest_framework import generics
from .models import UploadedImage
from .serializers import UploadedImageSerializer

class UploadedImageListCreateView(generics.ListCreateAPIView):
    queryset = UploadedImage.objects.all().order_by('-uploaded_at')
    serializer_class = UploadedImageSerializer

# 또는 'upload/index.html' 경로에 따라
def index(request):
    return render(request, 'index.html')


class UploadedImageDestroyView(generics.DestroyAPIView):
    queryset = UploadedImage.objects.all()
    serializer_class = UploadedImageSerializer
    
    def delete(self, request, *args, **kwargs):
        print("DELETE request for:", kwargs.get("pk"))
        return super().delete(request, *args, **kwargs)