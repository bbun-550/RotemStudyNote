from django.db import models
import os

# Create your models here.


class UploadedImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    description = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def delete(self, *args, **kwargs):
        if self.image:
            if os.path.isfile(self.image.path):
                os.remove(self.image.path)
        super().delete(*args, **kwargs)
