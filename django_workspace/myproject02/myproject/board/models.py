from django.db import models

# Create your models here.
class Post(models.Model):
    title = models.CharField(max_length=100)
    author = models.CharField(max_length=50)
    content = models.TextField()
    url = models.URLField(blank=True)
    email = models.EmailField()
    created_at = models.DateTimeField(auto_now_add=True)
    

    # def __str__(self):
    #     return self.title