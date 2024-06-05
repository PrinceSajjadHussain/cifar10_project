# cifar10_app/models.py
from django.db import models

class Image(models.Model):
    image = models.ImageField(upload_to='uploads/')
    prediction = models.CharField(max_length=255, blank=True)
    
    def __str__(self):
        return self.image.name
