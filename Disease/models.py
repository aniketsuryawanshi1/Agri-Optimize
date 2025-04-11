# Create your models here.
from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='img/')
    
