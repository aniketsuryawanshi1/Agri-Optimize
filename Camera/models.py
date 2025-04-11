from django.db import models

class Photo(models.Model):
    image_data = models.ImageField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.uploaded_at)
    

