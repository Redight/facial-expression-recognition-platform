from django.db import models
from django.db.models.fields.related import ForeignKey

# Create your models here.

class Photo(models.Model):
    emotion = models.CharField(max_length=100, null=False, blank=False)
    image = models.ImageField(null=False, blank=False)


class Video(models.Model):
    emotion = models.CharField(max_length=100, null=False, blank=False)
    video = models.FileField(null=False, blank=False)

