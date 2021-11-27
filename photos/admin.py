from django.contrib import admin

# Register your models here.

from .models import Photo, Emotion


admin.site.register(Emotion)
admin.site.register(Photo)