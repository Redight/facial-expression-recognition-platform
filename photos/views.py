from django.shortcuts import render, redirect

import photos
from .models import Emotion, Photo

def gallery(request):
    emotion = request.GET.get('emotion')
    if emotion == None:
        photos = Photo.objects.all()
    else:
        photos = Photo.objects.filter(emotion__name=emotion)
    
    
    emotions = Emotion.objects.all()

    context = {'emotions': emotions, 'photos': photos}
    return render(request, 'photos/gallery.html', context)

def viewPhoto(request, pk):
    photo = Photo.objects.get(id=pk)
    return render(request, 'photos/photo.html', {'photo': photo})

def addPhoto(request):
    emotions = Emotion.objects.all()

    if request.method == 'POST':
        data = request.POST
        image = request.FILES.get('image')

        if data['emotion'] != 'none':
            emotion = Emotion.objects.get(id=data['emotion'])
        elif data['emotion_new'] != '':
            emotion, created = Emotion.objects.get_or_create(name=data['emotion_new'])
        else:
            emotion = None

        photo = Photo.objects.create(
            emotion = emotion,
            description = data['description'],
            image = image,
        )

        return  redirect('gallery')
    context = {'emotions': emotions}
    return render(request, 'photos/add.html', context)
