from django.shortcuts import render, redirect

#Debug
import logging
from PIL import Image

#Loading model
from keras.models import load_model
#Files 
import cv2
import numpy as np
import os

import pandas as pd

from .models import Photo, Video

import matplotlib.pyplot as plt

def gallery(request):
    photos = Photo.objects.all()

    context = {'photos': photos}
    return render(request, 'photos/gallery.html', context)

def viewPhoto(request, pk):
    photo = Photo.objects.get(id=pk)
    return render(request, 'photos/photo.html', {'photo': photo})

class EmotionsDLModel:
    def load_keras_model():
        model = load_model('static/model.h5')
        #logging.debug(model.summary())
        return model

    def predictEmotion(images):
        emotions_with_images = []
        for image in images:
                image = cv2.imdecode(np.frombuffer(image.read(), dtype=np.int8), cv2.IMREAD_UNCHANGED)
                # Convert into grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
                    
                # Detect faces
                faces = face_cascade.detectMultiScale(gray_image, 1.1, minNeighbors=4)
                logging.debug(f"Number of faces in the image : {len(faces)}")
                list_of_faces = []
                #logging.debug(faces)
                #logging.debug("OK")
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x+w, y+h), 
                                    (0, 0, 255), 2)
                        
                    face = image[y:y + h, x:x + w]
                    list_of_faces.append(face)
                    #cv2.imwrite('face.jpg', faces)
                
                img_data = np.asarray(image, dtype="int32" )
                img_data = np.expand_dims(image, axis = 0)
                model = EmotionsDLModel.load_keras_model()
                
                #logging.debug(list_of_faces)
                #logging.debug(image.shape)
                image = np.expand_dims(image, 0)
                #logging.debug(image.shape)
                list_cropped_faces = []
                for f in list_of_faces:
                    #logging.debug(f.shape)
                    res = cv2.resize(f, dsize=(48, 48), interpolation=cv2.INTER_CUBIC)
                    #logging.debug(res.shape)
                    list_cropped_faces.append(np.expand_dims(res[:, :, 0], 0)/255)
                    break
                
                del list_of_faces
                for face in list_cropped_faces:
                    classify = model.predict(face)
                    max_index = np.argmax(classify[0])
                    emotion_array = ["angry", "disgust","fear","happy","sad","surprise","neutral"]
                    f_image = Image.fromarray(np.uint8(face[0, :, :] * 255) , 'L')
                    save_image = Image.fromarray(np.uint8(gray_image) , 'L')
                    # PIL IMAGE
                    emotions_with_images.append([save_image, emotion_array[max_index]])
                    break
        logging.debug("Hello")
        return emotions_with_images

                
    def predictEmotionVideo(video):
        obj = Video.objects.get(video=video)
        logging.debug(obj.video.__dict__)
        cap= cv2.VideoCapture("static/images/" + obj.video.name)
                
        emotion_array = ["angry", "disgust","fear","happy","sad","surprise","neutral"]

        # Load the cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        # To capture video from webcam.   
        ndetect = 0
        emotions = []
        model = EmotionsDLModel.load_keras_model()

        while True:  
            # Read the frame  
            _, img = cap.read()
            if img is None: # stop if end of video
                break
            # Convert to grayscale  
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            # Detect the faces  
            faces = face_cascade.detectMultiScale(gray, 1.1, 10)  
        
            # Draw the rectangle around each face

            for (x, y, w, h) in faces:  
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                res = cv2.resize(face, dsize=(48, 48), interpolation=cv2.INTER_CUBIC) / 255.
                pred = model.predict(res[None, :, :, None])
                max_index = np.argmax(pred[0])
                emotions.append(emotion_array[max_index])
                ndetect += 1
                emotion_array[max_index]
                

    
        # Release the VideoCapture object  
        cap.release()
        return emotions

def addPhoto(request):

    if request.method == 'POST':
        data = request.POST
        images = request.FILES.getlist('images')
        logging.debug(images)
        emotions_with_images = EmotionsDLModel.predictEmotion(images)
        for i in emotions_with_images:
            image = i[0]
            emotion = i[1]
            logging.debug(i[0])
            logging.debug(i[1])
        
        if emotion != None:    
            for image in images:
                photo = Photo.objects.create(
                    emotion = emotion,
                    image = image,
                )

            return  redirect('gallery')
        else:
            #if emotion is empty
            return redirect('failed')
    return render(request, 'photos/add.html')

def addVideo(request):

    if request.method == 'POST':
        video = request.FILES.get('video')
        video = Video.objects.create(
            video = video,
        )
        
        emotions = EmotionsDLModel.predictEmotionVideo(video.video)
        plt.bar(pd.value_counts(emotions).index, pd.value_counts(emotions))
        plt.show()
        
        emotion_to_label =  {"angry":0, "disgust": 1,"fear" : 2,"happy":3,"sad": 4,"surprise":5,"neutral":6}
        labs = [emotion_to_label[e] for e in  emotions]
        plt.figure(figsize= (18, 10))
        plt.plot(labs)
        plt.yticks(range(7), emotion_to_label.keys())
        plt.show()

        return  redirect('gallery')
    return render(request, 'photos/addVideo.html')

def home(request):
    return render(request, 'photos/home.html')

def failed(request):
    return render(request, 'photos/failed.html')
