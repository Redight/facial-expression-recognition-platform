from unicodedata import name
from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home, name="home"),
    path('', views.gallery, name="gallery"),
    path('photo/<str:pk>/', views.viewPhoto, name="photo"),
    path('add/', views.addPhoto, name="add"),
    path('addvideo/', views.addVideo, name="addVideo")
]