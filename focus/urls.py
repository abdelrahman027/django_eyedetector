from django.urls import path
from . import views

urlpatterns = [
    path('video_feed/', views.video_feed, name='video_feed'),
    path('live_feed/', views.live_feed, name='live_feed'),
    path('',views.index,name="index"),
    path('age-check',views.ageCheck,name="age_check"),

]
