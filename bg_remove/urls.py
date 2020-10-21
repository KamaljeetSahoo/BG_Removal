from django.urls import path
from .views import process_img

urlpatterns = [
    path('process_img/', process_img)
]
