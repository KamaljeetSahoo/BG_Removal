from django.urls import path
from .views import process_img, change_background

urlpatterns = [
    path('process_img/', process_img),
    path('change_bg/', change_background)
]
