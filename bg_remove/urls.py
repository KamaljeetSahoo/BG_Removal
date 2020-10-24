from django.urls import path
from .views import process_img, change_background

app_name = 'bg_remove'

urlpatterns = [
    path('process/', process_img, name = 'process'),
    path('change_bg/', change_background)
]
