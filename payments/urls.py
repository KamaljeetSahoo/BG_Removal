from django.urls import path, include
from .views import initiate_payment, callback
from users.views import costumerhomeview

app_name = "pay_urls"
urlpatterns = [
    path('pay/', initiate_payment, name='pay'),
    path('callback/', callback, name='callback'),
    path('', include('users.urls')),
]
