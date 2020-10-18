from django.urls import path
from .views import adminloginview, adminhomeview, authenticate_admin, costumer_login, costumer_register, signup, userlogin, costumerhomeview

urlpatterns = [
    #path('admin/', admin.site.urls),
    path('admin_mod/', adminloginview, name='admin_login'),
    path('admin_home/', adminhomeview, name='admin_home'),
    path('admin_mod/authenticate/', authenticate_admin),
    path('users/login', costumer_login, name='costumer_login'),
    path('users/register', costumer_register, name='costumer_register'),
    path('users/signup', signup),
    path('users/login', userlogin),
    path('', costumerhomeview)
]
