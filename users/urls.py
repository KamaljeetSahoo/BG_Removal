from django.urls import path
from django_email_verification import urls as mail_urls
from .views import adminloginview, adminhomeview, authenticate_admin, costumer_login, costumer_register, signup, userlogin, costumerhomeview, userlogout
from payments.views import initiate_payment

app_name = "users"
urlpatterns = [
    #path('admin/', admin.site.urls),
    #path('admin_mod/', adminloginview, name='admin_login'),
    #path('admin_home/', adminhomeview, name='admin_home'),
    #path('admin_mod/authenticate/', authenticate_admin),
    path('users/login/', costumer_login, name='costumer_login'),
    path('users/register/', costumer_register, name='costumer_register'),
    path('users/register/signup/', signup, name='user_signup'),
    path('users/login/authenticate/', initiate_payment, name= 'authenticate'),
    path('', costumerhomeview, name='costumer_dash'),
    path('logout/', userlogout, name = "logout_user")
]
