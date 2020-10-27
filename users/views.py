from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from .models import CostumerModel
from django_email_verification import sendConfirm


# Create your views here.
def adminloginview(request):
    return render(request, "admin/adminlogin.html")

def adminhomeview(request):
    return render(request, 'admin/admin_home.html')

def costumer_login(request):
    return render(request, 'costumers/login.html')

def costumer_register(request):
    return render(request, 'costumers/register.html')

@login_required
def costumerhomeview(request):
    return render(request, 'bg_remove/bg_remove.html')

def authenticate_admin(request):
    username = request.POST['username']
    password = request.POST['password']

    user = authenticate(username = username, password=password)

    if user is not None:
        login(request, user)
        return redirect('admin_home')

    if user is None:
        messages.add_message(request, messages.ERROR, 'Invalid Credentials')
        return redirect('admin_login')

def signup(request):
    username = request.POST['username']
    password = request.POST['pwd']
    r_password = request.POST['re_pwd']
    phone = request.POST['phone']
    email = request.POST['email']
    print(email, username, password)

    if username == '' or email == '':
        messages.add_message(request, messages.ERROR, 'Blank fields not accepted')
        return redirect('users:costumer_register')

    if password != r_password:
        messages.add_message(request, messages.ERROR, 'Password and Re-Enter password not matching')
        return redirect('users:costumer_register')

    if User.objects.filter(email = email).exists():
        messages.add_message(request, messages.ERROR, 'Email already exists')
        return redirect('users:costumer_register')

    if User.objects.filter(username = username).exists():
        messages.add_message(request, messages.ERROR, 'Username already Exists')
        return redirect('users:costumer_register')

    #user = get_user_model().objects.create(username=username, password=password, email=email)
    User.objects.create_user(username=username, password=password, email=email, is_active=False).save()
    user = get_user_model().objects.get(email=email)

    uid = User.objects.get(email = email).username
    print(uid)
    CostumerModel(userid = uid, phoneno=phone).save()
    sendConfirm(user)
    messages.add_message(request, messages.ERROR, 'Registration Succesful')
    return redirect('users:costumer_login')

def userlogin(request):
    username = request.POST['username']
    password = request.POST['password']

    user = authenticate(username = username, password=password)

    if user is not None:
        login(request, user)
        return redirect('users:costumer_dash')

    if user is None:
        messages.add_message(request, messages.ERROR, 'Invalid Credentials')
        return redirect('users:costumer_login')

def userlogout(request):
    logout(request)
    return redirect('users:costumer_login')
