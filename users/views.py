from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from .models import CostumerModel

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
    return render(request, 'costumers/costumerhome.html')

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

    if username == '' and email == '':
        messages.add_message(request, messages.ERROR, 'Blank fields not accepted')
        return redirect('costumer_register')

    if password != r_password:
        messages.add_message(request, messages.ERROR, 'Password and Re-Enter password not matching')
        return redirect('costumer_register')

    if User.objects.filter(email = email, username=username).exists():
        messages.add_message(request, messages.ERROR, 'Email already exists')
        return redirect('costumer_register')

    User.objects.create_user(username=username, password=password, email=email).save()
    lastobject = int(len(User.objects.all())-1)
    CostumerModel(userid = User.objects.all()[lastobject].id, phoneno=phone).save()
    messages.add_message(request, messages.ERROR, 'Registration Succesful')
    return redirect('costumer_login')

def userlogin(request):
    username = request.POST['username']
    password = request.POST['password']

    user = authenticate(username = username, password=password)

    if user is not None:
        login(request, user)
        return redirect('admin_home')

    if user is None:
        messages.add_message(request, messages.ERROR, 'Invalid Credentials')
        return redirect('admin_login')
