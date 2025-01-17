from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login
from django.conf import settings
from .models import Transaction, Callback
from .paytm import generate_checksum, verify_checksum
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from users.models import CostumerModel
from users.views import userlogout

def costumerhomeview(request):
    return render(request, 'bg_remove/bg_remove.html')
def costumer_login(request):
    return render(request, 'costumers/login.html')

def payment_update(id):
    print("verification re pasila")
    stat = Callback.objects.get(order_id = id).message
    print("stat add hela : {}".format(stat))
    user_id = Transaction.objects.get(order_id = id).user_id
    print("user id from verification is {}".format(user_id))
    costumer_obj = CostumerModel.objects.get(userid = user_id)
    costumer_obj.payment = True
    costumer_obj.save()
    print("payment stats updated : {}".format(costumer_obj.payment))


def initiate_payment(request):
    if request.method == "GET":
        # return render(request, 'payments/pay.html')
        return render(request, 'costumers/login.html')
    try:

        username = request.POST['username']
        password = request.POST['password']
        amount = 1
        user = authenticate(request, username=username, password=password)
        if user is None:
            messages.add_message(request, messages.ERROR, 'Invalid Credentials')
            return redirect('users:costumer_login')
        if user is not None:
            auth_login(request=request, user=user)

    except:
        # return render(request, 'payments/pay.html', context={'error': 'Wrong Accound Details or amount'})
        return render(request, 'costumers/login.html', context={'error': 'Wrong Accound Details or amount'})
    user_id = username
    payment_stats = CostumerModel.objects.get(userid = user_id).payment
    if(payment_stats):
        return redirect('users:costumer_dash')

    transaction = Transaction.objects.create(made_by=user, amount=amount, user_id = user_id)
    transaction.save()
    merchant_key = settings.PAYTM_SECRET_KEY

    params = (
        ('MID', settings.PAYTM_MERCHANT_ID),
        ('ORDER_ID', str(transaction.order_id)),
        ('CUST_ID', str(transaction.made_by.email)),
        ('TXN_AMOUNT', str(transaction.amount)),
        ('CHANNEL_ID', settings.PAYTM_CHANNEL_ID),
        ('WEBSITE', settings.PAYTM_WEBSITE),
        # ('EMAIL', request.user.email),
        # ('MOBILE_N0', '9911223388'),
        ('INDUSTRY_TYPE_ID', settings.PAYTM_INDUSTRY_TYPE_ID),
        ('CALLBACK_URL', 'http://127.0.0.1:8000/'),
        # ('PAYMENT_MODE_ONLY', 'NO'),
    )

    paytm_params = dict(params)
    checksum = generate_checksum(paytm_params, merchant_key)

    transaction.checksum = checksum
    transaction.save()
    paytm_params['CHECKSUMHASH'] = checksum
    print('SENT: ', checksum)
    return render(request, 'payments/redirect.html', context=paytm_params)

@csrf_exempt
def callback(request):
    if request.method == 'POST':
        received_data = dict(request.POST)
        paytm_params = {}
        paytm_checksum = received_data['CHECKSUMHASH'][0]
        for key, value in received_data.items():
            if key == 'CHECKSUMHASH':
                paytm_checksum = value[0]
            else:
                paytm_params[key] = str(value[0])
        order_id = received_data['ORDERID'][0]
        print(order_id)
        print(type(order_id))
        call_back = Callback.objects.create(order_id = order_id)

        # Verify checksum
        is_valid_checksum = verify_checksum(paytm_params, settings.PAYTM_SECRET_KEY, str(paytm_checksum))
        if is_valid_checksum:
            received_data['message'] = "Checksum Matched"
        else:
            received_data['message'] = "Checksum Mismatched"
            return render(request, 'payments/callback.html', context=received_data)

        call_back.message = received_data['STATUS']
        print(received_data)

        call_back.save()
        if received_data['RESPCODE'][0] =='01':
            payment_update(order_id)
            return redirect('users:costumer_dash')
        # return render(request, 'payments/callback.html', context=received_data)
        return redirect('users:logout_user')
