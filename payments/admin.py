from django.contrib import admin
from payments.models import Transaction, Callback

admin.site.register(Transaction)
admin.site.register(Callback)


# Register your models here.
