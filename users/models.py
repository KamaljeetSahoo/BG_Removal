from django.db import models

# Create your models here.
class CostumerModel(models.Model):
    userid = models.CharField(max_length = 100)
    phoneno = models.CharField(max_length = 10, null = True)
    payment = models.BooleanField(default=False)
