# Generated by Django 2.1.4 on 2020-10-22 21:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('payments', '0007_delete_final'),
    ]

    operations = [
        migrations.AddField(
            model_name='callback',
            name='user_id',
            field=models.TextField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='transaction',
            name='user_id',
            field=models.TextField(default=0),
            preserve_default=False,
        ),
    ]
