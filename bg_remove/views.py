from django.shortcuts import render
import torch
from torchvision  import models
from PIL import Image
import numpy as np
from .utils import VGG16, inference_once, inference_img_whole, dispart, generate_trimap

# Create your views here.
# model = VGG16(stage = 1)
# ckpt = torch.load('/content/drive/My Drive/BackGround_Subtraction/RemoveBG/chcekpoint_py3.ckpt')
# model.load_state_dict(ckpt['state_dict'], strict=True)
# dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

def process_img(request):
    # image = request.FILES['input_img']
    # img = Image.open(image)
    # img_arr = np.array(img)
    pass
