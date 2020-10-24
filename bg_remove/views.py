from django.shortcuts import render
import torch
from torchvision  import models
from PIL import Image
import numpy as np
import io
import sys
import cv2
import base64
import os
from .utils import VGG16, inference_once, inference_img_whole, dispart, generate_trimap, overlay_transparent

# Create your views here.
model = VGG16(stage = 1)
ckpt = torch.load('chcekpoint_py3.ckpt')
model.load_state_dict(ckpt['state_dict'], strict=True)
dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

def url_generator(im):
    b = io.BytesIO()
    im.save(b, format='PNG')
    b = b.getvalue()
    b64_im = base64.b64encode(b)
    image_url = u'data:img/jpeg;base64,'+b64_im.decode('utf-8')
    return image_url

def resizer(inp, option):
    f_img = inp
    print(f_img)
    width, height = f_img.size
    print("before size : ({},{})".format(width, height))
    factor = option
    print(factor)
    width = int(width/factor)
    height = int(height/factor)
    f_img = f_img.resize((width, height), Image.ANTIALIAS)
    print("after calculated : ({},{})".format(width, height))
    return f_img


final_response={}
def process_img(request):
    if 'remover_button' in request.POST:
        final_response.clear()
        image = request.FILES['input_img']
        img = Image.open(image) #PIL Image
        uploaded_img = img
        img_arr = np.array(img) #numpy image
        trimap = generate_trimap(dlab, img, device=torch.device('cpu'))
        with torch.no_grad():
            pred_mattes = inference_img_whole(model, img_arr, trimap)

        pred_mattes = (pred_mattes * 255).astype(np.uint8)
        pred_mattes[trimap == 255] = 255
        pred_mattes[trimap == 0  ] = 0
        im = img.convert('RGB')
        alpha = Image.fromarray(pred_mattes)
        im.putalpha(alpha)
        image_url = url_generator(im)
        uploaded_img_url = url_generator(uploaded_img)
        context = {'image': image_url,'uploaded' : uploaded_img_url, 'inp' : im, 'option' : 1, 'quality' : 'HIGH'}
        final_response.update(context)

    if 'bg_swap_button' in request.POST:
        final_response.update(bg_swap = True)


    if "download_trans" in request.POST:
        final_response.update(d_trans = True)


    if 'low' in request.POST:
        final_response.update(option = 4)
        final_response.update(quality = 'LOW')
    if 'medium' in request.POST:
        final_response.update(option = 2)
        final_response.update(quality = 'MEDIUM')

    if 'high' in request.POST:
        final_response.update(quality = 'HIGH')

    if 'high' in request.POST or 'low' in request.POST or 'medium' in request.POST:
        inp = final_response['inp']
        option = final_response['option']
        t_img_1 = resizer(inp, option)
        t_img = url_generator(t_img_1)
        final_response.update(tran_img = t_img)


    if 'change_bg_button' in request.POST:
        inp = final_response['inp']
        bg = request.FILES['bg_img']
        overlay = np.array(inp)
        background = np.array(Image.open(bg))
        background = cv2.resize(background, (overlay.shape[1], overlay.shape[0]))
        x = overlay_transparent(background, overlay, 0, 0)
        output = Image.fromarray(x)
        bg_swap_url = url_generator(output)
        final_response.update(bg_change_url = bg_swap_url)
    return render(request, 'bg_remove/bg_remove.html', final_response )

def change_background(request):
    inp = request.FILES['input_img']
    bg = request.FILES['bg_img']
    overlay = np.array(Image.open(inp))
    background = np.array(Image.open(bg))
    background = cv2.resize(background, (overlay.shape[1], overlay.shape[0]))
    x = overlay_transparent(background, overlay, 0, 0)
    output = Image.fromarray(x)
    b = io.BytesIO()
    output.save(b, format='PNG')
    b = b.getvalue()
    b64_im = base64.b64encode(b)
    image_url = u'data:img/jpeg;base64,'+b64_im.decode('utf-8')
    return render(request, 'bg_remove/change_bg.html', context={'image': image_url})
