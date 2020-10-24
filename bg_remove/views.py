from django.shortcuts import render
import torch
from torchvision  import models
from PIL import Image
import numpy as np
import io
import sys
# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import base64
from .utils import VGG16, inference_once, inference_img_whole, dispart, generate_trimap, overlay_transparent

# Create your views here.
model = VGG16(stage = 1)
ckpt = torch.load('chcekpoint_py3.ckpt')
model.load_state_dict(ckpt['state_dict'], strict=True)
dlab = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

def process_img(request):
    image = request.FILES['input_img']
    img = Image.open(image) #PIL Image
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
    b = io.BytesIO()
    im.save(b, format='PNG')
    b = b.getvalue()
    b64_im = base64.b64encode(b)
    image_url = u'data:img/jpeg;base64,'+b64_im.decode('utf-8')

    return render(request, 'bg_remove/bg_remove.html', context={'image': image_url})

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
