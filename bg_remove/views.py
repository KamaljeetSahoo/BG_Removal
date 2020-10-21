from django.shortcuts import render
import torch
from torchvision  import models
from PIL import Image
import numpy as np
import io
import base64
from .utils import VGG16, inference_once, inference_img_whole, dispart, generate_trimap

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
