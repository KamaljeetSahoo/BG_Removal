import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image

import os
import numpy as np
import math
import cv2
import time

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.filters import sobel
from skimage.color import rgb2gray

class VGG16(nn.Module):
    def __init__(self, stage):
        super(VGG16, self).__init__()
        self.stage = stage

        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=3,stride = 1, padding=1,bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3,stride = 1, padding=1,bias=True)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1,bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1,bias=True)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1,bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1,bias=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1,bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)

        # model released before 2019.09.09 should use kernel_size=1 & padding=0
        #self.conv6_1 = nn.Conv2d(512, 512, kernel_size=1, padding=0,bias=True)
        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1,bias=True)

        self.deconv6_1 = nn.Conv2d(512, 512, kernel_size=1,bias=True)
        self.deconv5_1 = nn.Conv2d(512, 512, kernel_size=5, padding=2,bias=True)
        self.deconv4_1 = nn.Conv2d(512, 256, kernel_size=5, padding=2,bias=True)
        self.deconv3_1 = nn.Conv2d(256, 128, kernel_size=5, padding=2,bias=True)
        self.deconv2_1 = nn.Conv2d(128, 64, kernel_size=5, padding=2,bias=True)
        self.deconv1_1 = nn.Conv2d(64, 64, kernel_size=5, padding=2,bias=True)

        self.deconv1 = nn.Conv2d(64, 1, kernel_size=5, padding=2,bias=True)

        if stage == 2:
            # for stage2 training
            for p in self.parameters():
                p.requires_grad=False

        if self.stage == 2 or self.stage == 3:
            self.refine_conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=True)
            self.refine_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            self.refine_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
            self.refine_pred = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.conv1_1(x))
        x12 = F.relu(self.conv1_2(x11))
        x1p, id1 = F.max_pool2d(x12,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 2
        x21 = F.relu(self.conv2_1(x1p))
        x22 = F.relu(self.conv2_2(x21))
        x2p, id2 = F.max_pool2d(x22,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 3
        x31 = F.relu(self.conv3_1(x2p))
        x32 = F.relu(self.conv3_2(x31))
        x33 = F.relu(self.conv3_3(x32))
        x3p, id3 = F.max_pool2d(x33,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 4
        x41 = F.relu(self.conv4_1(x3p))
        x42 = F.relu(self.conv4_2(x41))
        x43 = F.relu(self.conv4_3(x42))
        x4p, id4 = F.max_pool2d(x43,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 5
        x51 = F.relu(self.conv5_1(x4p))
        x52 = F.relu(self.conv5_2(x51))
        x53 = F.relu(self.conv5_3(x52))
        x5p, id5 = F.max_pool2d(x53,kernel_size=(2,2), stride=(2,2),return_indices=True)

        # Stage 6
        x61 = F.relu(self.conv6_1(x5p))

        # Stage 6d
        x61d = F.relu(self.deconv6_1(x61))

        # Stage 5d
        x5d = F.max_unpool2d(x61d,id5, kernel_size=2, stride=2)
        x5d = x5d + x53
        x51d = F.relu(self.deconv5_1(x5d))

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x4d = x4d + x43
        x41d = F.relu(self.deconv4_1(x4d))

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x3d = x3d + x33
        x31d = F.relu(self.deconv3_1(x3d))

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x2d = x2d + x22
        x21d = F.relu(self.deconv2_1(x2d))

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x1d = x1d + x12
        x12d = F.relu(self.deconv1_1(x1d))

        # Should add sigmoid? github repo add so.
        raw_alpha = self.deconv1(x12d)
        pred_mattes = F.sigmoid(raw_alpha)

        if self.stage <= 1:
            return pred_mattes, 0

        # Stage2 refine conv1
        refine0 = torch.cat((x[:, :3, :, :], pred_mattes),  1)
        refine1 = F.relu(self.refine_conv1(refine0))
        refine2 = F.relu(self.refine_conv2(refine1))
        refine3 = F.relu(self.refine_conv3(refine2))
        # Should add sigmoid?
        # sigmoid lead to refine result all converge to 0...
        #pred_refine = F.sigmoid(self.refine_pred(refine3))
        pred_refine = self.refine_pred(refine3)

        pred_alpha = F.sigmoid(raw_alpha + pred_refine)

        #print(pred_mattes.mean(), pred_alpha.mean(), pred_refine.sum())

        return pred_mattes, pred_alpha

# inference once for image, return numpy
def inference_once(model, scale_img, scale_trimap, aligned=True, stage=1):

    if aligned:
        assert(scale_img.shape[0] == 320)
        assert(scale_img.shape[1] == 320)

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
    ])

    scale_img_rgb = cv2.cvtColor(scale_img, cv2.COLOR_BGR2RGB)
    # first, 0-255 to 0-1
    # second, x-mean/std and HWC to CHW
    tensor_img = normalize(scale_img_rgb).unsqueeze(0)

    #scale_grad = compute_gradient(scale_img)
    #tensor_img = torch.from_numpy(scale_img.astype(np.float32)[np.newaxis, :, :, :]).permute(0, 3, 1, 2)
    tensor_trimap = torch.from_numpy(scale_trimap.astype(np.float32)[np.newaxis, np.newaxis, :, :])
    #tensor_grad = torch.from_numpy(scale_grad.astype(np.float32)[np.newaxis, np.newaxis, :, :])


    tensor_img = tensor_img.cuda()
    tensor_trimap = tensor_trimap.cuda()
    #tensor_grad = tensor_grad.cuda()
    #print('Img Shape:{} Trimap Shape:{}'.format(img.shape, trimap.shape))

    input_t = torch.cat((tensor_img, tensor_trimap / 255.), 1)

    # forward
    if stage <= 1:
        # stage 1
        pred_mattes, _ = model(input_t)
    else:
        # stage 2, 3
        _, pred_mattes = model(input_t)
    pred_mattes = pred_mattes.data

    pred_mattes = pred_mattes.cpu()
    pred_mattes = pred_mattes.numpy()[0, 0, :, :]
    return pred_mattes


# forward a whole image
def inference_img_whole(model, img, trimap):
    h, w, c = img.shape
    new_h = min(1600, h - (h % 32))
    new_w = min(1600, w - (w % 32))

    # resize for network input, to Tensor
    scale_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale_trimap = cv2.resize(trimap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pred_mattes = inference_once(model, scale_img, scale_trimap, aligned=False)

    # resize to origin size
    origin_pred_mattes = cv2.resize(pred_mattes, (w, h), interpolation = cv2.INTER_LINEAR)
    assert(origin_pred_mattes.shape == trimap.shape)
    return origin_pred_mattes

def dispart(deeplab, segments, cluster_type=quickshift, n_class=2):
    if cluster_type == 'watershed':
        a = len(np.unique(segments)) + 1
    else:
        a = len(np.unique(segments))
    b = n_class

    clust_stat = np.zeros((a, b))

    for i in range(deeplab.shape[0]):
        for j in range(deeplab.shape[1]):
            clust_stat[segments[i,j],deeplab[i,j]]+=1

    clust_select = np.argmax(clust_stat, axis=1)
    final_seg = np.zeros((deeplab.shape[0], deeplab.shape[1]))

    for i in range(deeplab.shape[0]):
        for j in range(deeplab.shape[1]):
            final_seg[i, j] = clust_select[segments[i, j]]

    return final_seg.astype('int16')

def generate_trimap(model, image_path, kernel_size=10, device='cuda', dispart_mode = None):
    model = model.to(device).eval()
    img = Image.open(image_path)

    trf = T.Compose([T.ToTensor(),
                     T.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])])

    inp = trf(img).unsqueeze(0).to(device)

    if device == 'cuda':
        torch.cuda.empty_cache()

    with torch.no_grad():
        om = model(inp)['out'][0]
        om = torch.softmax(om.squeeze(), 0)
    om = om.detach().cpu().numpy()

    del inp
    if device == 'cuda':
        torch.cuda.empty_cache()

    label_colors = np.array([(0, 0, 0),  # 0=background
            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
            (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
            # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
            (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
            # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
            (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
            # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
            (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(om[0,:,:]).astype(np.uint8)
    r_ = np.zeros_like(om[0,:,:]).astype(np.uint8)
    g = np.zeros_like(om[0,:,:]).astype(np.uint8)
    g_ = np.zeros_like(om[0,:,:]).astype(np.uint8)
    b = np.zeros_like(om[0,:,:]).astype(np.uint8)
    b_ = np.zeros_like(om[0,:,:]).astype(np.uint8)


    for l in range(0, 21):
        idx = om[l,:,:] > 0.05
        idx_ = om[l,:,:] > 0.95
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        r_[idx_] = label_colors[l, 0]
        g_[idx_] = label_colors[l, 1]
        b_[idx_] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    rgb_ = np.stack([r_, g_, b_], axis=2)

    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    _,rgb_bw = cv2.threshold(rgb_gray,10,255,cv2.THRESH_BINARY)

    rgb_fg_gray = cv2.cvtColor(rgb_, cv2.COLOR_BGR2GRAY)
    _,rgb_fg_bw = cv2.threshold(rgb_fg_gray,10,255,cv2.THRESH_BINARY)

    img = np.array(Image.open(image_path))

    #img_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    #img_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    #img_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    img_quick = quickshift(img, kernel_size=1, max_dist=6, ratio=0.5,sigma=0)
    gradient = sobel(rgb2gray(img))
    img_watershed = watershed(gradient, markers=250, compactness=0.001)

    if dispart_mode == 'fg':
        temp = rgb_fg_bw/255
        seg_ = dispart(temp.astype('int16'), img_quick)
        seg_ = seg_ * 255

        pixels = 2 * kernel_size + 1
        kernel = np.ones((pixels, pixels), np.uint8)

        dilation = cv2.dilate(rgb_bw, kernel, iterations=1)

        trimap = np.zeros_like(rgb_bw)  # bg
        trimap[dilation == 255] = 127   # confusion
        trimap[seg_  == 255] = 255     #fg

        return trimap

    if dispart_mode == 'bg':
        temp = rgb_bw/255
        seg_ = dispart(temp.astype('int16'), img_quick)
        seg_ = seg_ * 255

        pixels = 2 * kernel_size + 1
        kernel = np.ones((pixels, pixels), np.uint8)

        dilation = cv2.dilate(seg_, kernel, iterations=1)

        trimap = np.zeros_like(seg_)  # bg
        trimap[dilation == 255] = 127   # confusion
        trimap[rgb_fg_bw  == 255] = 255     #fg

        return trimap

    if dispart_mode == 'fg_bg':
        temp = rgb_fg_bw/255
        seg_fg = dispart(temp.astype('int16'), img_quick)
        seg_fg = seg_fg * 255

        temp = rgb_bw/255
        seg_bg = dispart(temp.astype('int16'), img_quick)
        seg_bg = seg_bg * 255

        pixels = 2 * kernel_size + 1
        kernel = np.ones((pixels, pixels), np.uint8)

        dilation = cv2.dilate(seg_bg, kernel, iterations=1)

        trimap = np.zeros_like(seg_bg)  # bg
        trimap[dilation == 255] = 127   # confusion
        trimap[seg_fg  == 255] = 255     #fg

        return trimap

    else:
        pixels = 2 * 10 + 1
        kernel = np.ones((pixels, pixels), np.uint8)

        dilation = cv2.dilate(rgb_bw, kernel, iterations=1)

        remake = np.zeros_like(rgb_bw)
        remake[dilation == 255] = 127
        remake[rgb_fg_bw  == 255] = 255

        return remake
