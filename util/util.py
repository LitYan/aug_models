from __future__ import print_function
import torch
import numpy as np
import skimage.morphology as sm
from PIL import Image
import os

# Converts a output tensor into a gray label map
def post_process(output_tensor,no_closing): #b,c,h,w->h,w
    out = torch.sigmoid(output_tensor[0])
    out_numpy = out.max(0)[1].cpu().numpy()
    out_numpy = np.array(out_numpy).astype('uint8')
    if not no_closing:
        out_numpy=sm.closing(out_numpy,sm.square(3))
    out_numpy_ske=sm.skeletonize(out_numpy)#骨架化
    out_numpy_ske=out_numpy_ske.astype('uint8')
    results={'out':out_numpy,'out_ske':out_numpy_ske}
    return results

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8, std=0.5, mean=0.5,nc=3):
    if isinstance(input_image, torch.Tensor):
        image_numpy = input_image[0].cpu().float().numpy()
    else:
        image_numpy = input_image
    if len(image_numpy.shape) == 2:#h,w->1,h,w
        image_numpy = image_numpy[None,:,:]
        std = 1
        mean = 0
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))*std+mean)*255.0#3,h,w->h,w,3
    if nc == 1:
        image_numpy = image_numpy[..., 0] * 0.299 + image_numpy[...,1] * 0.587 + image_numpy[...,2] * 0.114 #h,w,3->h,w
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
