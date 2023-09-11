"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
import torchvision
import torch.nn.functional as F
from kornia.filters import get_gaussian_kernel2d, filter2d
import logging
import shutil
from torch.nn.modules.loss import _Loss
import ipdb
import torch.nn as nn

import glob
import os
#import pywt
import random
import math

from typing import Tuple
#from pytorch_wavelets import DWTForward, DWTInverse 



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(**vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
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


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    _,h, w = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def correct_resize_label(t, size):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i, :1]
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = torch.from_numpy(np.array(one_image)).long()
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)


def correct_resize(t, size, mode=Image.BICUBIC):
    device = t.device
    t = t.detach().cpu()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = torchvision.transforms.functional.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return torch.stack(resized, dim=0).to(device)



def compute_ssim(img1, img2, window_size=11, reduction: str = "mean", max_val: float = 1.0, full: bool = False):
    window: torch.Tensor = get_gaussian_kernel2d(
        (window_size, window_size), (1.5, 1.5))
    window = window.requires_grad_(False)
    C1: float = (0.01 * max_val) ** 2
    C2: float = (0.03 * max_val) ** 2
    tmp_kernel: torch.Tensor = window.to(img1)
    tmp_kernel = torch.unsqueeze(tmp_kernel, dim=0)
    # compute local mean per channel
    mu1: torch.Tensor = filter2d(img1, tmp_kernel)
    mu2: torch.Tensor = filter2d(img2, tmp_kernel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # compute local sigma per channel
    sigma1_sq = filter2d(img1 * img1, tmp_kernel) - mu1_sq
    sigma2_sq = filter2d(img2 * img2, tmp_kernel) - mu2_sq
    sigma12 = filter2d(img1 * img2, tmp_kernel) - mu1_mu2

    ssim_map = ((2. * mu1_mu2 + C1) * (2. * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_score = ssim_map
    if reduction != 'none':
        ssim_score = torch.clamp(ssim_score, min=0, max=1)
        if reduction == "mean":
            ssim_score = torch.mean(ssim_score)
        elif reduction == "sum":
            ssim_score = torch.sum(ssim_score)
    if full:
        cs = torch.mean((2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        return ssim_score, cs
    return ssim_score


def compute_psnr(input: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError(f"Expected 2 torch tensors but got {type(input)} and {type(target)}")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    mse_val = F.mse_loss(input, target, reduction='mean')
    max_val_tensor: torch.Tensor = torch.tensor(max_val).to(input)
    return 10 * torch.log10(max_val_tensor * max_val_tensor / mse_val)


def compute_rmse(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(F.mse_loss(input, target))





def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def make_dir(path, refresh=False):
    
    """ function for making directory (to save results). """
    
    try: os.mkdir(path)
    except: 
        if(refresh): 
            shutil.rmtree(path)
            os.mkdir(path)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device="cuda"))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        loss = bmc_loss(pred, target, noise_var)
        return loss


def bmc_loss(pred, target, noise_var):
    b=pred.shape[0]
    # pred=F.normalize(pred)
    # target=F.normalize(target)
    pred=pred.reshape(b,-1)
    target=pred.reshape(b,-1)
    logits=-0.5 * euclidean_dist(pred,target)/ noise_var
    #logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    #ipdb.set_trace()
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]).cuda())
    loss = loss * (2 * noise_var).detach()

    return loss


def euclidean_dist(x, y):
    """
    Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
    Returns:
    dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist




# def wt_decomp(array, wt_type, wt_level, wt_pad):
#     ## padding
#     array_pad,padding = apply_wave_padding(array)

#     arr = pywt.wavedec2(array_pad, wavelet=wt_type, mode=wt_pad, level=wt_level)
#     arr[0] = np.zeros(arr[0].shape, dtype=np.float32)
#     arr_h = pywt.waverec2(arr, wavelet=wt_type, mode=wt_pad).astype(np.float32)

#     ## unpadding
#     (t, d), (l, r) = padding
#     arr_h = arr_h[t:-d, l:-r]

#     arr_l = array - arr_h
#     return arr_l, arr_h

# def apply_wave_padding(image: np.ndarray):
#     wavelet = pywt.Wavelet(name='db3')  # (name=self.opt.wavelet)
#     if wavelet.dec_len != wavelet.rec_len:
#         raise NotImplementedError('Padding assumes decomposition and reconstruction to have the same filter length')
#     assert image.ndim == 2, 'Image must be 2D.'
#     filter_len = wavelet.dec_len
#     level = 6  # self.opt.level
#     h, w = image.shape

#     # Extra length necessary to prevent artifacts due to separation of low and high frequencies.
#     # Size must be divisible by (2^level) for no shifting artifacts to occur.
#     # The final modulo ensures that divisible lengths add 0 instead of 2^level.
#     hh = ((2 ** level) - h % (2 ** level)) % (2 ** level)
#     ww = ((2 ** level) - w % (2 ** level)) % (2 ** level)

#     # Extra length necessary to prevent artifacts from kernel going over the edge into padding region.
#     # Padding size much be such that even the innermost decomposition is perfectly within the kernel.
#     # I have found that the necessary padding is filter_len, not (filter_len-1). The former is also usually even.
#     hh += filter_len * (2 ** level)
#     ww += filter_len * (2 ** level)

#     padding = ((hh // 2, hh - hh // 2), (ww // 2, ww - ww // 2))

#     return np.pad(image, pad_width=padding, mode='symmetric'),padding


# def get_local_weights(residual, ksize):

#     pad = (ksize - 1) // 2
#     residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')

#     unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
#     pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

#     return pixel_level_weight

# def get_refined_artifact_map(img_gt, img_output, img_ema, ksize):

#     residual_ema = torch.sum(torch.abs(img_gt - img_ema), 1, keepdim=True)
#     residual_SR = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)

#     patch_level_weight = torch.var(residual_SR.clone(), dim=(-1, -2, -3), keepdim=True) ** (1/5)
#     pixel_level_weight = get_local_weights(residual_SR.clone(), ksize)
#     overall_weight = patch_level_weight * pixel_level_weight

#     overall_weight[residual_SR < residual_ema] = 0

#     return overall_weight

# import ipdb
# def wt_decomp_torch(data):
#     #ipdb.set_trace()
#     datas=data.detach().squeeze().cpu().numpy()
#     datas_l=[]
#     datas_out=[]
#     for i in range(datas.shape[0]):
#         data_l, data_out = wt_decomp(datas[i,:,:], 'db3', 6, 'symmetric')
#         data_l=torch.from_numpy(data_l).unsqueeze(0)
#         data_out=torch.from_numpy(data_out).unsqueeze(0)
#         datas_l.append(data_l)
#         datas_out.append(data_out)


#     return torch.stack(datas_l,dim=0),torch.stack(datas_out,dim=0)



# def symm_pad(im: torch.Tensor, padding: Tuple[int, int, int, int]):
#      h, w = im.shape[-2:]
#      left, right, top, bottom = padding
 
#      x_idx = np.arange(-left, w+right)
#      y_idx = np.arange(-top, h+bottom)
 
#      def reflect(x, minx, maxx):
#          """ Reflects an array around two points making a triangular waveform that ramps up
#          and down,  allowing for pad lengths greater than the input length """
#          rng = maxx - minx
#          double_rng = 2*rng
#          mod = np.fmod(x - minx, double_rng)
#          normed_mod = np.where(mod < 0, mod+double_rng, mod)
#          out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
#          return np.array(out, dtype=x.dtype)

#      x_pad = reflect(x_idx, -0.5, w-0.5)
#      y_pad = reflect(y_idx, -0.5, h-0.5)
#      xx, yy = np.meshgrid(x_pad, y_pad)
#      return im[..., yy, xx]


# def pytorch_wt_decomp(array):
#     xfm = DWTForward(J=6, mode='symmetric', wave='db3').to(array.device)
#     ifm = DWTInverse(mode='symmetric', wave='db3').to(array.device)
#     ## padding
#     #ipdb.set_trace()
#     array_pad,padding = torch_wave_padding(array)
#     array_pad=array_pad
#     #ipdb.set_trace()
#     arr = list(xfm(array_pad))
    
#     arr[0] = torch.zeros(arr[0].shape).to(array.device)
#     arr_h = ifm(arr)
    
#     ## unpadding
#     t, d, l, r = padding
    
#     arr_h = arr_h[:,:,t:-d, l:-r]

#     arr_l = array - arr_h
#     #plt.imshow(arr_l,cmap='gray')
#     return arr_l, arr_h

# def torch_wave_padding(image: torch.tensor):
#     #ipdb.set_trace()
#     #wavelet = pywt.Wavelet(name='db3')  # (name=self.opt.wavelet)
#     assert image.ndim == 4, 'Image must be 2D.'
#     filter_len = 6
#     level = 6  # selarf.opt.level
#     _,_,h, w = image.shape

#     # Extra length necessary to prevent artifacts due to separation of low and high frequencies.
#     # Size must be divisible by (2^level) for no shifting artifacts to occur.
#     # The final modulo ensures that divisible lengths add 0 instead of 2^level.
#     hh = ((2 ** level) - h % (2 ** level)) % (2 ** level)
#     ww = ((2 ** level) - w % (2 ** level)) % (2 ** level)

#     # Extra length necessary to prevent artifacts from kernel going over the edge into padding region.
#     # Padding size much be such that even the innermost decomposition is perfectly within the kernel.
#     # I have found that the necessary padding is filter_len, not (filter_len-1). The former is also usually even.
#     hh += filter_len * (2 ** level)
#     ww += filter_len * (2 ** level)
    
#     padding = (hh // 2, hh - hh // 2, ww // 2, ww - ww // 2)
    
#     #plt.imshow(np.pad(image, pad_width=padding, mode='symmetric'),cmap='gray')
    
#     return symm_pad(image, padding=padding),padding



class SobelOperator(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        self.register_buffer('conv_x', torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None, None, :, :] / 4)
        self.register_buffer('conv_y', torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None, :, :] / 4)

    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)

        grad_x = F.conv2d(x, self.conv_x, bias=None, stride=1, padding=1)
        grad_y = F.conv2d(x, self.conv_y, bias=None, stride=1, padding=1)

        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        x = x.view(b, c, h, w)

        return x


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.sobel = SobelOperator(1e-4)

    def forward(self, pr, gt):
        gt_sobel = self.sobel(gt)
        pr_sobel = self.sobel(pr)
        grad_loss = F.l1_loss(gt_sobel, pr_sobel)
        return grad_loss