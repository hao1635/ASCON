import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import math
from .ESAU_net import ESAU
import random
import glob
import os
import ipdb

import segmentation_models_pytorch as smp
from einops import rearrange


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[],opt=None,initialize_weights=False):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None

    #ipdb.set_trace()
    if netG == 'redcnn':
        net = RED_CNN(out_ch=96)
    elif netG == 'unet':
        net = UNet(n_channels=1, n_classes=1, bilinear=False)
    elif netG == 'ESAU':
        net=ESAU(opt,in_channels=1,out_channels=1,n_channels=64,num_heads=[1,2,4,8],res=True)
        # net=ESAU(opt,in_channels=1,out_channels=1,n_channels=64,num_heads_s=[1,2,4,8],
        #                 num_heads_t=1, decouple='(2+1)D_C',bn=False,res=True,attention_s=True,attention_t=False,
        #                 center_frame_idx=None,encode_only=False)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=initialize_weights)


def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None,initialize_weights=False):

    if netF == 'sample':
        net = PatchSampleF(opt,use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = PatchSampleF(opt,use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids,initialize_weights=initialize_weights)


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PatchSampleF(nn.Module):
    def __init__(self, opt, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.opt=opt
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.k_size= [int(i) for i in opt.k_size.split(',')]   
        #self.nce_layers = [int(i) for i in opt.nce_layers.split(',')]   
        mlp_hidden_size= 1024
        projection_size= 256
        self.mlp1 = nn.Sequential(
                nn.Linear(512, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, projection_size)
            )

        self.mlp2 = nn.Sequential(*[nn.Linear(64, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])


    def forward(self, patch_size,feats, num_patches,patch_ids=None,patch_local_ids=None,top_idxs=None,pixweght=None):
        #ipdb.set_trace()
        return_ids = []
        return_feats = []
        return_local_ids=[]
        return_top_idxs=[]
        k_num=[(k_size**2-1)//2 for k_size in self.k_size]


        for feat_id, feat in enumerate(feats):
            b=feat.shape[0]
            h=feat.shape[2]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) #BxHWxC
            #feat_reshape = feat
            dim=feat_reshape.shape[2]
            N_patches=num_patches[feat_id]
            #ipdb.set_trace()
            if patch_ids is not None:
                #ipdb.set_trace()
                patch_id = patch_ids[feat_id]
                local_id = patch_local_ids[feat_id]        
            else:
                if pixweght is not None:
                    pixweght_map=pixweght
                    if h<pixweght_map.shape[3]:
                        scale=int(pixweght_map.shape[3]//h)
                        pixweght_map=torch.nn.functional.avg_pool2d(pixweght_map, kernel_size=(scale,scale), stride=(scale,scale))      
                    pixweght_map=BinarizedF(BinarizedF(pixweght_map,0.05).sum(dim=0),0.05)
                    ids=pixweght_map.view(-1).nonzero().squeeze()
                    #print(ids.shape)
                    if ids.shape[0]< N_patches:
                        N_patches=ids.shape[0]
                    #print(ids.shape[0])
                    patch_id = np.array(random.sample(list(ids.detach().cpu().numpy()),N_patches ))
                
                else:
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(N_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                    #print(patch_id)

                local_id = get_local_index(feat_reshape,patch_id=patch_id, k_size=self.k_size[feat_id]).to(feat_reshape.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat_reshape.device)

            if top_idxs is not None:
                top_idx = top_idxs[feat_id]
            else:
                x_norm = nn.functional.normalize(feat_reshape, dim=-1)
                sim_matrix = x_norm[:,local_id] @ x_norm[:,patch_id].unsqueeze(2).transpose(-2,-1)
                top_idx = sim_matrix.squeeze().topk(k=k_num[feat_id],dim=-1)[1].view(-1,k_num[feat_id],1)

            x_loc = feat_reshape[:,local_id].view(-1,local_id.shape[1],dim)
            x_loc = torch.gather(x_loc, 1, top_idx.expand(-1, -1, dim))
            x_center = feat_reshape[:,patch_id].view(-1,1,dim)

            if h==patch_size//16: 
                x_sample = F.adaptive_avg_pool2d(torch.cat((x_center,x_loc),dim=1),(1,dim))

            if h==patch_size:
            #x_sample = feat_reshape[:, local_id, :] # b,nxpatch,c
                x_sample = torch.cat((x_center,x_loc),dim=1) #bxnxpatch,C

            x_sample = x_sample.reshape(-1,dim)

            if h==patch_size//16: 
                x_sample = self.mlp1(x_sample)

            if h==patch_size:
                x_sample = self.mlp2(x_sample)
                x_sample = self.l2norm(x_sample)
                #x_sample =F.normalize(x_sample, dim=1)
                x_sample = x_sample.reshape(b*N_patches,-1,x_sample.shape[-1]) 

            #x_sample = self.l2norm(x_sample)
            return_feats.append(x_sample)   
            return_ids.append(patch_id)
            return_local_ids.append(local_id)
            return_top_idxs.append(top_idx)

        return return_feats,return_ids,return_local_ids,return_top_idxs

def BinarizedF(input,threshold):
    a = torch.ones_like(input)
    b = torch.zeros_like(input)
    output = torch.where(input>threshold,a,b)
    return output


def get_local_index(x,patch_id, k_size):
    loc_weight = []
    w = torch.LongTensor(list(range(int(math.sqrt(x.shape[1]))))).to(x.device)
    if k_size<13:
        for i in patch_id:
            ix, iy = i//len(w), i%len(w)
            wx = torch.zeros(int(math.sqrt(x.shape[1]))).to(x.device)
            wy = torch.zeros(int(math.sqrt(x.shape[1]))).to(x.device)
            #ipdb.set_trace()
            wx[ix]=1
            wy[iy]=1
            for j in range(1,int(k_size//2)+1):
                wx[(ix+j)%len(wx)]=1
                wx[(ix-j)%len(wx)]=1
                wy[(iy+j)%len(wy)]=1
                wy[(iy-j)%len(wy)]=1
            weight = (wy.unsqueeze(0)*wx.unsqueeze(1)).view(-1)
            weight[i]=0
            loc_weight.append(weight.nonzero().squeeze())
    if k_size>13:
        #ipdb.set_trace()
        for i in patch_id:
            ix, iy = i//len(w), i%len(w)
            wx = torch.zeros(int(math.sqrt(x.shape[1]))).to(x.device)
            wy = torch.zeros(int(math.sqrt(x.shape[1]))).to(x.device)
            #ipdb.set_trace()
            wx[ix]=1
            wy[iy]=1
            for j in range(1,int(k_size//2)+1):
                wx[(ix+j)%len(wx)]=1
                wx[(ix-j)%len(wx)]=1
                wy[(iy+j)%len(wy)]=1
                wy[(iy-j)%len(wy)]=1
            weight1 = (wy.unsqueeze(0)*wx.unsqueeze(1)).view(-1)
            weight1[i]=0
            #loc_weight.append(weight.nonzero().squeeze())

            wx = torch.zeros(int(math.sqrt(x.shape[1]))).to(x.device)
            wy = torch.zeros(int(math.sqrt(x.shape[1]))).to(x.device)
            wx[ix]=1
            wy[iy]=1
            for j in range(1,int(5//2)+1):
                wx[(ix+j)%len(wx)]=1
                wx[(ix-j)%len(wx)]=1
                wy[(iy+j)%len(wy)]=1
                wy[(iy-j)%len(wy)]=1
            #ipdb.set_trace()
            weight2= (wy.unsqueeze(0)*wx.unsqueeze(1)).view(-1)
            weight2[i]=0
            weight=weight1-weight2
            loc_weight.append(weight.nonzero().squeeze())
    
    return torch.stack(loc_weight)



            ########################################################
'''UNET'''
            ########################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.in_channels=in_channels
        #self.attn=Attention_block(input_channel=in_channels,num_heads_s=8,attn=False)
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Disentangle_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Disentangle_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512// factor)
        #factor = 2 if bilinear else 1
        self.down4 = Down(256, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def standard_output(self, x):
        #ipdb.set_trace()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def forward(self, input, layers=[], encode_only=False):
        layers=[1,4]
        local_m= True
        global_m= True
        if len(layers) > 0 and encode_only:
            feats = []
            x1 = self.inc(input)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)  
            #ipdb.set_trace()
            x5 = self.down4(x4)  # 32
            if global_m==True:
                feats.append(x5)
            #nce 1
            #x = self.up1(x5, x4)
            if local_m==True:
            #nce 2
                x = self.up2(x4, x3)  #128

                #feats.append(x)
                #nce 3
                x = self.up3(x, x2)  # 256
                #feats.append(x)
                #nce 4
                x = self.up4(x, x1)  # 512
                feats.append(x)
            # if layer_id == layers[-1] and encode_only:
            #     # print('encoder only return features')
            #     return feats  # return intermediate features alone; stop in the last layers
            return feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.standard_output(input)
            return fake


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60], gamma=0.1)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=1e-6)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=False):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


