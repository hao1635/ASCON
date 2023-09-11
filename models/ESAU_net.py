import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
import copy
from torch.nn.parameter import Parameter
import numbers
from einops import rearrange
from torch.nn import init
import os
import util.util as util
import ipdb


class Attention2d(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention2d, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        return out
    

class Attention_Block(nn.Module):
    def __init__(self,input_channel,output_channel,num_heads=8):
        super(Attention_Block,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.attention_s=Attention2d(dim=input_channel, num_heads=num_heads, bias=False)

    def forward(self, inputs):

        attn_s=self.attention_s(inputs)

        inputs_attn=inputs+attn_s

        return inputs_attn

class Conv_FFN(nn.Module):
    def __init__(self,input_channel,middle_channel,output_channel,res=True):
        super(Conv_FFN,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.conv_1=nn.Conv2d(input_channel,middle_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_2=nn.Conv2d(middle_channel,output_channel,kernel_size=3,stride=1,padding=1,bias=False)
        if self.input_channel != self.output_channel:
            self.shortcut=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=1,padding=0,stride=1,groups=1,bias=False)
        self.res=res
        self.act=nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        conv_S=self.act(self.conv_1(inputs))
        conv_S=self.act(self.conv_2(conv_S))

        if self.input_channel == self.output_channel:
            identity_out=inputs
        else:
            identity_out=self.shortcut(inputs)

        if self.res:
            output=conv_S+identity_out
        else:
            output=conv_S

        return output


class ESAU_Block(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads=8,res=True):
        super(ESAU_Block,self).__init__()
        self.esaublock=nn.Sequential(
            Attention_Block(in_channels,in_channels,num_heads=num_heads),
            Conv_FFN(in_channels,in_channels,out_channels,res=res),
        )
    def forward(self,x):
        return self.esaublock(x)
      
               
class Down(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads=8,res=True):
        super(Down,self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d((2,2), (2,2)),
            ESAU_Block(in_channels,out_channels,num_heads=num_heads,res=res)
        )
            
    def forward(self, x):
        return self.encoder(x)

    
class LastDown(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads=8,res=True):
        super(LastDown,self).__init__()

        self.encoder = nn.Sequential(
            nn.MaxPool2d((2,2), (2,2)),
            Attention_Block(in_channels,in_channels,num_heads=num_heads),
            Conv_FFN(in_channels,2*in_channels,out_channels,res=res),
            )
    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels,res_unet=True,trilinear=True, num_heads=8,res=True):
        super(Up,self).__init__()
        self.res_unet=res_unet
        if trilinear:
            self.up = nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels , kernel_size=2, stride=2)
        
        self.conv = ESAU_Block(in_channels, out_channels, num_heads=num_heads,res=res)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        
        if self.res_unet:
            x=x1+x2
        else:
            x = torch.cat([x2, x1], dim=1)

        return self.conv(x)



class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,decouple=None,bn=True,res=True,activation=False):
        super(SingleConv,self).__init__()
        self.act=activation
        self.conv =nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.activation = nn.LeakyReLU(inplace=True)
        

    def forward(self, x):
        x=self.conv(x)
        if self.act==True:
            x=self.activation(x)
        return x
        


class ESAU(nn.Module):
    def __init__(self,opt,in_channels=1,out_channels=1,n_channels=64,num_heads=[1,2,4,8],res=True):
        super(ESAU,self).__init__()
        #ipdb.set_trace()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        
        self.firstconv=SingleConv(in_channels, n_channels//2,res=res,activation=True)
        self.enc1 = ESAU_Block(n_channels//2, n_channels,num_heads=num_heads[0],res=res) 
        
        self.enc2 = Down(n_channels, 2 * n_channels,num_heads=num_heads[1],res=res)
        
        self.enc3 = Down(2 * n_channels, 4 * n_channels,num_heads=num_heads[2],res=res)
        
        self.enc4 = LastDown(4 * n_channels, 4 * n_channels,num_heads=num_heads[3],res=res)
        
        self.dec1 = Up(4 * n_channels, 2 * n_channels,num_heads=num_heads[2],res=res)
        
        self.dec2 = Up(2 * n_channels, 1 * n_channels,num_heads=num_heads[1],res=res)
        
        self.dec3 = Up(1 * n_channels, n_channels//2,num_heads=num_heads[0],res=res)

        self.out1 = SingleConv(n_channels//2,n_channels//2,res=res,activation=True)
        
        self.out2 = SingleConv(n_channels//2,out_channels,res=res,activation=False)


    
    def forward(self, x):
        b, c, h, w = x.size()

        x =self.firstconv(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        output = self.dec1(x4, x3)
        output = self.dec2(output, x2)
        output = self.dec3(output, x1)
        output = self.out1(output)

        output = self.out2(output)

        return output
