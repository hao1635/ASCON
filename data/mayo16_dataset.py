#dataset
import os, glob, shutil
import numpy as np
import tqdm
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
import ipdb
from util import transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image


import matplotlib.pyplot as plt

def sorted_list(path): 
    tmplist = glob.glob(path) # finding all files or directories and listing them.
    tmplist.sort() # sorting the found list

    return tmplist

class Mayo16Dataset(BaseDataset):
    def __init__(self, opt):
        #ipdb.set_trace()
        train_transforms,val_transforms=self.get_transforms(opt)
        if 'train' in opt.phase:
            self.transforms = train_transforms
        if 'test' in opt.phase:
            self.transforms = val_transforms
        
        self.phase=opt.phase
        self.opt=opt

        self.q_path_list=sorted_list(opt.dataroot+'/'+opt.phase+'/quarter_1mm/*')
        self.f_path_list=sorted_list(opt.dataroot+'/'+opt.phase+'/full_1mm/*')

        self.A_size = len(self.q_path_list)  # get the size of dataset A
        self.B_size = len(self.f_path_list)



    def __getitem__(self, index):
        assert self.f_path_list[index].split('-')[-1]==self.q_path_list[index].split('-')[-1]

        f_data=np.load(self.f_path_list[index]).astype(np.float32)
        q_data = np.load(self.q_path_list[index]).astype(np.float32)


        if self.transforms is not None:
            q_data = self.transforms[0](q_data)
            f_data = self.transforms[1](f_data)
        
        #weights=self._get_weights(f_data)
        weights=0

        A = q_data
        B = f_data
        A_path=self.q_path_list[index]
        B_path=self.f_path_list[index]

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,'weights':weights}



    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def get_transforms(self,opt):
        GLOBAL_RANDOM_STATE = np.random.RandomState(47)
        seed = GLOBAL_RANDOM_STATE.randint(10000000)
        RandomState1=np.random.RandomState(seed)
        RandomState2=np.random.RandomState(seed)

        min_value = -1000
        max_value =  2000

        train_raw_transformer=transforms.Compose([
        #transforms.CropToFixed(RandomState1, size=(opt.crop_size, opt.crop_size)),
        transforms.RandomFlip(RandomState1),
        transforms.RandomRotate90(RandomState1),
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
        ])

        train_label_transformer=transforms.Compose([
        #transforms.CropToFixed(RandomState2, size=(opt.crop_size, opt.crop_size)),
        transforms.RandomFlip(RandomState2),
        transforms.RandomRotate90(RandomState2),
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
        ])

        val_raw_transformer=transforms.Compose([
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
        ])

        val_label_transformer=transforms.Compose([
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
        ])

        train_transforms=[train_raw_transformer,train_label_transformer]
        val_transforms=[val_raw_transformer,val_label_transformer]

        return train_transforms,val_transforms



