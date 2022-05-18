import os
import json

import torch 
import torchvision 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 
import matplotlib.image as image 
import numpy as np

import pandas as pd
import albumentations as A
import albumentations.pytorch
import cv2
import math

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch.nn as nn
import torch.nn.functional as F

# IMAGE_SIZE = 448
# IMAGE_SIZE = 224
# box more than this size, crop it
# if backbone is swin, size should be selected carefully
MAX_IMAGE_SIZE = 448

# adaptive resizing threshold for small image
# if area( geometric mean of W*H is smaller than this, resize in some range (1, 1.5)
SMALL_IMAGE_SIZE = 100.
range_limit = 0.5 # range(1, 1 + range limit)

train_transforms = A.Compose([
    A.RandomScale(scale_limit=.05, p=0.7),
    A.OneOf([
        A.HorizontalFlip(p=.8),
        A.VerticalFlip(p=.8),
        A.RandomRotate90(p=.8)]
    ),
    A.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05, p=.8),
], p=1.0) 

val_transforms = A.Compose([
    A.HorizontalFlip(p=.001),
], p=1.0) 

test_transforms = A.Compose([
    A.HorizontalFlip(p=.001),
], p=1.0) 

contra_transforms = A.Compose([
    A.RandomScale(scale_limit=.05, p=0.7),
    A.OneOf([
        A.HorizontalFlip(p=.8),
        A.VerticalFlip(p=.8),
        A.RandomRotate90(p=.8)]
    ),
    A.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=.7),
    A.ToGray(p=0.2),
    # transforms.ToTensor(),
    # normalize,
])

def label_mapper(label) :
    label = str(label)
    if label == 'AS' or 'ASC-US with HPV infection' in label:
        return 'ASC-US'
    elif label == 'AH' or 'ASC-H with HPV infection' in label:
        return 'ASC-H' 
    elif label == 'LS' or 'LSIL with HPV infection' in label:
        return 'LSIL'
    elif label == 'HS' or label == 'HN' or 'HSIL with HPV infection' in label :
        return 'HSIL'
    elif label == 'SM' or label == 'SC' :
        return 'Carcinoma'
    elif label == 'C' or label == 'T' or label == 'H' or label == 'AC' :
        return 'Benign'
    else :
        return label

def label_id(label) :
    label = str(label)
    if label == 'ASC-US' :
        return 0
    elif label == 'LSIL' :
        return 1
    elif label == 'HSIL' :
        return 2
    elif label == 'Negative' :
        return 3
    elif label == 'ASC-H' :
        return 4   
    elif label == 'Carcinoma' :
        return 5
    else : #others
        return 6

class PapsDataset(Dataset):
    def __init__(self, df, defaultpath='/home/Dataset/scl/', transform=None):
        self.df = df
        self.df.label = self.df.label.apply(lambda x : label_mapper(x))
        self.df.label = self.df.label.apply(lambda x : label_id(x))
        self.num_classes = len(self.df.label.unique()) - 1
        self.df = self.df[self.df['label'] != self.num_classes]
        self.image_mean = np.array([0.485, 0.456, 0.406])
        self.image_std = np.array([0.229, 0.224, 0.225])
        print(self.df.shape)
        
        self.transform = transform
        self.dir = defaultpath
        
        # sorting by size, batch should be consider size for compute efficiency and performance
        # I think batchnorm will be affected if image with lots of padding, therefore
        # custom sampler should check area
        self.df = self.df.sort_values('area', axis=0)
        
        self.df.reset_index(inplace=True, drop=False)

    def __len__(self):
        return len(self.df)   
    
    def get_roi(self, idx):
        path = self.df.loc[idx]['file_name']
        image = cv2.imread(self.dir + path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        xmin, ymin, w, h = self.df.loc[idx][['xmin', 'ymin', 'w', 'h']]
        xmax = xmin + w
        ymax = ymin + h

        image = image[ymin:ymax,xmin:xmax,:]
        return image
    
    def get_crop(self, image):
        
        #crop more than max_image_size
        x, y, _ = image.shape
        s_x = s_y = 0
        pl_x = pl_y = pr_x = pr_y = 0
        if x > MAX_IMAGE_SIZE :
            s_x = (x-MAX_IMAGE_SIZE)//2
        if y > MAX_IMAGE_SIZE :
            s_y = (y - MAX_IMAGE_SIZE)//2
            
#         crop the image to MAX_IMAGE_SIZE if image is larger than MAX_IMAGE_SIZE, by center
        image = image[s_x:MAX_IMAGE_SIZE+s_x, s_y:MAX_IMAGE_SIZE+s_y,:]
    
        # need to select real statistics 
        image = image/255.
        image = (image - self.image_mean[None, None, :]) / self.image_std[None, None, :]
        
        # do this in collate_fn
        # image =  torch.tensor(image, dtype=torch.float32)
        # image = image.permute(2,0,1)   
        
        return image
    
    def get_label(self, idx):
        return self.df.loc[idx]['label']

    def __getitem__(self, idx):
        label = self.get_label(idx)
        image = self.get_roi(idx)
        area = self.df.loc[idx]['area']
        
        if area < SMALL_IMAGE_SIZE :
            # adaptive image scaling, range and scaling is varing by area
            scale = 1 + float(((SMALL_IMAGE_SIZE/area)**2)*0.1 + np.random.randn(1) * 0.1)
            image = cv2.resize(image, dsize=(0,0), fx=scale, fy=scale)
        
#         if image is uint8, normalization by 255 is done automatically by albumebtation(ToTensor method)
        if self.transform:
            timage = self.transform(image=image)
            image = timage['image']
            
        # crop big size image
        image = self.get_crop(image)
        
        return image, label, area #, path

    
class ContraPapsDataset(PapsDataset):
    def __init__(self, df, defaultpath='/home/Dataset/scl/', transform=contra_transforms):
        super(ContraPapsDataset, self).__init__(df, defaultpath=defaultpath, transform=transform)

    def __getitem__(self, idx):
        image = self.get_roi(idx)
        label = self.get_label(idx)
        
#         if image is uint8, normalization by 255 is done automatically by albumebtation(ToTensor method)
        if self.transform:
            timage1 = self.transform(image=image)
            timage2 = self.transform(image=image)
            image1 = timage1['image']
            image2 = timage2['image']
            
#         crop or pad for fixed size (IMAGE_SIZE*IMAGE_SIZE*3) based on center
        # return image1, image2, label #, path 
        image1 = self.get_croppad(image1)
        image2 = self.get_croppad(image2)
        
        return image1, image2, label #, path 
    

# class PapsDataModule(LightningDataModule):
#     def __init__(self, data_dir: str = '../lbp_data/'):
#         super().__init__()
#         self.data_dir = data_dir
#         self.train_transform = train_transforms
#         self.test_transform = test_transforms
#         self.workers = workers

#         # self.dims is returned when you call dm.size()
#         # Setting default dims here because we know them.
#         # Could optionally be assigned dynamically in dm.setup()
#         self.dims = (1, 28, 28)
#         # self.num_classes = 5

#     def prepare_data(self):
#         # download
#         pass

#     def setup(self, stage=None):

#         # Assign train/val datasets for use in dataloaders
#         if stage == "fit" or stage is None:
#             train_df = pd.read_csv(self.data_dir + 'train.csv')
#             self.train_dataset = PapsDataset(train_df, defaultpath=self.data_dir, transform=self.train_transforms)
            
#         test_df = pd.read_csv(self.data_dir + 'test.csv')
#         self.test_dataset = PapsDataset(test_df, defaultpath=self.data_dir, transform=self.test_transforms)     

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset, batch_size=BATCH_SIZE,
#             shuffle=True, num_workers=self.workers
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.test_dataset, batch_size=BATCH_SIZE,
#             shuffle=False, num_workers=self.workers
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset, batch_size=BATCH_SIZE,
#             shuffle=False, num_workers=self.workers
#         )   