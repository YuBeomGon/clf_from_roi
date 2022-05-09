# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/imagenet.py


import argparse
import os
from typing import Optional
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchmetrics import Accuracy, F1Score, Specificity

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import Trainer

from utils.dataset import PapsDataset, ContraPapsDataset, train_transforms, val_transforms, test_transforms, contra_transforms, IMAGE_SIZE
from utils.losses import SupConLoss, FocalLoss
import custom_models
# from models.efficientnet import EfficientNet, VALID_MODELS

parser = argparse.ArgumentParser(description='PyTorch Lightning ImageNet Training')
parser.add_argument('--data_path', metavar='DIR', default='./lbp_data/',
                    help='path to dataset (default: ./lbp_data/)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: (default: resnet18)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--accelerator', '--accelerator', default='gpu', type=str, help='default: gpu')

parser.add_argument('--devices', '--devices', default=4, type=int, help='number of gpus, default 2')
parser.add_argument('--img_size', default=400, type=int, help='input image resolution in swin models')
parser.add_argument('--num_classes', default=5, type=int, help='number of classes')
parser.add_argument('--saved_dir', default='./saved_models/tunning', type=str, help='directory for model checkpoint')
parser.add_argument('--resume', default='./saved_models/contra/model_last.ckpt', type=str, help='directory for model checkpoint')
# parser.add_argument('--is_contra', default=False, type=bool, help='supervised contrastive learning or not')


class PapsClsModel(LightningModule) :
    def __init__(
        self,
        data_path : str,
        arch: str = 'resnet18',
        pretrained: bool = False,
        lr: float = 0.9,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        batch_size: int =256,
        workers: int = 16,
        num_classes: int = 5,
        resume : str = './saved_models/contra',
        # is_contra: bool = False,
    ):
        
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self.num_classes = num_classes
        self.resume = resume
        # self.is_contra = is_contra
        
        if args.arch not in models.__dict__.keys() : 
            # self.model = EfficientNet.from_name(args.arch)  
            self.model = custom_models.__dict__[self.arch](pretrained=False, img_size=args.img_size)
        else :
            print('only resnet is supported') 
            self.model = models.__dict__[self.arch](pretrained=self.pretrained) 
            
        if self.resume :
            print('checkpoint is loaded from ', self.resume)
            self.model = model.load_from_checkpoint(self.resume)
        
        shape = self.model.fc.weight.shape
        self.model.fc = nn.Linear(shape[1], self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
            
        print("=> creating model '{}'".format(args.arch))
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.train_acc1 = Accuracy(top_k=1)
        self.eval_acc1 = Accuracy(top_k=1)
        self.f1 = F1Score(average='macro', num_classes=self.num_classes)
        self.specificity = Specificity(average='macro', num_classes=self.num_classes)
        
    def forward(self, x) :
        return self.model(x)

    
    def training_step(self, batch, batch_idx) :
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss)

        #update metric
        self.train_acc1(outputs, targets)
        self.log('train_acc', self.train_acc1, prog_bar=True)
        
        return loss
    
    def eval_step(self, batch, batch_idx, prefix: str) :
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log(f'{prefix}_loss', loss)
        
        self.eval_acc1(outputs, targets)
        self.log(f'{prefix}_acc1', self.eval_acc1, prog_bar=True)
        self.f1(outputs, targets)
        self.log(f'{prefix}_f1_score', self.f1, prog_bar=True)
        self.specificity(outputs, targets)
        self.log(f'{prefix}_specificity', self.specificity, prog_bar=True)        
        return loss
            
        
    def validation_step(self, batch, batch_idx) :
        return self.eval_step(batch, batch_idx, 'val')
    
    def test_setup(self, batch, batch_idx) :
        return self.eval_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self) :
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch : 0.1 **(epoch //30))
        return [optimizer], [scheduler]
    
    def setup(self, stage: Optional[str] = None) :
        # if isinstance(self.trainer.strategy, ParallelStrategy) :
        if isinstance(self.trainer.strategy, ParallelStrategy) :
            # When using a single GPU per process and per `DistributedDataParallel`, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)   
            
        if stage in (None, 'fit') :
            train_df = pd.read_csv(self.data_path + '/train.csv') 
        test_df = pd.read_csv(self.data_path + '/test.csv')
        self.train_dataset = PapsDataset(train_df, defaultpath=self.data_path, transform=train_transforms)  
        self.eval_dataset = PapsDataset(test_df, defaultpath=self.data_path, transform=test_transforms)              

        
    def train_dataloader(self) :
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True
        )
    
    def val_dataloader(self) :
        return torch.utils.data.DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True
        )
    
    def test_loader(self) :
        return val_dataloader()
            
            
if __name__ == "__main__":
    
    args = parser.parse_args()
    if torch.cuda.is_available() :
        args.accelerator = 'gpu'
        args.devices = torch.cuda.device_count()
        
    args.img_size = IMAGE_SIZE
    
    trainer_defaults = dict(
        callbacks = [
            # the PyTorch example refreshes every 10 batches
            TQDMProgressBar(refresh_rate=10),
            # save when the validation top1 accuracy improves
            # ModelCheckpoint(monitor="val_acc1", mode="max"),
            ModelCheckpoint(monitor="val_acc1", mode="max",
                            dirpath=args.saved_dir,
                            filename='paps_tunning_{epoch:02d}_{val_acc1:.2f}'),            
        ],    
        # plugins = "deepspeed_stage_2_offload",
        precision = 16,
        max_epochs = args.epochs,
        accelerator = args.accelerator, # auto, or select device, "gpu"
        devices = args.devices, # number of gpus
        logger = True,
        benchmark = True,
        strategy = "ddp",
        )
    
    model = PapsClsModel(
        data_path=args.data_path,
        arch=args.arch,
        pretrained=False,
        workers=args.workers,
        lr = args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        resume=args.resume)
    
    trainer = Trainer(**trainer_defaults)
    trainer.fit(model)            
        
        