# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/imagenet.py


import argparse
import os
from typing import Optional
import pandas as pd
from shutil import copyfile

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
from train import *
            
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
    model = model.load_from_checkpoint(checkpoint_path=args.saved_dir + '/paps_tunning_best.ckpt', strict=False)
    trainer.test(model)
        
        