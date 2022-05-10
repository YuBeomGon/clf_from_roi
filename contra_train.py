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
from torchmetrics import Accuracy, F1Score

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from utils.dataset import PapsDataset, ContraPapsDataset, train_transforms, val_transforms, test_transforms, contra_transforms, IMAGE_SIZE
from utils.losses import SupConLoss, FocalLoss
import custom_models
from train import *


class PapsContraModel(PapsClsModel) :
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
        # is_contra: bool = False,
    ):
        
        super(PapsContraModel, self).__init__(data_path=data_path, arch=arch, pretrained=pretrained,
                         lr=lr, momentum=momentum, weight_decay=weight_decay,
                         batch_size=batch_size, workers=workers, num_classes=num_classes)
        
        shape = self.model.fc.weight.shape
        self.contra_layer = 128
        self.model.fc = nn.Linear(shape[1], self.contra_layer)
        self.criterion = SupConLoss()
            
        print("=> creating model '{}'".format(args.arch))
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

        
    def forward(self, x) :
        return self.model(x)
    
    def contra_forward(self, batch) :
        images1, images2, targets = batch
        batch_size = targets.shape[0]
        images = torch.cat([images1, images2], dim=0)
        outputs = self(images)
        o1, o2 = torch.split(outputs, [batch_size, batch_size], dim=0)
        outputs = torch.cat([o1.unsqueeze(1), o2.unsqueeze(1)], dim=1)
        loss = self.criterion(outputs, targets)
        
        return loss
    
    def training_step(self, batch, batch_idx) :
        loss = self.contra_forward(batch)
        self.log('train_loss', loss)  
        
        #for tensorboard
        logs={"train_loss": loss}
        batch_dictionary={
            'loss':loss,
            'log':logs,
        }             

        return batch_dictionary
    
    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        # logging histograms
        # self.custom_histogram_adder()
        
        # creating log dictionary
        # self.logger.experiment.add_scalar('loss/train', avg_loss, self.current_epoch)
        tensorboard_logs = {'loss': avg_loss}
        
        epoch_dictionary={
            # required
            'loss': avg_loss,
            # for logging purposes
            'log': tensorboard_logs
        }
    
    def eval_step(self, batch, batch_idx, prefix: str) :
        loss = self.contra_forward(batch)
        self.log('val_loss', loss) 
        
        if prefix == 'val' :
            logs={"val_loss": loss}
            batch_dictionary={
                'loss':loss,
                'log':logs,
            }  
            return batch_dictionary            

        return loss
    
    def validation_step(self, batch, batch_idx) :
        return self.eval_step(batch, batch_idx, 'val')  
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        tensorboard_logs = {'loss': avg_loss}
        
        epoch_dictionary={
            # required
            'loss': avg_loss,
            # for logging purposes
            'log': tensorboard_logs
        }    
    
    def configure_optimizers(self) :
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch : 0.1 **(epoch //30))
        return [optimizer], [scheduler]
    
    def setup(self, stage: Optional[str] = None) :
        if isinstance(self.trainer.strategy, ParallelStrategy) :
            # When using a single GPU per process and per `DistributedDataParallel`, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)   
            
        if stage in (None, 'fit') :
            train_df = pd.read_csv(self.data_path + '/train.csv') 
            self.train_dataset = ContraPapsDataset(train_df, defaultpath=self.data_path, transform=contra_transforms)  
            
        test_df = pd.read_csv(self.data_path + '/test.csv')
        self.eval_dataset = ContraPapsDataset(test_df, defaultpath=self.data_path, transform=contra_transforms)
            
            
if __name__ == "__main__":
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    args = parser.parse_args()
    if torch.cuda.is_available() :
        args.accelerator = 'gpu'
        args.devices = torch.cuda.device_count()
        
    args.batch_size = args.batch_size
    args.lr = args.lr
    args.epochs = 90
    args.saved_dir = './saved_models/contra'        
        
    args.img_size = IMAGE_SIZE
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir="contra_logs/" + args.arch)
    logger_tb = TensorBoardLogger('./tuning_logs' +'/' + args.arch, name=now)
    logger_wandb = WandbLogger(project='Paps_clf_contra', name=now, mode='online') # online or disabled        
    
    trainer_defaults = dict(
        callbacks = [
            # the PyTorch example refreshes every 10 batches
            TQDMProgressBar(refresh_rate=50),
            # save when the validation top1 accuracy improves
            ModelCheckpoint(monitor="val_loss", mode="min",
                            dirpath=args.saved_dir + '/' + args.arch,
                            filename='paps-contra_{epoch:02d}_{val_loss:.2f}'),
            ModelCheckpoint(monitor="val_loss", mode="min",
                            dirpath=args.saved_dir + '/' + args.arch,
                            filename='paps-contra_best'),            
        ],    
        # plugins = "deepspeed_stage_2_offload",
        precision = 16,
        max_epochs = args.epochs,
        accelerator = args.accelerator, # auto, or select device, "gpu"
        devices = args.devices, # number of gpus
        # logger = True,
        logger = [logger_tb, logger_wandb],
        benchmark = True,
        strategy = "ddp",
        )
    
    
    model = PapsContraModel(
        data_path=args.data_path,
        arch=args.arch,
        pretrained=False,
        workers=args.workers,
        lr = args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes)
    
    trainer = Trainer(**trainer_defaults)
    trainer.fit(model)
        
        