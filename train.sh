#!/bin/bash
# arch='swin_t'
arch='resnet18'

python contra_train.py --arch $arch
python train.py --arch $arch
python test.py --arch $arch