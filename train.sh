#!/bin/bash
# arch='swin_t'
arch='resnet18'
num_workers=`grep -c processor /proc/cpuinfo`
echo $num_workers

python contra_train.py --arch $arch --workers $num_workers
python train.py --arch $arch --workers $num_workers
python test.py --arch $arch --workers $num_workers