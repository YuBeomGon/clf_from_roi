#!/bin/bash

python contra_train.py --arch 'resnet18'
python train.py --arch 'resnet18'
python test.py --arch 'resnet18'