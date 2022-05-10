#!/bin/bash
# arch='swin_t'
arch='resnet18'

docker run --gpus all -it --rm -v $PWD:/opt -v /home/Dataset/scl/patch_images:/opt/lbp_data/patch_images $@ beomgon/pl_deepspeed:paps_contra --arch $arch

docker run --gpus all -it --rm -v $PWD:/opt -v /home/Dataset/scl/patch_images:/opt/lbp_data/patch_images $@ beomgon/pl_deepspeed:paps_downstream  --arch $arch

docker run --gpus all -it --rm -v $PWD:/opt -v /home/Dataset/scl/patch_images:/opt/lbp_data/patch_images $@ beomgon/pl_deepspeed:paps_test  --arch $arch