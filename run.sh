#!/bin/bash
# arch='swin_t'
arch='resnet18'
num_workers=`grep -c processor /proc/cpuinfo`

# https://docs.docker.com/engine/reference/run/#ipc-settings---ipc
# set --ipc=host for solving out of shared memory 

docker run --gpus all -it --ipc=host --rm -v $PWD:/opt -v /home/Dataset/scl/patch_images:/opt/lbp_data/patch_images $@ beomgon/pl_deepspeed:paps_contra --arch $arch --workers $num_workers

docker run --gpus all -it --ipc=host --rm -v $PWD:/opt -v /home/Dataset/scl/patch_images:/opt/lbp_data/patch_images $@ beomgon/pl_deepspeed:paps_downstream  --arch $arch --workers $num_workers

docker run --gpus all -it --ipc=host --rm -v $PWD:/opt -v /home/Dataset/scl/patch_images:/opt/lbp_data/patch_images $@ beomgon/pl_deepspeed:paps_test  --arch $arch --workers $num_workers