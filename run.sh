#docker run --gpus all -it --rm -v $PWD:/opt -v /home/Dataset/scl/patch_images:/opt/lbp_data/patch_images $@ beomgon/pl_deepspeed:paps_supervised_contra --arch 'swin_t'

docker run --gpus all -it --rm -v $PWD:/opt -v /home/Dataset/scl/patch_images:/opt/lbp_data/patch_images $@ beomgon/pl_deepspeed:paps_downstream  --arch 'swin_t'