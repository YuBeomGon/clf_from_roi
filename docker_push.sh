#!/bin/bash

# build the base image
docker build --file dockerfiles/dockerfile.pl.deepspeed -t beomgon/pl_deepspeed:base .

docker build --file dockerfiles/dockerfile.downstream -t beomgon/pl_deepspeed:downstream .

# docker build --file dockerfiles/dockerfile.sup_contra -t beomgon/pl_deepspeed:supervised_contrastive .

# push the image to dockhub(registry server)
docker push beomgon/pl_deepspeed:base

docker push beomgon/pl_deepspeed:downstream

#docker push beomgon/pl_deepspeed:supervised_contrastive

