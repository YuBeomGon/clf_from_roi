#!/bin/bash

# build the base image
docker build --file dockerfiles/dockerfile.pl.deepspeed -t beomgon/pl_deepspeed:paps_base .

docker build --file dockerfiles/dockerfile.downstream -t beomgon/pl_deepspeed:paps_downstream .

docker build --file dockerfiles/dockerfile.sup_contra -t beomgon/pl_deepspeed:paps_contra .

docker build --file dockerfiles/dockerfile.test -t beomgon/pl_deepspeed:paps_test .

# push the image to dockhub(registry server)
docker push beomgon/pl_deepspeed:paps_base

docker push beomgon/pl_deepspeed:paps_downstream

docker push beomgon/pl_deepspeed:paps_contra

docker push beomgon/pl_deepspeed:paps_test

#docker push beomgon/pl_deepspeed:supervised_contrastive

