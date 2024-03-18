#!/bin/bash

dataset=imagenet100

#perclass=1
#channel=128
workers=8
arch=resnet18
batch_size=128
repeat=1
epochs=0
repeat=3
miltstone1=30
milestone2=60
milestone3=80
baseclass=50
for i in $(seq 1 $repeat);
do
  for phase in 5 10 25 50
  do
    for EN in 17000 18000
    do
      python main.py \
      -a $arch \
      -j $workers \
      --data /home/dataset/dataset/ImageNet \
      --dataset $dataset \
      --wd 1e-4 \
      --epochs $epochs \
      --lr-decay-milestone $miltstone1 $milestone2 $milestone3 \
      --phase $phase \
      --baseclass $baseclass \
      --basetraining \
      --incremental \
      --Hidden $EN \
      --resume ./save_model/imagenet100/resnet18/1-10_16-23-53-70/model_best.pth.tar \
      --gpu $1
    done
  done
done
