#!/bin/bash
# 30% datasets 

# python train.py --model 'norm_resnext29_16x8d' --name "norm_8d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" & \
# python train.py --model 'norm_resnext29_16x16d' --name "norm_16d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 

python train.py --model 'norm_resnext29_16x32d' --name "norm_32d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

python train.py --model 'norm_resnext29_16x64d' --name "norm_64d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

python train.py --model 'cbam_resnext29_16x8d' --name "cbam_8d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" & \
python train.py --model 'cbam_resnext29_16x16d' --name "cbam_16d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

python train.py --model 'cbam_resnext29_16x32d' --name "cbam_32d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10" 
python train.py --model 'cbam_resnext29_16x64d' --name "cbam_64d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"



# python train.py --model 'shake_resnet26_2x64d' --cutout True  --name "shakeC" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --mixup True --name "shakeM" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --cutout True --mixup True --name "shakeMC" --sched 'cosine' --epochs 200
