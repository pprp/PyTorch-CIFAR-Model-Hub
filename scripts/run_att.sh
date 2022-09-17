#!/bin/bash
# 10% datasets

module load anaconda/2021.05
module load  cuda/11.1
module load cudnn/8.2.1_cuda11.x
source activate hb
export PYTHONUNBUFFERED=1

python train.py --model 'norm_resnext29_16x8d' --name "norm_8d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

# python train.py --model 'norm_resnext29_16x32d' --name "norm_32d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

# python train.py --model 'norm_resnext29_16x64d' --name "norm_64d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

# python train.py --model 'cbam_resnext29_16x8d' --name "cbam_8d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

# python train.py --model 'cbam_resnext29_16x32d' --name "cbam_32d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"

# python train.py --model 'cbam_resnext29_16x64d' --name "cbam_64d" --sched 'cosine' --epochs 200 --fast True --dataset "cifar10"



# python train.py --model 'shake_resnet26_2x64d' --cutout True  --name "shakeC" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --mixup True --name "shakeM" --sched 'cosine' --epochs 200
# python train.py --model 'shake_resnet26_2x64d' --cutout True --mixup True --name "shakeMC" --sched 'cosine' --epochs 200
